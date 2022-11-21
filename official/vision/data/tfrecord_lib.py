# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for creating TFRecord datasets."""

import hashlib
import io
import itertools
import os

from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from pycocotools import mask

import tensorflow as tf

import multiprocessing as mp

flags.DEFINE_boolean(
    'panoptic_skip_crowd', False, 'Whether to skip crowd or not for panoptic '
    'annotations. default: False.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_VOID_LABEL = 0
_VOID_INSTANCE_ID = 0
_THING_CLASS_ID = 1
_STUFF_CLASSES_OFFSET = 90


LOG_EVERY = 100


def coco_segmentation_to_mask_png(segmentation, height, width, is_crowd):
  """Encode a COCO mask segmentation as PNG string."""
  run_len_encoding = mask.frPyObjects(segmentation, height, width)
  binary_mask = mask.decode(run_len_encoding)
  if not is_crowd:
    binary_mask = np.amax(binary_mask, axis=2)

  return encode_mask_as_png(binary_mask)


def convert_to_feature(value, value_type=None):
  """Converts the given python object to a tf.train.Feature.

  Args:
    value: int, float, bytes or a list of them.
    value_type: optional, if specified, forces the feature to be of the given
      type. Otherwise, type is inferred automatically. Can be one of
      ['bytes', 'int64', 'float', 'bytes_list', 'int64_list', 'float_list']

  Returns:
    feature: A tf.train.Feature object.
  """

  if value_type is None:

    element = value[0] if isinstance(value, list) else value

    if isinstance(element, bytes):
      value_type = 'bytes'

    elif isinstance(element, (int, np.integer)):
      value_type = 'int64'

    elif isinstance(element, (float, np.floating)):
      value_type = 'float'

    else:
      raise ValueError('Cannot convert type {} to feature'.
                       format(type(element)))

    if isinstance(value, list):
      value_type = value_type + '_list'

  if value_type == 'int64':
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  elif value_type == 'int64_list':
    value = np.asarray(value).astype(np.int64).reshape(-1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  elif value_type == 'float':
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  elif value_type == 'float_list':
    value = np.asarray(value).astype(np.float32).reshape(-1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  elif value_type == 'bytes':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  elif value_type == 'bytes_list':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  else:
    raise ValueError('Unknown value_type parameter - {}'.format(value_type))


def image_info_to_feature_dict(height, width, filename, image_id,
                               encoded_str, encoded_format):
  """Convert image information to a dict of features."""

  key = hashlib.sha256(encoded_str).hexdigest()

  return {
      'image/height': convert_to_feature(height),
      'image/width': convert_to_feature(width),
      'image/filename': convert_to_feature(filename.encode('utf8')),
      'image/source_id': convert_to_feature(str(image_id).encode('utf8')),
      'image/key/sha256': convert_to_feature(key.encode('utf8')),
      'image/encoded': convert_to_feature(encoded_str),
      'image/format': convert_to_feature(encoded_format.encode('utf8')),
  }


def read_image(image_path):
  pil_image = Image.open(image_path)
  return np.asarray(pil_image)


def encode_mask_as_png(binary_mask):
  pil_image = Image.fromarray(binary_mask)
  output_io = io.BytesIO()
  pil_image.save(output_io, format='PNG')
  return output_io.getvalue()


def encode_caption_annotations(caption_annotations):
  captions = []
  for caption_annotation in caption_annotations:
    captions.append(caption_annotation['caption'].encode('utf8'))

  return captions


def generate_coco_panoptics_masks(segments_info, mask_path,
                                  include_panoptic_masks,
                                  is_category_thing):
  """Creates masks for panoptic segmentation task.

  Args:
    segments_info: a list of dicts, where each dict has keys: [u'id',
      u'category_id', u'area', u'bbox', u'iscrowd'], detailing information for
      each segment in the panoptic mask.
    mask_path: path to the panoptic mask.
    include_panoptic_masks: bool, when set to True, category and instance
      masks are included in the outputs. Set this to True, when using
      the Panoptic Quality evaluator.
    is_category_thing: a dict with category ids as keys and, 0/1 as values to
      represent "stuff" and "things" classes respectively.

  Returns:
    A dict with keys: [u'semantic_segmentation_mask', u'category_mask',
      u'instance_mask']. The dict contains 'category_mask' and 'instance_mask'
      only if `include_panoptic_eval_masks` is set to True.
  """
  rgb_mask = read_image(mask_path)
  r, g, b = np.split(rgb_mask, 3, axis=-1)

  # decode rgb encoded panoptic mask to get segments ids
  # refer https://cocodataset.org/#format-data
  segments_encoded_mask = (r + g * 256 + b * (256**2)).squeeze()

  semantic_segmentation_mask = np.ones_like(
      segments_encoded_mask, dtype=np.uint8) * _VOID_LABEL
  if include_panoptic_masks:
    category_mask = np.ones_like(
        segments_encoded_mask, dtype=np.uint8) * _VOID_LABEL
    instance_mask = np.ones_like(
        segments_encoded_mask, dtype=np.uint8) * _VOID_INSTANCE_ID

  for idx, segment in enumerate(segments_info):
    segment_id = segment['id']
    category_id = segment['category_id']
    is_crowd = segment['iscrowd']
    if FLAGS.panoptic_skip_crowd and is_crowd:
      continue
    if is_category_thing[category_id]:
      encoded_category_id = _THING_CLASS_ID
      instance_id = idx + 1
    else:
      encoded_category_id = category_id - _STUFF_CLASSES_OFFSET
      instance_id = _VOID_INSTANCE_ID

    segment_mask = (segments_encoded_mask == segment_id)
    semantic_segmentation_mask[segment_mask] = encoded_category_id

    if include_panoptic_masks:
      category_mask[segment_mask] = category_id
      instance_mask[segment_mask] = instance_id

  outputs = {
      'semantic_segmentation_mask': encode_mask_as_png(
          semantic_segmentation_mask)
      }

  if include_panoptic_masks:
    outputs.update({
        'category_mask': encode_mask_as_png(category_mask),
        'instance_mask': encode_mask_as_png(instance_mask)
        })
  return outputs


def create_tf_example(image,
                      image_dirs,
                      panoptic_masks_dir=None,
                      bbox_annotations=None,
                      id_to_name_map=None,
                      caption_annotations=None,
                      panoptic_annotation=None,
                      is_category_thing=None,
                      include_panoptic_masks=False,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    image_dirs: list of directories containing the image files.
    panoptic_masks_dir: `str` of the panoptic masks directory.
    bbox_annotations:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    id_to_name_map: a dict mapping category IDs to string names.
    caption_annotations:
      list of dict with keys: [u'id', u'image_id', u'str'].
    panoptic_annotation: dict with keys: [u'image_id', u'file_name',
      u'segments_info']. Where the value for segments_info is a list of dicts,
      with each dict containing information for a single segment in the mask.
    is_category_thing: `bool`, whether it is a category thing.
    include_panoptic_masks: `bool`, whether to include panoptic masks.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.

  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG,
      does not exist, or is not unique across image directories.
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  if len(image_dirs) > 1:
    full_paths = [os.path.join(image_dir, filename) for image_dir in image_dirs]
    full_existing_paths = [p for p in full_paths if tf.io.gfile.exists(p)]
    if not full_existing_paths:
      raise ValueError(
          '{} does not exist across image directories.'.format(filename))
    if len(full_existing_paths) > 1:
      raise ValueError(
          '{} is not unique across image directories'.format(filename))
    full_path, = full_existing_paths
  # If there is only one image directory, it's not worth checking for existence,
  # since trying to open the file will raise an informative error message if it
  # does not exist.
  else:
    image_dir, = image_dirs
    full_path = os.path.join(image_dir, filename)

  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  feature_dict = image_info_to_feature_dict(
      image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

  feature_dict_len = len(feature_dict)
  num_annotations_skipped = 0
  if bbox_annotations:
    box_feature_dict, num_skipped = bbox_annotations_to_feature_dict(
        bbox_annotations, image_height, image_width, id_to_name_map,
        include_masks)
    num_annotations_skipped += num_skipped
    feature_dict.update(box_feature_dict)

  if caption_annotations:
    encoded_captions = encode_caption_annotations(caption_annotations)
    feature_dict.update(
        {'image/caption': convert_to_feature(encoded_captions)})

  if panoptic_annotation:
    segments_info = panoptic_annotation['segments_info']
    panoptic_mask_filename = os.path.join(
        panoptic_masks_dir,
        panoptic_annotation['file_name'])
    encoded_panoptic_masks = generate_coco_panoptics_masks(
        segments_info, panoptic_mask_filename, include_panoptic_masks,
        is_category_thing)
    feature_dict.update(
        {'image/segmentation/class/encoded': convert_to_feature(
            encoded_panoptic_masks['semantic_segmentation_mask'])})

    if include_panoptic_masks:
      feature_dict.update({
          'image/panoptic/category_mask': convert_to_feature(
              encoded_panoptic_masks['category_mask']),
          'image/panoptic/instance_mask': convert_to_feature(
              encoded_panoptic_masks['instance_mask'])})

  if feature_dict_len == len(feature_dict):
    example = None
  else:
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

  return example, num_annotations_skipped


def coco_annotations_to_lists(bbox_annotations, id_to_name_map,
                              image_height, image_width, include_masks):
  """Converts COCO annotations to feature lists."""

  data = dict((k, list()) for k in
              ['xmin', 'xmax', 'ymin', 'ymax', 'is_crowd',
               'category_id', 'category_names', 'area'])
  if include_masks:
    data['encoded_mask_png'] = []

  num_annotations_skipped = 0

  for object_annotations in bbox_annotations:
    (x, y, width, height) = tuple(object_annotations['bbox'])

    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    data['xmin'].append(float(x) / image_width)
    data['xmax'].append(float(x + width) / image_width)
    data['ymin'].append(float(y) / image_height)
    data['ymax'].append(float(y + height) / image_height)
    data['is_crowd'].append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    data['category_id'].append(category_id)
    data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
    data['area'].append(object_annotations['area'])

  return data, num_annotations_skipped


def bbox_annotations_to_feature_dict(
    bbox_annotations, image_height, image_width, id_to_name_map, include_masks):
  """Convert COCO annotations to an encoded feature dict."""

  data, num_skipped = coco_annotations_to_lists(
      bbox_annotations, id_to_name_map, image_height, image_width,
      include_masks)
  feature_dict = {}
  if len(bbox_annotations) != num_skipped:
    feature_dict = {
        'image/object/bbox/xmin':
            convert_to_feature(data['xmin']),
        'image/object/bbox/xmax':
            convert_to_feature(data['xmax']),
        'image/object/bbox/ymin':
            convert_to_feature(data['ymin']),
        'image/object/bbox/ymax':
            convert_to_feature(data['ymax']),
        'image/object/class/text':
            convert_to_feature(data['category_names']),
        'image/object/class/label':
            convert_to_feature(data['category_id']),
        'image/object/is_crowd':
            convert_to_feature(data['is_crowd']),
        'image/object/area':
            convert_to_feature(data['area'], 'float_list')
    }
    if include_masks:
      feature_dict['image/object/mask'] = (
          convert_to_feature(data['encoded_mask_png']))

  return feature_dict, num_skipped


def generate_annotations(images, image_dirs,
                         panoptic_masks_dir=None,
                         img_to_obj_annotation=None,
                         img_to_caption_annotation=None,
                         img_to_panoptic_annotation=None,
                         is_category_thing=None,
                         id_to_name_map=None,
                         include_panoptic_masks=False,
                         include_masks=False):
  """Generator for COCO annotations."""
  for image in images:
    object_annotation = (img_to_obj_annotation.get(image['id'], None) if
                         img_to_obj_annotation else None)

    caption_annotaion = (img_to_caption_annotation.get(image['id'], None) if
                         img_to_caption_annotation else None)

    panoptic_annotation = (img_to_panoptic_annotation.get(image['id'], None) if
                           img_to_panoptic_annotation else None)
    yield (image, image_dirs, panoptic_masks_dir, object_annotation,
           id_to_name_map, caption_annotaion, panoptic_annotation,
           is_category_thing, include_panoptic_masks, include_masks)


def write_tf_record_dataset(output_path, annotation_iterator,
                            process_func, num_shards,
                            multiple_processes=None, unpack_arguments=True):
  """Iterates over annotations, processes them and writes into TFRecords.

  Args:
    output_path: The prefix path to create TF record files.
    annotation_iterator: An iterator of tuples containing details about the
      dataset.
    process_func: A function which takes the elements from the tuples of
      annotation_iterator as arguments and returns a tuple of (tf.train.Example,
      int). The integer indicates the number of annotations that were skipped.
    num_shards: int, the number of shards to write for the dataset.
    multiple_processes: integer, the number of multiple parallel processes to
      use.  If None, uses multi-processing with number of processes equal to
      `os.cpu_count()`, which is Python's default behavior. If set to 0,
      multi-processing is disabled.
      Whether or not to use multiple processes to write TF Records.
    unpack_arguments:
      Whether to unpack the tuples from annotation_iterator as individual
        arguments to the process func or to pass the returned value as it is.

  Returns:
    num_skipped: The total number of skipped annotations.
  """

  writers = [
      tf.io.TFRecordWriter(
          output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards))
      for i in range(num_shards)
  ]

  total_num_annotations_skipped = 0

  if multiple_processes is None or multiple_processes > 0:
    pool = mp.Pool(
        processes=multiple_processes)
    if unpack_arguments:
      tf_example_iterator = pool.starmap(process_func, annotation_iterator)
    else:
      tf_example_iterator = pool.imap(process_func, annotation_iterator)
  else:
    if unpack_arguments:
      tf_example_iterator = itertools.starmap(process_func, annotation_iterator)
    else:
      tf_example_iterator = map(process_func, annotation_iterator)

  for idx, (tf_example, num_annotations_skipped) in enumerate(
      tf_example_iterator):
    if idx % LOG_EVERY == 0:
      logging.info('On image %d', idx)

    total_num_annotations_skipped += num_annotations_skipped
    if tf_example:
      writers[idx % num_shards].write(tf_example.SerializeToString())

  if multiple_processes is None or multiple_processes > 0:
    pool.close()
    pool.join()

  for writer in writers:
    writer.close()

  logging.info('Finished writing, skipped %d annotations.',
               total_num_annotations_skipped)
  return total_num_annotations_skipped


def check_and_make_dir(directory):
  """Creates the directory if it doesn't exist."""
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)
