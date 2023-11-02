# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert raw COCO dataset to TFRecord format.

This scripts follows the label map decoder format and supports detection
boxes, instance masks and captions.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""

import collections
import json
import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

from pycocotools import mask
import tensorflow as tf, tf_keras

import multiprocessing as mp
from official.vision.data import tfrecord_lib


flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
flags.DEFINE_multi_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', '', 'File containing image information. '
    'Tf Examples in the output files correspond to the image '
    'info entries in this file. If this file is not provided '
    'object_annotations_file is used if present. Otherwise, '
    'caption_annotations_file is used to get image info.')
flags.DEFINE_string(
    'object_annotations_file', '', 'File containing object '
    'annotations - boxes and instance masks.')
flags.DEFINE_string('caption_annotations_file', '', 'File containing image '
                    'captions.')
flags.DEFINE_string('panoptic_annotations_file', '', 'File containing panoptic '
                    'annotations.')
flags.DEFINE_string('panoptic_masks_dir', '',
                    'Directory containing panoptic masks annotations.')
flags.DEFINE_boolean(
    'include_panoptic_masks', False, 'Whether to include category and '
    'instance masks in the result. These are required to run the PQ evaluator '
    'default: False.')
flags.DEFINE_boolean(
    'panoptic_skip_crowd', False, 'Whether to skip crowd or not for panoptic '
    'annotations. default: False.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', None,
    ('Number of parallel processes to use. '
     'If set to 0, disables multi-processing.'))


FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_VOID_LABEL = 0
_VOID_INSTANCE_ID = 0
_THING_CLASS_ID = 1
_STUFF_CLASSES_OFFSET = 90


def coco_segmentation_to_mask_png(segmentation, height, width, is_crowd):
  """Encode a COCO mask segmentation as PNG string."""
  run_len_encoding = mask.frPyObjects(segmentation, height, width)
  binary_mask = mask.decode(run_len_encoding)
  if not is_crowd:
    binary_mask = np.amax(binary_mask, axis=2)

  return tfrecord_lib.encode_mask_as_png(binary_mask)


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
  rgb_mask = tfrecord_lib.read_image(mask_path)
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
      'semantic_segmentation_mask': tfrecord_lib.encode_mask_as_png(
          semantic_segmentation_mask)
      }

  if include_panoptic_masks:
    outputs.update({
        'category_mask': tfrecord_lib.encode_mask_as_png(category_mask),
        'instance_mask': tfrecord_lib.encode_mask_as_png(instance_mask)
        })
  return outputs


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

    if include_masks:
      data['encoded_mask_png'].append(
          coco_segmentation_to_mask_png(object_annotations['segmentation'],
                                        image_height, image_width,
                                        object_annotations['iscrowd'])
      )

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
        'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(data['xmin']),
        'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(data['xmax']),
        'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(data['ymin']),
        'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(data['ymax']),
        'image/object/class/text': tfrecord_lib.convert_to_feature(
            data['category_names']
        ),
        'image/object/class/label': tfrecord_lib.convert_to_feature(
            data['category_id']
        ),
        'image/object/is_crowd': tfrecord_lib.convert_to_feature(
            data['is_crowd']
        ),
        'image/object/area': tfrecord_lib.convert_to_feature(
            data['area'], 'float_list'
        ),
    }
    if include_masks:
      feature_dict['image/object/mask'] = tfrecord_lib.convert_to_feature(
          data['encoded_mask_png']
      )

  return feature_dict, num_skipped


def encode_caption_annotations(caption_annotations):
  captions = []
  for caption_annotation in caption_annotations:
    captions.append(caption_annotation['caption'].encode('utf8'))

  return captions


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

  feature_dict = tfrecord_lib.image_info_to_feature_dict(
      image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

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
        {'image/caption': tfrecord_lib.convert_to_feature(encoded_captions)})

  if panoptic_annotation:
    segments_info = panoptic_annotation['segments_info']
    panoptic_mask_filename = os.path.join(
        panoptic_masks_dir,
        panoptic_annotation['file_name'])
    encoded_panoptic_masks = generate_coco_panoptics_masks(
        segments_info, panoptic_mask_filename, include_panoptic_masks,
        is_category_thing)
    feature_dict.update(
        {'image/segmentation/class/encoded': tfrecord_lib.convert_to_feature(
            encoded_panoptic_masks['semantic_segmentation_mask'])})

    if include_panoptic_masks:
      feature_dict.update({
          'image/panoptic/category_mask': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['category_mask']),
          'image/panoptic/instance_mask': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['instance_mask'])
            })

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example, num_annotations_skipped


def _load_object_annotations(object_annotations_file):
  """Loads object annotation JSON file."""
  with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
    obj_annotations = json.load(fid)

  images = obj_annotations['images']
  id_to_name_map = dict((element['id'], element['name']) for element in
                        obj_annotations['categories'])

  img_to_obj_annotation = collections.defaultdict(list)
  logging.info('Building bounding box index.')
  for annotation in obj_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  for image in images:
    image_id = image['id']
    if image_id not in img_to_obj_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing bboxes.', missing_annotation_count)

  return img_to_obj_annotation, id_to_name_map


def _load_caption_annotations(caption_annotations_file):
  """Loads caption annotation JSON file."""
  with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
    caption_annotations = json.load(fid)

  img_to_caption_annotation = collections.defaultdict(list)
  logging.info('Building caption index.')
  for annotation in caption_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_caption_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  images = caption_annotations['images']
  for image in images:
    image_id = image['id']
    if image_id not in img_to_caption_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing captions.', missing_annotation_count)

  return img_to_caption_annotation


def _load_panoptic_annotations(panoptic_annotations_file):
  """Loads panoptic annotation from file."""
  with tf.io.gfile.GFile(panoptic_annotations_file, 'r') as fid:
    panoptic_annotations = json.load(fid)

  img_to_panoptic_annotation = dict()
  logging.info('Building panoptic index.')
  for annotation in panoptic_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_panoptic_annotation[image_id] = annotation

  is_category_thing = dict()
  for category_info in panoptic_annotations['categories']:
    is_category_thing[category_info['id']] = category_info['isthing'] == 1

  missing_annotation_count = 0
  images = panoptic_annotations['images']
  for image in images:
    image_id = image['id']
    if image_id not in img_to_panoptic_annotation:
      missing_annotation_count += 1
  logging.info(
      '%d images are missing panoptic annotations.', missing_annotation_count)

  return img_to_panoptic_annotation, is_category_thing


def _load_images_info(images_info_file):
  with tf.io.gfile.GFile(images_info_file, 'r') as fid:
    info_dict = json.load(fid)
  return info_dict['images']


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


def _create_tf_record_from_coco_annotations(images_info_file,
                                            image_dirs,
                                            output_path,
                                            num_shards,
                                            object_annotations_file=None,
                                            caption_annotations_file=None,
                                            panoptic_masks_dir=None,
                                            panoptic_annotations_file=None,
                                            include_panoptic_masks=False,
                                            include_masks=False):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    images_info_file: JSON file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val/test annotation json
      files Eg. 'image_info_test-dev2017.json',
      'instance_annotations_train2017.json',
      'caption_annotations_train2017.json', etc.
    image_dirs: List of directories containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    panoptic_masks_dir: Directory containing panoptic masks.
    panoptic_annotations_file: JSON file containing panoptic annotations.
    include_panoptic_masks: Whether to include 'category_mask'
      and 'instance_mask', which is required by the panoptic quality evaluator.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  """

  logging.info('writing to output path: %s', output_path)

  images = _load_images_info(images_info_file)

  img_to_obj_annotation = None
  img_to_caption_annotation = None
  id_to_name_map = None
  img_to_panoptic_annotation = None
  is_category_thing = None
  if object_annotations_file:
    img_to_obj_annotation, id_to_name_map = (
        _load_object_annotations(object_annotations_file))
  if caption_annotations_file:
    img_to_caption_annotation = (
        _load_caption_annotations(caption_annotations_file))
  if panoptic_annotations_file:
    img_to_panoptic_annotation, is_category_thing = (
        _load_panoptic_annotations(panoptic_annotations_file))

  coco_annotations_iter = generate_annotations(
      images=images,
      image_dirs=image_dirs,
      panoptic_masks_dir=panoptic_masks_dir,
      img_to_obj_annotation=img_to_obj_annotation,
      img_to_caption_annotation=img_to_caption_annotation,
      img_to_panoptic_annotation=img_to_panoptic_annotation,
      is_category_thing=is_category_thing,
      id_to_name_map=id_to_name_map,
      include_panoptic_masks=include_panoptic_masks,
      include_masks=include_masks)

  num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, coco_annotations_iter, create_tf_example, num_shards,
      multiple_processes=_NUM_PROCESSES.value)

  logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
          FLAGS.caption_annotations_file), ('All annotation files are '
                                            'missing.')
  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file
  elif FLAGS.object_annotations_file:
    images_info_file = FLAGS.object_annotations_file
  else:
    images_info_file = FLAGS.caption_annotations_file

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  _create_tf_record_from_coco_annotations(images_info_file, FLAGS.image_dir,
                                          FLAGS.output_file_prefix,
                                          FLAGS.num_shards,
                                          FLAGS.object_annotations_file,
                                          FLAGS.caption_annotations_file,
                                          FLAGS.panoptic_masks_dir,
                                          FLAGS.panoptic_annotations_file,
                                          FLAGS.include_panoptic_masks,
                                          FLAGS.include_masks)


if __name__ == '__main__':
  app.run(main)
