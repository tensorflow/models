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

import tensorflow as tf

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

  coco_annotations_iter = tfrecord_lib.generate_annotations(
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
      output_path, coco_annotations_iter, tfrecord_lib.create_tf_example,
      num_shards, multiple_processes=_NUM_PROCESSES.value)

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
