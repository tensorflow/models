# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# https://github.com/PurdueDualityLab/tf-models/blob/master/official/vision/beta/data/create_coco_tf_record.py reference

r"""Convert raw Pix3D dataset to TFRecord format.
This scripts follows the label map decoder format and supports detection
boxes, instance masks and captions.
Example usage:
    python create_pix3d_tf_record.py --logtostderr \
      --pix3d_dir="${TRAIN_IMAGE_DIR}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""

import collections
import json
import logging
import os
import json

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

import multiprocessing as mp
from official.vision.beta.data import tfrecord_lib

flags.DEFINE_multi_string('pix3d_dir', '', 'Directory containing Pix3d.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

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

  return image

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
      output_path, coco_annotations_iter, create_tf_example, num_shards)

  logging.info('Finished writing, skipped %d annotations.', num_skipped)

def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  

  


if __name__ == '__main__':
  app.run(main)