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

r"""Convert raw cityscale dataset to TFRecord format.
This scripts follows the label map decoder format and supports detection
boxes, instance masks and captions.
Example usage:
    python create_cityscale_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""

import shutil
from tqdm import tqdm 
import argparse
from sampler import Sampler
import cv2
# -----
import json
import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

import multiprocessing as mp
from official.vision.data import tfrecord_lib

flags.DEFINE_string('dataroot', './dataset', 'Directory containing dataset.')
flags.DEFINE_string('output_file_prefix', './tfrecord/train', 'Path to output file')
flags.DEFINE_integer('roi_size', 128, 'ROI size for sampling')
flags.DEFINE_integer('image_size', 2048, 'Input image size')
flags.DEFINE_integer('edge_move_ahead_length', 30, 'Edge move length')
flags.DEFINE_integer('num_queries', 10, 'Number of queries')
flags.DEFINE_integer('noise', 8, 'Noise parameter')
flags.DEFINE_integer('max_num_frame', 10000, 'Maximum frame for an image')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', 0,
    ('Number of parallel processes to use. '
     'If set to 0, disables multi-processing.'))

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def create_tf_example(sat_ROI, label_masks_ROI, historical_ROI,
                      gt_probs,gt_coords, list_len, gt_masks):
  """Converts image and annotations to a tf.Example proto.
  Args:
    sat_ROI: [roi_size, roi_size, 3]
    label_masks_ROI: [roi_size, roi_size, 2]
    historical_ROI: [roi_size, roi_size]
    gt_probs: [num_queries]
    gt_coords: [num_queries, 2]
    list_len: int
    gt_masks: [roi_size, roi_size, num_queries]
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  """
  feature_dict = {
    "sat_roi": tfrecord_lib.convert_to_feature(sat_ROI, 'int64_list'),
    "label_masks_roi": tfrecord_lib.convert_to_feature(label_masks_ROI, 'int64_list'),
    "historical_roi": tfrecord_lib.convert_to_feature(historical_ROI, 'int64_list'),
    "gt_probs": tfrecord_lib.convert_to_feature(gt_probs, 'float_list'),
    "gt_coords": tfrecord_lib.convert_to_feature(gt_coords, 'float_list'),
    "list_len": tfrecord_lib.convert_to_feature(list_len),
    "gt_masks": tfrecord_lib.convert_to_feature(gt_masks, 'int64_list')
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

  return example, 0

def generate_samples(dataroot, image_size, roi_size, num_queries,
                     noise, edge_move_ahead_length, max_num_frame):
  # iterate training data
  with open(f'./{dataroot}/data_split.json','r') as jf:
    tile_list = json.load(jf)['train']

  for i, tile_name in enumerate(tqdm(tile_list)):
    sampler = Sampler(dataroot, image_size, roi_size, num_queries,
                      noise, edge_move_ahead_length, tile_name)
    while 1:
      if sampler.finish_current_image:
        break
      # crop
      v_current = sampler.current_coord.copy()
      sat_ROI, label_masks_ROI ,historical_ROI = sampler.crop_ROI(sampler.current_coord)
      # vertices in the next step
      v_nexts, ahead_segments = sampler.step_expert_BC_sampler()
      # save training sample
      gt_probs, gt_coords, list_len = sampler.calcualte_label(v_current,v_nexts)

      gt_masks = np.zeros((roi_size, roi_size, num_queries))
      kernel = np.ones((4,4), np.uint8)
      for ii,segment in enumerate(ahead_segments):
        for v in segment:
          try:
            gt_masks[v[1],v[0],ii] = 255
          except:
            print(segment)
            raise Exception
        gt_masks[:,:,ii] = cv2.dilate(gt_masks[:,:,ii], kernel, iterations=1)

      yield (sat_ROI.astype(np.int64), label_masks_ROI.astype(np.uint8),
             historical_ROI.astype(np.uint8), gt_probs, gt_coords, list_len,
             gt_masks.astype(np.uint8))
      if sampler.step_counter > max_num_frame:
        break

def main(_):
  assert FLAGS.dataroot, '`dataroot` missing.'
  prefix = f'./{FLAGS.output_file_prefix}-noise-{FLAGS.noise}'
  directory = os.path.dirname(prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  cityscale_annotations_iter = generate_samples(FLAGS.dataroot,
                                                FLAGS.image_size,
                                                FLAGS.roi_size,
                                                FLAGS.num_queries,
                                                FLAGS.noise,
                                                FLAGS.edge_move_ahead_length,
                                                FLAGS.max_num_frame)
  prefix = f'./{FLAGS.output_file_prefix}-noise-{FLAGS.noise}'
  num_skipped = tfrecord_lib.write_tf_record_dataset(
      prefix, cityscale_annotations_iter, create_tf_example, FLAGS.num_shards,
      multiple_processes=_NUM_PROCESSES.value)

if __name__ == '__main__':
  app.run(main)
