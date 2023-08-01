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

import math
from tqdm import tqdm
import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

import multiprocessing as mp
from official.vision.data import tfrecord_lib

flags.DEFINE_string('numpy_path', './dataset/samples', 'Directory containing dataset.')
flags.DEFINE_string('output_dir', './tfrecord', 'Path to output file')
flags.DEFINE_string('output_prefix', 'train', 'Path to output file')
flags.DEFINE_integer('noise', 8, 'Directory containing dataset.')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def _convert_to_example(sample):
  """Builds an Example proto for an ImageNet example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto

  """
  #print(sample['list_len'])
  #print(sample['list_len'].dtype)
  feature_dict = {
    "sat_roi": tfrecord_lib.convert_to_feature(sample['sat'], 'int64_list'),
    "label_masks_roi": tfrecord_lib.convert_to_feature(sample['label_masks'], 'int64_list'),
    "historical_roi": tfrecord_lib.convert_to_feature(sample['historical_ROI'], 'int64_list'),
    "gt_probs": tfrecord_lib.convert_to_feature(sample['gt_probs'], 'float_list'),
    "gt_coords": tfrecord_lib.convert_to_feature(sample['gt_coords'], 'float_list'),
    "list_len": tfrecord_lib.convert_to_feature(sample['list_len'], 'int64'),
    "gt_masks": tfrecord_lib.convert_to_feature(sample['gt_masks'], 'int64_list')
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def _process_image_files_batch(output_file, filenames):
  """Processes and saves a list of images as TFRecords.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set.
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: map of string to integer; id for all synset labels.

  """
  writer = tf.io.TFRecordWriter(output_file)

  #for filename in filenames:
  for i, filename in enumerate(tqdm(filenames)):
    sample = np.load(filename,allow_pickle=True)
    example = _convert_to_example(sample)
    writer.write(example.SerializeToString())

  writer.close()

def main(_):
  data_path = f'{FLAGS.numpy_path}/noise_{FLAGS.noise}'
  data_list = os.listdir(data_path)
  data_list = [os.path.join(data_path,x) for x in data_list]
  #print(len(data_list)) # 420,000
  #print(data_list[0])
  chunksize = int(math.ceil(len(data_list) / FLAGS.num_shards))
  #print(chunksize)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  for shard in range(FLAGS.num_shards):
    chunk_files = data_list[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(
        FLAGS.output_dir,
        '%s-noise-%d-%.5d-of-%.5d' % (FLAGS.output_prefix, FLAGS.noise,
                                      shard, FLAGS.num_shards))
    _process_image_files_batch(output_file, chunk_files)
    logging.info('Finished writing file: %s', output_file)

if __name__ == '__main__':
  app.run(main)