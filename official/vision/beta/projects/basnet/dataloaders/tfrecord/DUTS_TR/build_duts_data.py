# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Converts DUTS-TR data to TFRecord file format with Example protos.

DUTS-TR dataset is expected to have the following directory structure:

  + DUTS_TR
    - build_data.py
    - build_duts_data.py (current working directory).
    + tfrecord

Image folder:
  ~/DUTS-TR/DUTS-TR-Image-Flip (augmented by hflipping to 21106 images)

Mask folder:
  ~/DUTS-TR/DUTS-TR-Mask-Flip (augmented by hflipping to 21106 images)


This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('image_folder',
                           './DUTS-TR/DUTS-TR-Image',
                           'Folder containing images.')

tf.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './DUTS-TR/DUTS-TR-Mask',
    'Folder containing semantic segmentation annotations.')

tf.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 4


def _convert_dataset():
  """Converts the specified dataset split to TFRecord format.


  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = 'DUTS-TR-hflip'
  sys.stdout.write('Processing ' + dataset)

  filenames_ext = os.listdir(FLAGS.image_folder)
  print("filenames_ext")
  print(filenames_ext)


  filenames = []
  for data in filenames_ext:
    filenames.append(os.path.splitext(data)[0])
  print("filenames")
  print(filenames)

  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.gfile.GFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  _convert_dataset()


if __name__ == '__main__':
  tf.app.run()
