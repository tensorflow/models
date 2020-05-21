# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

from six.moves import range
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf import extractor

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.io.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Read list of images.
  tf.compat.v1.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  tf.compat.v1.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(cmd_args.config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(cmd_args.output_dir):
    tf.io.gfile.makedirs(cmd_args.output_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.compat.v1.train.string_input_producer(
        image_paths, shuffle=False)
    reader = tf.compat.v1.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.io.decode_jpeg(value, channels=3)

    with tf.compat.v1.Session() as sess:
      init_op = tf.compat.v1.global_variables_initializer()
      sess.run(init_op)

      extractor_fn = extractor.MakeExtractor(sess, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          tf.compat.v1.logging.info(
              'Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.compat.v1.logging.info(
              'Processing image %d out of %d, last %d '
              'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
              elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0] + _DELF_EXT
        out_desc_fullpath = os.path.join(cmd_args.output_dir, out_desc_filename)
        if tf.io.gfile.exists(out_desc_fullpath):
          tf.compat.v1.logging.info('Skipping %s', image_paths[i])
          continue

        # Extract and save features.
        extracted_features = extractor_fn(im)
        locations_out = extracted_features['local_features']['locations']
        descriptors_out = extracted_features['local_features']['descriptors']
        feature_scales_out = extracted_features['local_features']['scales']
        attention_out = extracted_features['local_features']['attention']

        feature_io.WriteToFile(out_desc_fullpath, locations_out,
                               feature_scales_out, descriptors_out,
                               attention_out)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
