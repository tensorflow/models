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
"""Extracts DELF features for query images from Revisited Oxford/Paris datasets.

Note that query images are cropped before feature extraction, as required by the
evaluation protocols of these datasets.

The program checks if descriptors already exist, and skips computation for
those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
from delf import extractor

cmd_args = None

# Extensions.
_DELF_EXTENSION = '.delf'
_IMAGE_EXTENSION = '.jpg'

# To avoid PIL crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _PilLoader(path):
  """Helper function to read image with PIL.

  Args:
    path: Path to image to be loaded.

  Returns:
    PIL image in RGB format.
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Read list of query images from dataset file.
  tf.compat.v1.logging.info(
      'Reading list of query images and boxes from dataset file...')
  query_list, _, ground_truth = dataset.ReadDatasetFile(
      cmd_args.dataset_file_path)
  num_images = len(query_list)
  tf.compat.v1.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(cmd_args.delf_config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(cmd_args.output_features_dir):
    tf.io.gfile.makedirs(cmd_args.output_features_dir)

  with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
      # Initialize variables, construct DELF extractor.
      init_op = tf.compat.v1.global_variables_initializer()
      sess.run(init_op)
      extractor_fn = extractor.MakeExtractor(sess, config)

      start = time.clock()
      for i in range(num_images):
        query_image_name = query_list[i]
        input_image_filename = os.path.join(cmd_args.images_dir,
                                            query_image_name + _IMAGE_EXTENSION)
        output_feature_filename = os.path.join(
            cmd_args.output_features_dir, query_image_name + _DELF_EXTENSION)
        if tf.io.gfile.exists(output_feature_filename):
          tf.compat.v1.logging.info('Skipping %s', query_image_name)
          continue

        # Crop query image according to bounding box.
        bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
        im = np.array(_PilLoader(input_image_filename).crop(bbox))

        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = extractor_fn(im)

        feature_io.WriteToFile(output_feature_filename, locations_out,
                               feature_scales_out, descriptors_out,
                               attention_out)

      elapsed = (time.clock() - start)
      print('Processed %d query images in %f seconds' % (num_images, elapsed))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--delf_config_path',
      type=str,
      default='/tmp/delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--images_dir',
      type=str,
      default='/tmp/images',
      help="""
      Directory where dataset images are located, all in .jpg format.
      """)
  parser.add_argument(
      '--output_features_dir',
      type=str,
      default='/tmp/features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
