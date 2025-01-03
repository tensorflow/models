# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Times DELF/G extraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
import numpy as np
from six.moves import range
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import utils
from delf import extractor

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'delf_config_path', '/tmp/delf_config_example.pbtxt',
    'Path to DelfConfig proto text file with configuration to be used for DELG '
    'extraction. Local features are extracted if use_local_features is True; '
    'global features are extracted if use_global_features is True.')
flags.DEFINE_string('list_images_path', '/tmp/list_images.txt',
                    'Path to list of images whose features will be extracted.')
flags.DEFINE_integer('repeat_per_image', 10,
                     'Number of times to repeat extraction per image.')
flags.DEFINE_boolean(
    'binary_local_features', False,
    'Whether to binarize local features after extraction, and take this extra '
    'latency into account. This should only be used if use_local_features is '
    'set in the input DelfConfig from `delf_config_path`.')

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


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read list of images.
  print('Reading list of images...')
  image_paths = _ReadImageList(FLAGS.list_images_path)
  num_images = len(image_paths)
  print(f'done! Found {num_images} images')

  # Load images in memory.
  print('Loading images, %d times per image...' % FLAGS.repeat_per_image)
  im_array = []
  for filename in image_paths:
    im = np.array(utils.RgbLoader(filename))
    for _ in range(FLAGS.repeat_per_image):
      im_array.append(im)
  np.random.shuffle(im_array)
  print('done!')

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(FLAGS.delf_config_path, 'r') as f:
    text_format.Parse(f.read(), config)

  extractor_fn = extractor.MakeExtractor(config)

  start = time.time()
  for i, im in enumerate(im_array):
    if i == 0:
      print('Starting to extract DELF features from images...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print(f'Processing image {i} out of {len(im_array)}, last '
            f'{_STATUS_CHECK_ITERATIONS} images took {elapsed} seconds,'
            f'ie {elapsed/_STATUS_CHECK_ITERATIONS} secs/image.')
      start = time.time()

    # Extract and save features.
    extracted_features = extractor_fn(im)

    # Binarize local features, if desired (and if there are local features).
    if (config.use_local_features and FLAGS.binary_local_features and
        extracted_features['local_features']['attention'].size):
      packed_descriptors = np.packbits(
          extracted_features['local_features']['descriptors'] > 0, axis=1)


if __name__ == '__main__':
  app.run(main)
