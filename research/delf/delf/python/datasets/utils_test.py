# Lint as: python3
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
# ==============================================================================
"""Tests for dataset utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from delf.python.datasets import utils as image_loading_utils

FLAGS = flags.FLAGS


class UtilsTest(tf.test.TestCase):

  def testDefaultLoader(self):
    # Create a dummy image.
    dummy_image = np.random.rand(1024, 750, 3) * 255
    img_out = Image.fromarray(dummy_image.astype('uint8')).convert('RGB')
    filename = os.path.join(FLAGS.test_tmpdir, 'test_image.png')
    # Save the dummy image.
    img_out.save(filename)

    max_img_size = 1024
    # Load the saved dummy image.
    img = image_loading_utils.default_loader(
        filename, imsize=max_img_size, preprocess=False)

    # Make sure the values are the same before and after loading.
    self.assertAllEqual(np.array(img_out), img)

    self.assertAllLessEqual(tf.shape(img), max_img_size)

  def testDefaultLoaderWithBoundingBox(self):
    # Create a dummy image.
    dummy_image = np.random.rand(1024, 750, 3) * 255
    img_out = Image.fromarray(dummy_image.astype('uint8')).convert('RGB')
    filename = os.path.join(FLAGS.test_tmpdir, 'test_image.png')
    # Save the dummy image.
    img_out.save(filename)

    max_img_size = 1024
    # Load the saved dummy image.
    expected_size = 400
    img = image_loading_utils.default_loader(
        filename,
        imsize=max_img_size,
        bounding_box=[120, 120, 120 + expected_size, 120 + expected_size],
        preprocess=False)

    # Check that the final shape is as expected.
    self.assertAllEqual(tf.shape(img), [expected_size, expected_size, 3])


if __name__ == '__main__':
  tf.test.main()
