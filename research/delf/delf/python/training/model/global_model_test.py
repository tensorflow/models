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
"""Tests for the GlobalFeatureNet backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import numpy as np
from PIL import Image
import tensorflow as tf

from delf.python.training.model import global_model

FLAGS = flags.FLAGS


class GlobalFeatureNetTest(tf.test.TestCase):
  """Tests for the GlobalFeatureNet backbone."""

  def testInitModel(self):
    """Testing GlobalFeatureNet initialization."""
    # Testing GlobalFeatureNet initialization.
    model_params = {'architecture': 'ResNet101', 'pooling': 'gem',
                    'whitening': False, 'pretrained': True}
    model = global_model.GlobalFeatureNet(**model_params)
    expected_meta = {'architecture': 'ResNet101', 'pooling': 'gem',
                     'whitening': False, 'outputdim': 2048}
    self.assertEqual(expected_meta, model.meta)

  def testExtractVectors(self):
    """Tests extraction of global descriptors from list."""
    # Initializing network for testing.
    model_params = {'architecture': 'ResNet101', 'pooling': 'gem',
                    'whitening': False, 'pretrained': True}
    model = global_model.GlobalFeatureNet(**model_params)

    # Number of images to be created.
    n = 2
    image_paths = []

    # Create `n` dummy images.
    for i in range(n):
      dummy_image = np.random.rand(1024, 750, 3) * 255
      img_out = Image.fromarray(dummy_image.astype('uint8')).convert('RGB')
      filename = os.path.join(FLAGS.test_tmpdir, 'test_image_{}.jpg'.format(i))
      img_out.save(filename)
      image_paths.append(filename)

    descriptors = global_model.extract_global_descriptors_from_list(
            model, image_paths, image_size=1024, bounding_boxes=None,
            scales=[1., 3.], multi_scale_power=2, print_freq=1)
    self.assertAllEqual([2048, 2], tf.shape(descriptors))

  def testExtractMultiScale(self):
    """Tests multi-scale global descriptor extraction."""
    # Initializing network for testing.
    model_params = {'architecture': 'ResNet101', 'pooling': 'gem',
                    'whitening': False, 'pretrained': True}
    model = global_model.GlobalFeatureNet(**model_params)

    input = tf.random.uniform([2, 1024, 750, 3], dtype=tf.float32, seed=0)
    descriptors = global_model.extract_multi_scale_descriptor(
            model, input, scales=[1., 3.], multi_scale_power=2)
    self.assertAllEqual([2, 2048], tf.shape(descriptors))


if __name__ == '__main__':
  tf.test.main()
