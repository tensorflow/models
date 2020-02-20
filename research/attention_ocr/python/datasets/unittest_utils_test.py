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

"""Tests for unittest_utils."""

import numpy as np
import io
from PIL import Image as PILImage
import tensorflow as tf

from datasets import unittest_utils


class UnittestUtilsTest(tf.test.TestCase):
  def test_creates_an_image_of_specified_shape(self):
    image, _ = unittest_utils.create_random_image('PNG', (10, 20, 3))
    self.assertEqual(image.shape, (10, 20, 3))

  def test_encoded_image_corresponds_to_numpy_array(self):
    image, encoded = unittest_utils.create_random_image('PNG', (20, 10, 3))
    pil_image = PILImage.open(io.BytesIO(encoded))
    self.assertAllEqual(image, np.array(pil_image))

  def test_created_example_has_correct_values(self):
    example_serialized = unittest_utils.create_serialized_example({
        'labels': [1, 2, 3],
        'data': [b'FAKE']
    })
    example = tf.train.Example()
    example.ParseFromString(example_serialized)
    self.assertProtoEquals("""
      features {
        feature {
          key: "labels"
           value { int64_list {
             value: 1
             value: 2
             value: 3
           }}
         }
         feature {
           key: "data"
           value { bytes_list {
             value: "FAKE"
           }}
         }
      }
    """, example)


if __name__ == '__main__':
  tf.test.main()
