# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import attention


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class AttentionLayerTest(keras_parameterized.TestCase):

  def test_non_masked_attention(self):
    """Test that the attention layer can be created without a mask tensor."""
    test_layer = attention.Attention(num_heads=12, head_size=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    from_tensor = tf.keras.Input(shape=(40, 80))
    to_tensor = tf.keras.Input(shape=(20, 80))
    output = test_layer([from_tensor, to_tensor])
    self.assertEqual(output.shape.as_list(), [None, 40, 12, 64])

  def test_non_masked_self_attention(self):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = attention.Attention(num_heads=12, head_size=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    from_tensor = tf.keras.Input(shape=(40, 80))
    output = test_layer([from_tensor, from_tensor])
    self.assertEqual(output.shape.as_list(), [None, 40, 12, 64])

  def test_masked_attention(self):
    """Test with a mask tensor."""
    test_layer = attention.Attention(num_heads=2, head_size=2)
    # Create a 3-dimensional input (the first dimension is implicit).
    from_tensor = tf.keras.Input(shape=(4, 8))
    to_tensor = tf.keras.Input(shape=(2, 8))
    mask_tensor = tf.keras.Input(shape=(4, 2))
    output = test_layer([from_tensor, to_tensor, mask_tensor])

    # Create a model containing the test layer.
    model = tf.keras.Model([from_tensor, to_tensor, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((3, 4, 8))
    to_data = 10 * np.random.random_sample((3, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(2, size=(3, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are masked).
    null_mask_data = np.ones((3, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = attention.Attention(
        num_heads=12,
        head_size=64,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    from_tensor = tf.keras.Input(shape=(40, 80))
    output = test_layer([from_tensor, from_tensor])
    self.assertEqual(output.shape.as_list(), [None, 40, 12, 64])


if __name__ == '__main__':
  tf.test.main()
