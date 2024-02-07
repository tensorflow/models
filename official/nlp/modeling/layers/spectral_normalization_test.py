# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for normalization layers.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""
from absl.testing import parameterized

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import spectral_normalization

DenseLayer = tf_keras.layers.Dense(10)
Conv2DLayer = tf_keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid')


def _compute_spectral_norm(weight):
  if weight.ndim > 2:
    # Computes Conv2D via FFT transform as in [1].
    weight = np.fft.fft2(weight, weight.shape[1:3], axes=[0, 1])
  return np.max(np.linalg.svd(weight, compute_uv=False))


class NormalizationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NormalizationTest, self).setUp()
    self.num_iterations = 1000
    self.norm_multiplier = 0.95

  @parameterized.named_parameters(
      ('Dense',
       (None, 10), DenseLayer, spectral_normalization.SpectralNormalization),
      ('Conv2D', (None, 32, 32, 3), Conv2DLayer,
       spectral_normalization.SpectralNormalizationConv2D))
  def test_spec_norm_magnitude(self, input_shape, layer, norm_wrapper):
    """Tests if the weights spectral norm converges to norm_multiplier."""
    layer.build(input_shape)
    sn_layer = norm_wrapper(
        layer,
        iteration=self.num_iterations,
        norm_multiplier=self.norm_multiplier)

    # Perform normalization.
    sn_layer.build(input_shape)
    sn_layer.update_weights()
    normalized_kernel = sn_layer.layer.kernel.numpy()

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel)
    spectral_norm_expected = self.norm_multiplier
    self.assertAllClose(
        spectral_norm_computed, spectral_norm_expected, atol=1e-1)

    # Test that the normalized layer is K-Lipschitz. In particular, if the layer
    # is a function f, then ||f(x1) - f(x2)||_2 <= K * ||(x1 - x2)||_2, where K
    # is the norm multiplier.
    new_input_shape = (16,) + input_shape[1:]
    new_input = tf.random.uniform(new_input_shape)
    delta_vec = tf.random.uniform(new_input_shape)
    output1 = sn_layer(new_input)
    output2 = sn_layer(new_input + delta_vec)

    delta_input = tf.norm(tf.reshape(delta_vec, (-1,))).numpy()
    delta_output = tf.norm(tf.reshape(output2 - output1, (-1,))).numpy()
    self.assertLessEqual(delta_output, self.norm_multiplier * delta_input)


if __name__ == '__main__':
  tf.test.main()
