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
"""Tests for whitening module."""

import numpy as np
import tensorflow as tf

from delf import whiten


class WhitenTest(tf.test.TestCase):

  def testApplyWhitening(self):
    # Testing the application of the learned whitening.
    vectors = np.array([[0.14022471, 0.96360618], [0.37601032, 0.25528411]])
    # Learn whitening for the `vectors`. First element in the `vectors` is
    # viewed is the example query and the second element is the corresponding
    # positive.
    mean_vector, projection = whiten.learn_whitening(vectors, [0], [1])
    # Apply the computed whitening.
    whitened_vectors = whiten.apply_whitening(vectors, mean_vector, projection)
    expected_whitened_vectors = np.array([[0., 9.99999000e-01],
                                          [0., -2.81240452e-13]])
    # Compare the obtained whitened vectors with the expected result.
    self.assertAllClose(whitened_vectors, expected_whitened_vectors)

  def testLearnWhitening(self):
    # Testing whitening learning function.
    descriptors = np.array([[0.14022471, 0.96360618], [0.37601032, 0.25528411]])
    # Obtain the mean descriptor vector and the projection matrix.
    mean_vector, projection = whiten.learn_whitening(descriptors, [0], [1])
    expected_mean_vector = np.array([[0.14022471], [0.37601032]])
    expected_projection = np.array([[1.18894378e+00, -1.74326044e-01],
                                    [1.45071361e+04, 9.89421193e+04]])
    # Check that the both calculated values are close to the expected values.
    self.assertAllClose(mean_vector, expected_mean_vector)
    self.assertAllClose(projection, expected_projection)

  def testCholeskyPositiveDefinite(self):
    # Testing the Cholesky decomposition for the positive definite matrix.
    descriptors = np.array([[1, -2j], [2j, 5]])
    output = whiten.cholesky(descriptors)
    expected_output = np.array([[1. + 0.j, 0. + 0.j], [0. + 2.j, 1. + 0.j]])
    # Check that the expected output is obtained.
    self.assertAllClose(output, expected_output)
    # Check that the properties of the Cholesky decomposition are satisfied.
    self.assertAllClose(np.matmul(output, output.T.conj()), descriptors)

  def testCholeskyNonPositiveDefinite(self):
    # Testing the Cholesky decomposition for a non-positive definite matrix.
    input_matrix = np.array([[1., 2.], [-2., 1.]])
    decomposition = whiten.cholesky(input_matrix)
    expected_output = np.array([[2., -2.], [-2., 2.]])
    # Check that the properties of the Cholesky decomposition are satisfied.
    self.assertAllClose(
        np.matmul(decomposition, decomposition.T), expected_output)


if __name__ == '__main__':
  tf.test.main()
