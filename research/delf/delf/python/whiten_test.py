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

import tensorflow as tf
import numpy as np

from delf.python import whiten


class WhitenTest(tf.test.TestCase):

  def testWhitenapply(self):
    # Testing the application of the learned whitening.
    vectors = np.array([[0.14022471, 0.96360618],
                         [0.37601032, 0.25528411]])
    # Learn whitening for the `vectors`. First element in the `vectors` is
    # viewed is the example query and the second element is the corresponding
    # positive.
    m, P = whiten.whitenlearn(vectors, [0], [1])
    # Apply the computed whitening.
    whitened_vectors = whiten.whitenapply(vectors, m, P)
    expected_whitened_vectors = np.array([[0., 9.99999000e-01],
                                          [0., -2.81240452e-13]])
    # Compare the obtained whitened vectors with the expected result.
    self.assertAllClose(whitened_vectors, expected_whitened_vectors)

  def testWhitenlearn(self):
    # Testing whitening learning function.
    X = np.array([[0.14022471, 0.96360618],
                   [0.37601032, 0.25528411]])
    # Obtain the mean descriptor vector `m` and the projection matrix `P`.
    m, P = whiten.whitenlearn(X, [0], [1])
    expected_m = np.array([[0.14022471],
                           [0.37601032]])
    expected_P = np.array([[1.18894378e+00, -1.74326044e-01],
                           [1.45071361e+04, 9.89421193e+04]])
    # Check that the both calculated values are close to the expected values.
    self.assertAllClose(m, expected_m)
    self.assertAllClose(P, expected_P)

  def testCholeskyPositiveDefinite(self):
    # Testing the Cholesky decomposition for the positive definite matrix.
    A = np.array([[1, -2j], [2j, 5]])
    L = whiten.cholesky(A)
    expected_L = np.array([[1. + 0.j, 0. + 0.j], [0. + 2.j, 1. + 0.j]])
    # Check that the expected output is obtained.
    self.assertAllClose(L, expected_L)
    # Check that the properties of the Cholesky decomposition are satisfied.
    self.assertAllClose(np.matmul(L, L.T.conj()), A)

  def testCholeskyNonPositiveDefinite(self):
    # Testing the Cholesky decomposition for a non-positive definite matrix.
    A = np.array([[1., 2.], [-2., 1.]])
    L = whiten.cholesky(A)
    expected_A = np.array([[2., -2.], [-2., 2.]])
    # Check that the properties of the Cholesky decomposition are satisfied.
    self.assertAllClose(np.matmul(L, L.T), expected_A)


if __name__ == '__main__':
  tf.test.main()
