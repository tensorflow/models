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

import math
import os
import random
import stat
import sys

import tensorflow as tf
from tensorflow.python.framework import test_util
import activations
import skeleton_pb2

import numpy as np
from numpy.polynomial.polynomial import polyval

class TestActivations(test_util.TensorFlowTestCase):

  def testActivation(self):
    act = activations.CreateActivation(skeleton_pb2.RELU)
    self.assertAlmostEqual(act.Dual(1.0), 1.0)
    self.assertAlmostEqual(act.HermiteDual(1.0), 1.0)
    self.assertAlmostEqual(act.HermiteDual(0.5), act.Dual(0.5), delta=0.000005)

    act.SetCoeffs(1000)
    x = np.arange(-1, 1, 0.001)
    y_act = math.sqrt(2) * np.maximum(x, 0)
    poly_coeffs = activations.HermiteToPoly(act.hermite_coeffs)
    y = polyval(x, poly_coeffs)
    # Accurate to within about 1%
    self.assertAlmostEqual(np.absolute(y - y_act).max(), 0, delta=0.015)


    act.SetCoeffs(10000)
    x = np.arange(-1, 1, 0.001)
    self.assertAlmostEqual(np.absolute(act.Dual(x) - act.HermiteDual(x)).max(), 0)


  def testPolynomialActivation(self):
    act = activations.CreateActivation(skeleton_pb2.POLYNOMIAL)
    params = skeleton_pb2.ActivationParams()
    s = math.sqrt(30)
    params.coefficients.extend([1/s, 2/s, 3/s, 4/s])
    act.SetParams(params)
    self.assertAlmostEqual(
        act.poly_coeffs[0],
        params.coefficients[0] - params.coefficients[2] / math.sqrt(2.0))
    self.assertAlmostEqual(
        act.poly_coeffs[1],
        params.coefficients[1] - params.coefficients[3] * 3 / math.sqrt(6.0))
    self.assertAlmostEqual(act.poly_coeffs[2],
                           params.coefficients[2] / math.sqrt(2.0))
    self.assertAlmostEqual(act.poly_coeffs[3],
                           params.coefficients[3] / math.sqrt(6.0))
    for i in range(len(params.coefficients)):
      self.assertAlmostEqual(act.dual_coeffs[i],
                             params.coefficients[i] * params.coefficients[i])

  def testSineActivation(self):
    act = activations.CreateActivation(skeleton_pb2.SINE)
    params = skeleton_pb2.ActivationParams()
    a = params.scale = 1.0
    act.SetParams(params)
    norm = np.sqrt(2.0 / (1.0 - math.exp(- 2 * a * a)))
    x = np.arange(-4, 4, 0.001)
    y_act = norm * np.sin(a * x)
    self.assertAlmostEqual(np.absolute(act.Act(x) - y_act).max(), 0)

    # Check that the series represented by the Hermite poly coefficients is the
    # same function as the activation function
    act.SetCoeffs(50)
    poly_coeffs = activations.HermiteToPoly(act.hermite_coeffs)
    y = polyval(x, poly_coeffs)
    self.assertAlmostEqual(np.absolute(y - y_act).max(), 0)


  def testCosineActivation(self):
    act = activations.CreateActivation(skeleton_pb2.COSINE)
    params = skeleton_pb2.ActivationParams()
    a = params.scale = np.pi / 2.0
    act.SetParams(params)
    norm = np.sqrt(2.0 * math.exp(a * a) /
                   (math.exp(a * a) + math.exp(-a * a)))
    x = np.arange(-4, 4, 0.001)
    y_act = norm * np.cos(a * x)

    self.assertAlmostEqual(np.absolute(act.Act(x) - y_act).max(), 0)

    # Check that the series represented by the Hermite poly coefficients is the
    # same function as the activation function
    act.SetCoeffs(50)
    poly_coeffs = activations.HermiteToPoly(act.hermite_coeffs)
    y = polyval(x, poly_coeffs)
    self.assertAlmostEqual(np.absolute(y - y_act).max(), 0)


if __name__ == '__main__':
  tf.test.main()
