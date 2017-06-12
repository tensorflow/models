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
import os.path
import subprocess
import sys

import tensorflow as tf

import numpy as np
from numpy import linalg as LA
import scipy.special

import random
from types import TypeType
import utils
import skeleton_pb2
from tensorflow.python.framework import function
from numpy.polynomial.polynomial import polyval
import scipy.misc

DEFAULT_COEFF_SIZE = 10

def GenerateHermiteCoeffs(k):
  if k == 0:
    return np.ones((1,1))
  coeffs = np.zeros((k + 1, k + 1))
  coeffs[0, 0] = 1
  coeffs[1, 1] = 1
  for n in range(1, k):
    coeffs[n + 1, 1:k + 1] = coeffs[n, 0:k] / math.sqrt(n + 1)
    coeffs[n + 1] -= math.sqrt(n / (n + 1.0)) * coeffs[n - 1]
  return coeffs


def HermiteToPoly(hermite_coeffs):
  return np.dot(hermite_coeffs, GenerateHermiteCoeffs(hermite_coeffs.size - 1))


def EvaluatePoly_TF(x, coeffs):
  series = []
  for i in xrange(1, coeffs.size):
    if coeffs[i] == 0:
      continue
    if i == 1:
      series.append(x * coeffs[i])
    else:
      series.append(tf.pow(x, i) * coeffs[i])
  return tf.add(tf.add_n(series), coeffs[0])


def CreateActivation(name):
  for (i, j) in globals().iteritems():
    if isinstance(j, TypeType) and issubclass(j, Activation) and j.name == name:
      return j()
  raise ValueError('Activation is not defined - "%s".' %
                   skeleton_pb2.ActivationType.Name(name))


class Activation(object):

  name = None

  def __init__(self):
    # coefficients of the activation in the standard polynomial basis
    self.poly_coeffs = np.zeros(0)
    # coefficients of the activation in hermite basis Act(x) = Sum_i a_i h_i(x)
    self.hermite_coeffs = np.zeros(0)
    # coefficients of the dual activation in standard basis
    # Act(rho) = Sum_i b_i rho^i
    # Note that b_i = a_i^2, where a_i is the i-th coeff in the Hermite basis
    self.dual_coeffs = np.zeros(0)

  def HermiteDual(self, rho):
    return polyval(rho, self.dual_coeffs)

  def HermiteDual_tf(self, rho):
    """ Hermite polynomial approximation for the dual activation.
    Faster than the exact dual activation, but sometimes less accurate. """
    return EvaluatePoly_TF(rho, self.dual_coeffs)

  def RandomDistr(self):
    """ Generate a random number between 0 and n, using the dual_coeffs as
    the discrete distribution"""
    return np.random.choice(self.dual_coeffs.size, 1, p=self.dual_coeffs)

  def RemoveDualBias(self):
    """ Set the constant term in the dual coefficients to zero, and normalize.
    This is useful for generating random features - we get more dense features.

    WARNING: This breaks the mathematical correspondence between the activation
    and the dual activation.  Also we are only updating the coefficients, if
    the Dual or Dual_tf functions are used to compute the dual, they are not
    updated for now. To compute the updated dual, use HermiteDual or
    HermiteDual_TF.
    """
    assert self.dual_coeffs.size > 0
    bias = self.dual_coeffs[0]
    if bias < 1e-7: return
    assert bias < 1
    self.dual_coeffs[0] = 0
    self.dual_coeffs *= 1.0 / (1.0 - bias)

  def Act(self, x):
    return polyval(x, self.poly_coeffs)

  def Act_tf(self, x):
    return EvaluatePoly_TF(x, self.poly_coeffs)

  # Default - dual is evaluated using hermite coeffs.
  # Individual activations should override this with the function if known
  def Dual(self, rho):
    return self.HermiteDual(rho)

  def Dual_tf(self, rho):
    return self.HermiteDual_tf(rho)

  # Setup all the coefficients for an activation (poly, hermite, dual)
  def SetCoeffs(self, max_size):
    return

  def SetParams(self, params):
    return


class Identity_Activation(Activation):

  name = skeleton_pb2.IDENTITY

  def __init__(self):
    self.dual_coeffs = np.zeros(2)
    self.dual_coeffs[1] = 1.0

  def Act(self, x):
    return x

  def Act_tf(self, x):
    return x

  def Dual(self, rho):
    return rho

  def Dual_tf(self, rho):
    return rho

# We define the dual activation for the Relu explicitly, as the automatic
# differentiation implemented in tensorflow produces NaN for the gradient
# at rho = 1.

@function.Defun(tf.float32, tf.float32)
def DualReluGradient(x, dy):
  return 0.5 + (tf.asin(x) / np.pi)

@function.Defun(tf.float32, func_name='ReluDual', grad_func=DualReluGradient)
def DualRelu(rho):
  rho_c = tf.maximum(tf.minimum(rho, 1.0), -1.0)
  return (1.0 / np.pi) * (tf.sqrt(1 - rho_c * rho_c) + rho_c *
                          (np.pi / 2 + tf.asin(rho_c)))


class RELU_Activation(Activation):

  name = skeleton_pb2.RELU

  def __init__(self):
    self.dual_coeffs = np.array([])
    self.SetCoeffs(DEFAULT_COEFF_SIZE)


  def SetCoeffs(self, max_size):
    if self.dual_coeffs.size > max_size:
      return
    max_size = max(3, max_size)  # need at least 3 elements in the dual_coeffs.
    self.dual_coeffs = np.zeros(max_size + 1)
    self.dual_coeffs[0] = 1.0 / np.pi
    self.dual_coeffs[1] = 0.5
    self.dual_coeffs[2] = 0.5 / np.pi
    for n in xrange(3, max_size):
      if n % 2 == 1:
        continue
      if n <= 150:
        ff = math.factorial(n - 3) / (math.factorial(n / 2 - 2) *
                                      math.pow(2, n / 2 - 2))
        self.dual_coeffs[n] = ff * ff / (np.pi * math.factorial(n))
      else:
        """
        (n-3!!)^2 / (pi * n!) = Gamma((n+1)/2) / (Gamma(n/2 + 1) pi^1.5 (n-1)^2)
        Expanding this around n=infinity gives the result below.
        """
        c = math.pow(1.0 / n, 2.5)
        coeff = 2 * c
        c /= n
        coeff += 7.0 / 2.0 * c
        c /= n
        coeff += 81.0 / 16.0 * c
        c /= n
        coeff += 429.0 / 64.0 * c
        self.dual_coeffs[n] = coeff / (math.sqrt(2 * np.pi) * np.pi)
    self.hermite_coeffs = np.sqrt(self.dual_coeffs)
    self.hermite_coeffs[4:max_size:4] *= -1
    # Add a last element which represents the tail so that kReluProbs sums to 1
    self.dual_coeffs[max_size] = 1.0 - self.dual_coeffs.sum()

  def Act(self, x):
    return math.sqrt(2.0) * np.maximum(x, 0)

  def Act_tf(self, x):
    return math.sqrt(2.0) * tf.nn.relu(x)

  def Dual(self, rho):
    assert np.abs(rho).max() < 1.001
    rho_c = np.maximum(np.minimum(rho, 1.0), -1.0)
    return (np.sqrt(1 - rho_c * rho_c) +
            rho_c * (np.pi / 2 + np.arcsin(rho_c))) / np.pi

  def Dual_tf(self, rho):
    return DualRelu(rho)


class Exponential_Activation(Activation):

  name = skeleton_pb2.EXPONENTIAL

  def __init__(self):
    self.dual_coeffs = np.array([])

  def SetParams(self, params):
    assert params.scale > 0
    self.scale = params.scale
    self.scale_sqr = self.scale * self.scale
    self.SetCoeffs(DEFAULT_COEFF_SIZE)

  def SetCoeffs(self, max_size):
    max_size = min(max_size, 50)  # no need to make this bigger - 50! is big.
    if self.dual_coeffs.size > max_size:
      return
    self.dual_coeffs = np.zeros(max_size + 1)
    normalizer = 1.0 / math.pow(np.e, self.scale_sqr)
    for n in xrange(0, max_size):
      self.dual_coeffs[n] = normalizer * math.pow(self.scale_sqr, n) / math.factorial(n)
    self.dual_coeffs[max_size] = 1.0 - self.dual_coeffs.sum()

  def Act(self, x):
    return np.exp(self.scale * x - 2.0 * self.scale_sqr)

  def Act_tf(self, x):
    return tf.exp(self.scale * x - 2.0 * self.scale_sqr)

  def Dual(self, rho):
    return np.exp(self.scale_sqr * (rho - 1.0))

  def Dual_tf(self, rho):
    return tf.exp(self.scale_sqr * (rho - 1.0))


class Polynomial_Activation(Activation):

  name = skeleton_pb2.POLYNOMIAL

  def SetParams(self, params):
    """ The params specify the coefficients of the activation in the
    Hermite basis.  The dual activation coeffs are the squares of these.
    We convert the activation to the standard basis for ease of computation -
    we compute the requred number of Hermite polynomials, normalize them and
    add them up.
    """
    self.hermite_coeffs = np.array(params.coefficients)
    if abs(self.hermite_coeffs[0]) > 1e-6:
      print "Warning: The POLY activation has bias ", self.hermite_coeffs[0]
    self.dual_coeffs = self.hermite_coeffs * self.hermite_coeffs
    sum = self.dual_coeffs.sum()
    if abs(sum - 1) > 1e-6:
       print "Warning: The activation is not normalized - Normalizing ..."
       self.dual_coeffs /= sum
       self.hermite_coeffs /= math.sqrt(sum)
    self.poly_coeffs = HermiteToPoly(self.hermite_coeffs)


class Sine_Activation(Activation):

  name = skeleton_pb2.SINE

  def SetParams(self, params):
    assert params.scale > 0
    self.scale = params.scale
    self.scale_sqr = self.scale * self.scale
    self.dual_norm = 2.0 / (math.exp(self.scale_sqr) -
                            math.exp(-self.scale_sqr))
    self.norm = math.sqrt(math.exp(self.scale_sqr) * self.dual_norm)
    self.SetCoeffs(DEFAULT_COEFF_SIZE)

  def SetCoeffs(self, max_size):
    max_size = min(max_size, 50)  # no need to make this bigger - 50! is big.
    if self.dual_coeffs.size > max_size:
      return
    k = np.arange(1, max_size, 2)
    self.hermite_coeffs = np.zeros(max_size)
    self.hermite_coeffs[1::2] = (math.sqrt(self.dual_norm) *
                                 np.power(self.scale, k) * np.power(-1, k / 2)
                                 / np.sqrt(scipy.misc.factorial(k)))
    self.dual_coeffs = np.zeros(max_size + 1)
    self.dual_coeffs[0:max_size] = self.hermite_coeffs * self.hermite_coeffs
    self.dual_coeffs[max_size] = 1.0 - self.dual_coeffs.sum()

  def Act(self, x):
    return self.norm * np.sin(self.scale * x)

  def Act_tf(self, x):
    return self.norm * tf.sin(self.scale * x)

  def Dual(self, rho):
    return (self.dual_norm / 2.0) * (
        np.exp(self.scale_sqr * rho) - np.exp(-self.scale_sqr * rho))

  def Dual_tf(self, rho):
    return (self.dual_norm / 2.0) * (
        tf.exp(self.scale_sqr * rho) - tf.exp(-self.scale_sqr * rho))


class Cosine_Activation(Activation):
  """ Cosine activation, normalized. """

  name = skeleton_pb2.COSINE

  def SetParams(self, params):
    assert params.scale > 0
    self.scale = params.scale
    self.scale_sqr = self.scale * self.scale
    self.dual_norm = 2.0 / (math.exp(self.scale_sqr) +
                            math.exp(-self.scale_sqr) )
    self.norm = math.sqrt(math.exp(self.scale_sqr) * self.dual_norm)
    self.SetCoeffs(DEFAULT_COEFF_SIZE)

  def SetCoeffs(self, max_size):
    max_size = min(max_size, 50)  # no need to make this bigger - 50! is big.
    if self.dual_coeffs.size > max_size:
      return
    k = np.arange(0, max_size, 2)
    self.hermite_coeffs = np.zeros(max_size)
    self.hermite_coeffs[0::2] = (math.sqrt(self.dual_norm) *
                                 np.power(self.scale, k) * np.power(-1, k / 2)
                                 / np.sqrt(scipy.misc.factorial(k)))
    self.dual_coeffs = np.zeros(max_size + 1)
    self.dual_coeffs[0:max_size] = self.hermite_coeffs * self.hermite_coeffs
    self.dual_coeffs[max_size] = 1.0 - self.dual_coeffs.sum()


  def Act(self, x):
    return self.norm * np.cos(self.scale * x)

  def Act_tf(self, x):
    return self.norm * tf.cos(self.scale * x)

  def Dual(self, rho):
    return (self.dual_norm / 2.0) * (
        np.exp(self.scale_sqr * rho) + np.exp(-self.scale_sqr * rho))

  def Dual_tf(self, rho):
    return (self.dual_norm / 2.0) * (
        tf.exp(self.scale_sqr * rho) + tf.exp(-self.scale_sqr * rho))
