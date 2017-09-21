# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Basic data management and plotting utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import getpass
import numpy as np
import gc
import tensorflow as tf

#
# Python utlities
#
def exp_moving_average(x, alpha=0.9):
  res = []
  mu = 0
  alpha_factor = 1
  for x_i in x:
    mu += (1 - alpha)*(x_i - mu)
    alpha_factor *= alpha
    res.append(mu/(1 - alpha_factor))

  return np.array(res)

def sanitize(s):
  return s.replace('.', '_')

#
# Tensorflow utilities
#
def softplus(x):
  '''
  Let m = max(0, x), then,

  sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
             = m + log(e(-m) + e(x - m))

  The term inside of the log is guaranteed to be between 1 and 2.
  '''
  m = tf.maximum(tf.zeros_like(x), x)
  return m + tf.log(tf.exp(-m) + tf.exp(x - m))

def safe_log_prob(x, eps=1e-8):
  return tf.log(tf.clip_by_value(x, eps, 1.0))

def rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))

def center(x):
  mu = (tf.reduce_sum(x) - x)/tf.to_float(tf.shape(x)[0] - 1)
  return x - mu

def vectorize(grads_and_vars, set_none_to_zero=False, skip_none=False):
  if set_none_to_zero:
    return tf.concat([tf.reshape(g, [-1]) if g is not None else
                         tf.reshape(tf.zeros_like(v), [-1]) for g, v in grads_and_vars], 0)
  elif skip_none:
    return tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars if g is not None], 0)
  else:
    return tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars], 0)

def add_grads_and_vars(a, b):
  '''Add grads_and_vars from two calls to tf.compute_gradients.'''
  res = []
  for (g_a, v_a), (g_b, v_b) in zip(a, b):
    assert v_a == v_b
    if g_a is None:
      res.append((g_b, v_b))
    elif g_b is None:
      res.append((g_a, v_a))
    else:
      res.append((g_a + g_b, v_a))
  return res

def binary_log_likelihood(y, log_y_hat):
  """Computes binary log likelihood.

  Args:
    y: observed data
    log_y_hat: parameters of the binary variables

  Returns:
    log_likelihood
  """
  return tf.reduce_sum(y*(-softplus(-log_y_hat)) +
                       (1 - y)*(-log_y_hat-softplus(-log_y_hat)),
                       1)

def cov(a, b):
  """Compute the sample covariance between two vectors."""
  mu_a = tf.reduce_mean(a)
  mu_b = tf.reduce_mean(b)
  n = tf.to_float(tf.shape(a)[0])

  return tf.reduce_sum((a - mu_a)*(b - mu_b))/(n - 1.0)

def corr(a, b):
  return cov(a, b)*tf.rsqrt(cov(a, a))*tf.rsqrt(cov(b, b))

def logSumExp(t, axis=0, keep_dims = False):
  '''Computes the log(sum(exp(t))) numerically stabily.

  Args:
    t: input tensor
    axis: which axis to sum over
    keep_dims: whether to keep the dim or not

  Returns:
    tensor with result

  '''
  m = tf.reduce_max(t, [axis])
  res = m + tf.log(tf.reduce_sum(tf.exp(t - tf.expand_dims(m, axis)), [axis]))

  if keep_dims:
    return tf.expand_dims(res, axis)
  else:
    return res
