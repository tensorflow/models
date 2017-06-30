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

"""Utility functions for blocks."""

from __future__ import division
from __future__ import unicode_literals

import math

import numpy as np
import tensorflow as tf


class RsqrtInitializer(object):
  """Gaussian initializer with standard deviation 1/sqrt(n).

  Note that tf.truncated_normal is used internally. Therefore any random sample
  outside two-sigma will be discarded and re-sampled.
  """

  def __init__(self, dims=(0,), **kwargs):
    """Creates an initializer.

    Args:
      dims: Dimension(s) index to compute standard deviation:
        1.0 / sqrt(product(shape[dims]))
      **kwargs: Extra keyword arguments to pass to tf.truncated_normal.
    """
    if isinstance(dims, (int, long)):
      self._dims = [dims]
    else:
      self._dims = dims
    self._kwargs = kwargs

  def __call__(self, shape, dtype):
    stddev = 1.0 / np.sqrt(np.prod([shape[x] for x in self._dims]))
    return tf.truncated_normal(
        shape=shape, dtype=dtype, stddev=stddev, **self._kwargs)


class RectifierInitializer(object):
  """Gaussian initializer with standard deviation sqrt(2/fan_in).

  Note that tf.random_normal is used internally to ensure the expected weight
  distribution. This is intended to be used with ReLU activations, specially
  in ResNets.

  For details please refer to:
  Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
  Classification
  """

  def __init__(self, dims=(0,), scale=2.0, **kwargs):
    """Creates an initializer.

    Args:
      dims: Dimension(s) index to compute standard deviation:
        sqrt(scale / product(shape[dims]))
      scale: A constant scaling for the initialization used as
        sqrt(scale / product(shape[dims])).
      **kwargs: Extra keyword arguments to pass to tf.truncated_normal.
    """
    if isinstance(dims, (int, long)):
      self._dims = [dims]
    else:
      self._dims = dims
    self._kwargs = kwargs
    self._scale = scale

  def __call__(self, shape, dtype):
    stddev = np.sqrt(self._scale / np.prod([shape[x] for x in self._dims]))
    return tf.random_normal(
        shape=shape, dtype=dtype, stddev=stddev, **self._kwargs)


class GaussianInitializer(object):
  """Gaussian initializer with a given standard deviation.

  Note that tf.truncated_normal is used internally. Therefore any random sample
  outside two-sigma will be discarded and re-sampled.
  """

  def __init__(self, stddev=1.0):
    self._stddev = stddev

  def __call__(self, shape, dtype):
    return tf.truncated_normal(shape=shape, dtype=dtype, stddev=self._stddev)
