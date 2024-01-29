# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


class OptimizerFactory(object):
  """Class to generate optimizer function."""

  def __init__(self, params):
    """Creates optimized based on the specified flags."""
    if params.type == 'momentum':
      self._optimizer = functools.partial(
          tf.keras.optimizers.SGD,
          momentum=params.momentum,
          nesterov=params.nesterov)
    elif params.type == 'adam':
      self._optimizer = tf.keras.optimizers.Adam
    elif params.type == 'adadelta':
      self._optimizer = tf.keras.optimizers.Adadelta
    elif params.type == 'adagrad':
      self._optimizer = tf.keras.optimizers.Adagrad
    elif params.type == 'rmsprop':
      self._optimizer = functools.partial(
          tf.keras.optimizers.RMSprop, momentum=params.momentum)
    else:
      raise ValueError('Unsupported optimizer type `{}`.'.format(params.type))

  def __call__(self, learning_rate):
    return self._optimizer(learning_rate=learning_rate)
