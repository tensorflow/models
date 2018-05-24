# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
r"""Helpers for testing the regularizers framework.

Contains:

- Code to build a simple convolutional model with concatenation and a residual
  connection.

- Logic for creating Stubs for OpRegularizers for the convolutions in the model.

- Helpers that calculate the expected values of the alive and regularization
  vectors.


The model is:

             -> conv1 --+     -> conv3 -->  conv4 --
            /           |    /                      \
      image          [concat]                      (add) --> output
            \           |    \                      /
             -> conv2 --+     -> -------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers


layers = tf.contrib.layers


class OpRegularizerStub(generic_regularizers.OpRegularizer):
  """A stub that exponses a constant regularization_vector and alive_vector."""

  def __init__(self, regularization_vector, alive_vector):
    self._regularization_vector = tf.constant(
        regularization_vector, dtype=tf.float32)
    self._alive_vector = tf.constant(alive_vector, dtype=tf.bool)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


ALIVE_STUB = {
    'conv1': [False, True, True, False, True, False, True],
    'conv2': [True, False, True, False, False],
    'conv3': [False, False, True, True],
    'conv4': [False, True, False, True, True, False, True,
              False, False, True, False, False],
    'conv5': [False, True, False]
}


REG_STUB = {
    'conv1': [0.1, 0.3, 0.6, 0.2, 0.4, 0.0, 0.8],
    'conv2': [0.15, 0.25, 0.05, 0.55, 0.45],
    'conv3': [0.07, 0.27, 0.17, 0.37],
    'conv4': [
        0.07, 0.27, 0.17, 0.37, 0.28, 0.32, 0.12, 0.22, 0.19, 0.11, 0.02, 0.06
    ],
    'conv5': [0.24, 0.34, 0.29]
}


def _create_stub(key):
  return OpRegularizerStub(REG_STUB[key], ALIVE_STUB[key])


def build_model():
  image = tf.constant(0.0, shape=[1, 17, 19, 3])
  conv1 = layers.conv2d(image, 7, [7, 5], padding='SAME', scope='conv1')
  conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv2')
  concat = tf.concat([conv1, conv2], 3)
  conv3 = layers.conv2d(concat, 4, [1, 1], padding='SAME', scope='conv3')
  conv4 = layers.conv2d(conv3, 12, [3, 3], padding='SAME', scope='conv4')
  conv5 = layers.conv2d(
      concat + conv4, 3, [3, 3], stride=2, padding='SAME', scope='conv5')
  return conv5.op


def _create_conv2d_regularizer(conv_op, manager=None):
  del manager  # unused
  for key in REG_STUB:
    if conv_op.name.startswith(key):
      return _create_stub(key)
  raise ValueError('No regularizer for %s' % conv_op.name)


MOCK_REG_DICT = {'Conv2D': _create_conv2d_regularizer}


def expected_regularization():
  """Build the expected alive vectors applying the rules of concat and group."""
  concat = REG_STUB['conv1'] + REG_STUB['conv2']
  # Grouping: Activation is alive after grouping if one of the constituents is
  # alive.
  grouped = [max(a, b) for a, b in zip(concat, REG_STUB['conv4'])]
  conv1_length = len(REG_STUB['conv1'])
  return {
      'conv1': grouped[:conv1_length],
      'conv2': grouped[conv1_length:],
      'conv3': REG_STUB['conv3'],
      'conv4': grouped,
      'conv5': REG_STUB['conv5'],
      'add': grouped,
      'concat': grouped
  }


def expected_alive():
  """Build the expected alive vectors applying the rules of concat and group."""
  concat = ALIVE_STUB['conv1'] + ALIVE_STUB['conv2']
  # Grouping: Activation is alive after grouping if one of the constituents is
  # alive.
  grouped = [a or b for a, b in zip(concat, ALIVE_STUB['conv4'])]
  conv1_length = len(ALIVE_STUB['conv1'])
  return {
      'conv1': grouped[:conv1_length],
      'conv2': grouped[conv1_length:],
      'conv3': ALIVE_STUB['conv3'],
      'conv4': grouped,
      'conv5': ALIVE_STUB['conv5'],
      'add': grouped,
      'concat': grouped
  }
