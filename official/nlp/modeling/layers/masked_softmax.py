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
"""Keras-based softmax layer with optional masking."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
class MaskedSoftmax(tf.keras.layers.Layer):
  """Performs a softmax with optional masking on a tensor.

  Attributes:
    mask_expansion_axes: Any axes that should be padded on the mask tensor.
  """

  def __init__(self, mask_expansion_axes=None, **kwargs):
    self._mask_expansion_axes = mask_expansion_axes
    super(MaskedSoftmax, self).__init__(**kwargs)

  def call(self, inputs):
    if isinstance(inputs, list) and len(inputs) == 2:
      scores, mask = inputs
    else:
      scores, mask = (inputs, None)

    if mask is not None:
      if self._mask_expansion_axes is not None:
        mask = tf.expand_dims(mask, axis=self._mask_expansion_axes)

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(mask, scores.dtype)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      scores += adder

    return tf.nn.softmax(scores)

  def get_config(self):
    config = {'mask_expansion_axes': self._mask_expansion_axes}
    base_config = super(MaskedSoftmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
