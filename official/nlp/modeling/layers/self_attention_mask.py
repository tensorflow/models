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

"""Keras layer that creates a self-attention mask."""

import tensorflow as tf

from official.nlp.keras_nlp import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class SelfAttentionMask(layers.SelfAttentionMask):
  """Creates 3D attention mask from a 2D tensor mask.

    **Warning: Please use the `keras_nlp.layers.SelfAttentionMask`.**
    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      `(batch_size, from_seq_length, ...)`.
    inputs[1]: to_mask: int32 Tensor of shape `(batch_size, to_seq_length)`.

    Returns:
      Float Tensor of shape `(batch_size, from_seq_length, to_seq_length)`.
  """

  def call(self, inputs):
    if isinstance(inputs, list):
      return super().call(inputs[0], inputs[1])
    else:
      return super().call(inputs)
