# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Quantization Wrappers."""
import tensorflow_model_optimization as tfmot


class MultiHeadAttentionQuantizeWrapper(
    tfmot.quantization.keras.QuantizeWrapperV2):
  """Custom quantize wrapper for the MultiHeadAttention layer."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._first_call_built = False

  def build(self, input_shape):
    self.layer.build(input_shape)

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None):
    if not self._first_call_built:
      # pylint: disable=protected-access
      self.layer._build_from_signature(query=query, value=value, key=key)
      # pylint: enable=protected-access
      self.layer.call(
          query, value, key=key, attention_mask=attention_mask,
          return_attention_scores=return_attention_scores,
          training=training)
      super().build(input_shape=None)
      self._first_call_built = True

    return super().call(
        query, value=value, key=key, attention_mask=attention_mask,
        return_attention_scores=return_attention_scores,
        training=training
    )
