# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Implementation of pQRNN model."""
# pylint: disable=arguments-renamed

from absl import logging
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import transformer_layers # import seq_flow_lite module


class Model(tf.keras.layers.Layer):
  """Quantized transformer encoder."""

  def __init__(self, config, mode):

    def _get_params(varname, default_value=None):
      value = config[varname] if varname in config else default_value
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("intermediate_size")
    _get_params("max_time_step")
    _get_params("embedding_size")
    _get_params("vocabulary_size")
    _get_params("num_layers")
    _get_params("labels")
    _get_params("regularizer_scale")
    _get_params("num_heads")
    _get_params("model_dimension")
    _get_params("quantize")
    _get_params("activation_dropout_rate", 0.0)
    _get_params("attention_dropout_rate", 0.0)
    self.parameters = base_layers.Parameters(mode, self.quantize,
                                             self.regularizer_scale)

    super(Model, self).__init__()

  def build(self, input_shape):
    self.transformer = transformer_layers.TransformerEncoderStack(
        parameters=self.parameters,
        num_layers=self.num_layers,
        intermediate_size=self.intermediate_size,
        embedding_size=self.embedding_size,
        max_time_step=self.max_time_step,
        num_heads=self.num_heads,
        model_dimension=self.model_dimension,
        vocabulary_size=self.vocabulary_size,
        activation_dropout_rate=self.activation_dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate)

  def call(self, indices, sequence_length):
    return self.transformer(indices, sequence_length)


class ModelWithEmbeddings(Model):
  """Quantized transformer encoder which takes embeddings instead of indices."""

  def build(self, input_shape):
    self.transformer_with_input_embedding = transformer_layers.TransformerEncoderStackWithInputEmbedding(
        parameters=self.parameters,
        num_layers=self.num_layers,
        intermediate_size=self.intermediate_size,
        embedding_size=self.embedding_size,
        max_time_step=self.max_time_step,
        num_heads=self.num_heads,
        model_dimension=self.model_dimension,
        vocabulary_size=self.vocabulary_size,
        activation_dropout_rate=self.activation_dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate)

  def call(self, embeddings, sequence_length):
    return self.transformer_with_input_embedding(embeddings, sequence_length)


class FunnelTransformerModel(Model):
  """Quantized transformer encoder which takes embeddings instead of indices."""

  def __init__(self, config, mode):
    self.pool_windows = config.get("pool_windows", None)
    super(FunnelTransformerModel, self).__init__(config, mode)

  def build(self, input_shape):
    self.funnel_transformer = transformer_layers.FunnelTransformerEncoderStack(
        parameters=self.parameters,
        num_layers=self.num_layers,
        intermediate_size=self.intermediate_size,
        embedding_size=self.embedding_size,
        max_time_step=self.max_time_step,
        num_heads=self.num_heads,
        model_dimension=self.model_dimension,
        vocabulary_size=self.vocabulary_size,
        activation_dropout_rate=self.activation_dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        pool_windows=self.pool_windows)

  def call(self, embeddings, sequence_length):
    return self.funnel_transformer(embeddings, sequence_length)
