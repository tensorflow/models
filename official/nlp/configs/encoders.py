# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Transformer Encoders.

Includes configurations and instantiation methods.
"""

import dataclasses
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.modeling import networks


@dataclasses.dataclass
class TransformerEncoderConfig(base_config.Config):
  """BERT encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02


def instantiate_encoder_from_cfg(
    config: TransformerEncoderConfig) -> networks.TransformerEncoder:
  """Instantiate a Transformer encoder network from TransformerEncoderConfig."""
  encoder_network = networks.TransformerEncoder(
      vocab_size=config.vocab_size,
      hidden_size=config.hidden_size,
      num_layers=config.num_layers,
      num_attention_heads=config.num_attention_heads,
      intermediate_size=config.intermediate_size,
      activation=tf_utils.get_activation(config.hidden_activation),
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      sequence_length=None,
      max_sequence_length=config.max_position_embeddings,
      type_vocab_size=config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=config.initializer_range))
  return encoder_network
