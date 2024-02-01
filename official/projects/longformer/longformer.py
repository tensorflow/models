# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Longformer model configurations and instantiation methods."""
import dataclasses
from typing import List

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.projects.longformer.longformer_encoder import LongformerEncoder


@dataclasses.dataclass
class LongformerEncoderConfig(encoders.BertEncoderConfig):
  """Extra paramerters for Longformer configs.

  Attributes:
    attention_window: list of ints representing the window size for each layer.
    global_attention_size: the size of global attention used for each token.
    pad_token_id: the token id for the pad token
  """
  attention_window: List[int] = dataclasses.field(default_factory=list)
  global_attention_size: int = 0
  pad_token_id: int = 1


@base_config.bind(LongformerEncoderConfig)
def get_encoder(encoder_cfg: LongformerEncoderConfig):
  """Gets a 'LongformerEncoder' object.

  Args:
    encoder_cfg: A 'LongformerEncoderConfig'.

  Returns:
    A encoder object.
  """
  encoder = LongformerEncoder(
      attention_window=encoder_cfg.attention_window,
      global_attention_size=encoder_cfg.global_attention_size,
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      inner_dim=encoder_cfg.intermediate_size,
      inner_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      output_dropout=encoder_cfg.dropout_rate,
      attention_dropout=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf_keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      norm_first=encoder_cfg.norm_first)
  return encoder
