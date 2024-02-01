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

"""Token dropping encoder configuration and instantiation."""
import dataclasses
from typing import Tuple
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.projects.token_dropping import encoder


@dataclasses.dataclass
class TokenDropBertEncoderConfig(encoders.BertEncoderConfig):
  token_loss_init_value: float = 10.0
  token_loss_beta: float = 0.995
  token_keep_k: int = 256
  token_allow_list: Tuple[int, ...] = (100, 101, 102, 103)
  token_deny_list: Tuple[int, ...] = (0,)


@base_config.bind(TokenDropBertEncoderConfig)
def get_encoder(encoder_cfg: TokenDropBertEncoderConfig):
  """Instantiates 'TokenDropBertEncoder'.

  Args:
    encoder_cfg: A 'TokenDropBertEncoderConfig'.

  Returns:
    A 'encoder.TokenDropBertEncoder' object.
  """
  return encoder.TokenDropBertEncoder(
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      intermediate_size=encoder_cfg.intermediate_size,
      activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      dropout_rate=encoder_cfg.dropout_rate,
      attention_dropout_rate=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf_keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      return_all_encoder_outputs=encoder_cfg.return_all_encoder_outputs,
      dict_outputs=True,
      norm_first=encoder_cfg.norm_first,
      token_loss_init_value=encoder_cfg.token_loss_init_value,
      token_loss_beta=encoder_cfg.token_loss_beta,
      token_keep_k=encoder_cfg.token_keep_k,
      token_allow_list=encoder_cfg.token_allow_list,
      token_deny_list=encoder_cfg.token_deny_list)
