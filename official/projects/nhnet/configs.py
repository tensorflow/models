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

"""Common NHNet/Bert2Bert configuration."""

import dataclasses
from typing import List, Text

from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class BERT2BERTConfig(base_config.Config):
  """High-level configurations for BERT2BERT model.

  These include parameters that are not directly related to the experiment,
  e.g. encoder, decoder, prediction, training, etc.
  """
  vocab_size: int = 30522
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_act: str = "gelu"
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  decoder_intermediate_size: int = 3072
  num_decoder_attn_heads: int = 12
  num_decoder_layers: int = 12

  label_smoothing: float = 0.1
  learning_rate: float = 0.05
  learning_rate_warmup_steps: int = 20000
  optimizer: str = "Adam"
  adam_beta1: float = 0.9
  adam_beta2: float = 0.997
  adam_epsilon: float = 1e-09

  # predict params
  beam_size: int = 5
  alpha: float = 0.6
  initializer_gain: float = 1.0
  use_cache: bool = True

  # input params
  input_sharding: bool = False
  input_data_not_padded: bool = False
  pad_token_id: int = 0
  end_token_id: int = 102
  start_token_id: int = 101


@dataclasses.dataclass
class NHNetConfig(BERT2BERTConfig):
  """High-level configurations for NHNet model.

  These include parameters that are not directly related to the experiment,
  e.g. encoder, decoder, prediction, training, etc.
  """
  multi_channel_cross_attention: bool = True
  passage_list: List[Text] = dataclasses.field(
      default_factory=lambda: [chr(ord("b") + i) for i in range(5)])

  # Initialization method.
  # If init_from_bert2bert is false, we assume the checkpoint is from BERT
  # pretraining and only encoder and self-attention variables are initialized.
  init_from_bert2bert: bool = True


UNITTEST_CONFIG = {
    "attention_probs_dropout_prob": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 16,
    "initializer_range": 0.02,
    "intermediate_size": 32,
    "max_position_embeddings": 128,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "type_vocab_size": 2,
    "vocab_size": 30522,
    "initializer_gain": 1.0,
    "decoder_intermediate_size": 32,
    "num_decoder_attn_heads": 2,
    "num_decoder_layers": 1,
    "use_cache": True,
    "input_data_not_padded": False,
    "pad_token_id": 0,
    "end_token_id": 102,
    "start_token_id": 101,
}
