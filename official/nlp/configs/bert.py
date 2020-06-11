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
"""A multi-head BERT encoder network for pretraining."""
from typing import List, Optional, Text

import dataclasses
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import encoders
from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_pretrainer


@dataclasses.dataclass
class ClsHeadConfig(base_config.Config):
  inner_dim: int = 0
  num_classes: int = 2
  activation: Optional[Text] = "tanh"
  dropout_rate: float = 0.0
  cls_token_idx: int = 0
  name: Optional[Text] = None


@dataclasses.dataclass
class BertPretrainerConfig(base_config.Config):
  """BERT encoder configuration."""
  num_masked_tokens: int = 76
  encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  cls_heads: List[ClsHeadConfig] = dataclasses.field(default_factory=list)


def instantiate_from_cfg(
    config: BertPretrainerConfig,
    encoder_network: Optional[tf.keras.Model] = None):
  """Instantiates a BertPretrainer from the config."""
  encoder_cfg = config.encoder
  if encoder_network is None:
    encoder_network = networks.TransformerEncoder(
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
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range))
  if config.cls_heads:
    classification_heads = [
        layers.ClassificationHead(**cfg.as_dict()) for cfg in config.cls_heads
    ]
  else:
    classification_heads = []
  return bert_pretrainer.BertPretrainerV2(
      config.num_masked_tokens,
      mlm_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      mlm_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      encoder_network=encoder_network,
      classification_heads=classification_heads)


@dataclasses.dataclass
class BertPretrainDataConfig(cfg.DataConfig):
  """Data config for BERT pretraining task."""
  input_path: str = ""
  global_batch_size: int = 512
  is_training: bool = True
  seq_length: int = 512
  max_predictions_per_seq: int = 76
  use_next_sentence_label: bool = True
  use_position_id: bool = False


@dataclasses.dataclass
class BertPretrainEvalDataConfig(BertPretrainDataConfig):
  """Data config for the eval set in BERT pretraining task."""
  input_path: str = ""
  global_batch_size: int = 512
  is_training: bool = False


@dataclasses.dataclass
class BertSentencePredictionDataConfig(cfg.DataConfig):
  """Data of sentence prediction dataset."""
  input_path: str = ""
  global_batch_size: int = 32
  is_training: bool = True
  seq_length: int = 128


@dataclasses.dataclass
class BertSentencePredictionDevDataConfig(cfg.DataConfig):
  """Dev data of MNLI sentence prediction dataset."""
  input_path: str = ""
  global_batch_size: int = 32
  is_training: bool = False
  seq_length: int = 128
  drop_remainder: bool = False
