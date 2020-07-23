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
"""Multi-head BERT encoder network with classification heads.

Includes configurations and instantiation methods.
"""
from typing import List, Optional, Text

import dataclasses
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.modeling import layers
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
  encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  cls_heads: List[ClsHeadConfig] = dataclasses.field(default_factory=list)


def instantiate_classification_heads_from_cfgs(
    cls_head_configs: List[ClsHeadConfig]) -> List[layers.ClassificationHead]:
  return [
      layers.ClassificationHead(**cfg.as_dict()) for cfg in cls_head_configs
    ] if cls_head_configs else []


def instantiate_pretrainer_from_cfg(
    config: BertPretrainerConfig,
    encoder_network: Optional[tf.keras.Model] = None
) -> bert_pretrainer.BertPretrainerV2:
  """Instantiates a BertPretrainer from the config."""
  encoder_cfg = config.encoder
  if encoder_network is None:
    encoder_network = encoders.instantiate_encoder_from_cfg(encoder_cfg)
  return bert_pretrainer.BertPretrainerV2(
      mlm_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      mlm_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      encoder_network=encoder_network,
      classification_heads=instantiate_classification_heads_from_cfgs(
          config.cls_heads))
