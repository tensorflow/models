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
"""ELECTRA model configurations and instantiation methods."""
from typing import List, Optional

import dataclasses
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.modeling import layers
from official.nlp.modeling.models import electra_pretrainer


@dataclasses.dataclass
class ELECTRAPretrainerConfig(base_config.Config):
  """ELECTRA pretrainer configuration."""
  num_masked_tokens: int = 76
  sequence_length: int = 512
  num_classes: int = 2
  discriminator_loss_weight: float = 50.0
  tie_embeddings: bool = True
  disallow_correct: bool = False
  generator_encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  discriminator_encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  cls_heads: List[bert.ClsHeadConfig] = dataclasses.field(default_factory=list)


def instantiate_classification_heads_from_cfgs(
    cls_head_configs: List[bert.ClsHeadConfig]
) -> List[layers.ClassificationHead]:
  if cls_head_configs:
    return [
        layers.ClassificationHead(**cfg.as_dict()) for cfg in cls_head_configs
    ]
  else:
    return []


def instantiate_pretrainer_from_cfg(
    config: ELECTRAPretrainerConfig,
    generator_network: Optional[tf.keras.Model] = None,
    discriminator_network: Optional[tf.keras.Model] = None,
    ) -> electra_pretrainer.ElectraPretrainer:
  """Instantiates ElectraPretrainer from the config."""
  generator_encoder_cfg = config.generator_encoder
  discriminator_encoder_cfg = config.discriminator_encoder
  # Copy discriminator's embeddings to generator for easier model serialization.
  if discriminator_network is None:
    discriminator_network = encoders.instantiate_encoder_from_cfg(
        discriminator_encoder_cfg)
  if generator_network is None:
    if config.tie_embeddings:
      embedding_layer = discriminator_network.get_embedding_layer()
      generator_network = encoders.instantiate_encoder_from_cfg(
          generator_encoder_cfg, embedding_layer=embedding_layer)
    else:
      generator_network = encoders.instantiate_encoder_from_cfg(
          generator_encoder_cfg)

  return electra_pretrainer.ElectraPretrainer(
      generator_network=generator_network,
      discriminator_network=discriminator_network,
      vocab_size=config.generator_encoder.vocab_size,
      num_classes=config.num_classes,
      sequence_length=config.sequence_length,
      num_token_predictions=config.num_masked_tokens,
      mlm_activation=tf_utils.get_activation(
          generator_encoder_cfg.hidden_activation),
      mlm_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=generator_encoder_cfg.initializer_range),
      classification_heads=instantiate_classification_heads_from_cfgs(
          config.cls_heads),
      disallow_correct=config.disallow_correct)
