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

"""ELECTRA model configurations and instantiation methods."""
from typing import List

import dataclasses

from official.modeling.hyperparams import base_config
from official.nlp.configs import bert
from official.nlp.configs import encoders


@dataclasses.dataclass
class ElectraPretrainerConfig(base_config.Config):
  """ELECTRA pretrainer configuration."""
  num_masked_tokens: int = 76
  sequence_length: int = 512
  num_classes: int = 2
  discriminator_loss_weight: float = 50.0
  tie_embeddings: bool = True
  disallow_correct: bool = False
  generator_encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  discriminator_encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  cls_heads: List[bert.ClsHeadConfig] = dataclasses.field(default_factory=list)
