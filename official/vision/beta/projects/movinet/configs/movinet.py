# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Definitions for MoViNet structures.

Reference: "MoViNets: Mobile Video Networks for Efficient Video Recognition"
https://arxiv.org/pdf/2103.11511.pdf

MoViNets are efficient video classification networks that are part of a model
family, ranging from the smallest model, MoViNet-A0, to the largest model,
MoViNet-A6. Each model has various width, depth, input resolution, and input
frame-rate associated with them. See the main paper for more details.
"""

import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.configs import backbones_3d
from official.vision.beta.configs import common
from official.vision.beta.configs import video_classification


@dataclasses.dataclass
class Movinet(hyperparams.Config):
  """Backbone config for Base MoViNet."""
  model_id: str = 'a0'
  causal: bool = False
  use_positional_encoding: bool = False
  # Choose from ['3d', '2plus1d', '3d_2plus1d']
  # 3d: default 3D convolution
  # 2plus1d: (2+1)D convolution with Conv2D (2D reshaping)
  # 3d_2plus1d: (2+1)D convolution with Conv3D (no 2D reshaping)
  conv_type: str = '3d'
  # Choose from ['3d', '2d', '2plus3d']
  # 3d: default 3D global average pooling.
  # 2d: 2D global average pooling.
  # 2plus3d: concatenation of 2D and 3D global average pooling.
  se_type: str = '3d'
  activation: str = 'swish'
  gating_activation: str = 'sigmoid'
  stochastic_depth_drop_rate: float = 0.2
  use_external_states: bool = False


@dataclasses.dataclass
class MovinetA0(Movinet):
  """Backbone config for MoViNet-A0.

  Represents the smallest base MoViNet searched by NAS.

  Reference: https://arxiv.org/pdf/2103.11511.pdf
  """
  model_id: str = 'a0'


@dataclasses.dataclass
class MovinetA1(Movinet):
  """Backbone config for MoViNet-A1."""
  model_id: str = 'a1'


@dataclasses.dataclass
class MovinetA2(Movinet):
  """Backbone config for MoViNet-A2."""
  model_id: str = 'a2'


@dataclasses.dataclass
class MovinetA3(Movinet):
  """Backbone config for MoViNet-A3."""
  model_id: str = 'a3'


@dataclasses.dataclass
class MovinetA4(Movinet):
  """Backbone config for MoViNet-A4."""
  model_id: str = 'a4'


@dataclasses.dataclass
class MovinetA5(Movinet):
  """Backbone config for MoViNet-A5.

  Represents the largest base MoViNet searched by NAS.
  """
  model_id: str = 'a5'


@dataclasses.dataclass
class MovinetT0(Movinet):
  """Backbone config for MoViNet-T0.

  MoViNet-T0 is a smaller version of MoViNet-A0 for even faster processing.
  """
  model_id: str = 't0'


@dataclasses.dataclass
class Backbone3D(backbones_3d.Backbone3D):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, on the of fields below.
    movinet: movinet backbone config.
  """
  type: str = 'movinet'
  movinet: Movinet = Movinet()


@dataclasses.dataclass
class MovinetModel(video_classification.VideoClassificationModel):
  """The MoViNet model config."""
  model_type: str = 'movinet'
  backbone: Backbone3D = Backbone3D()
  norm_activation: common.NormActivation = common.NormActivation(
      activation='swish',
      norm_momentum=0.99,
      norm_epsilon=1e-3,
      use_sync_bn=True)
  output_states: bool = False


@exp_factory.register_config_factory('movinet_kinetics600')
def movinet_kinetics600() -> cfg.ExperimentConfig:
  """Video classification on Videonet with MoViNet backbone."""
  exp = video_classification.video_classification_kinetics600()
  exp.task.train_data.dtype = 'bfloat16'
  exp.task.validation_data.dtype = 'bfloat16'

  model = MovinetModel()
  exp.task.model = model

  return exp
