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

"""Backbones configurations."""
import dataclasses
from typing import List, Optional, Tuple

from official.modeling import hyperparams


@dataclasses.dataclass
class Transformer(hyperparams.Config):
  """Transformer config."""
  mlp_dim: int = 1
  num_heads: int = 1
  num_layers: int = 1
  attention_dropout_rate: float = 0.0
  dropout_rate: float = 0.0


@dataclasses.dataclass
class VisionTransformer(hyperparams.Config):
  """VisionTransformer config."""
  model_name: str = 'vit-b16'
  # pylint: disable=line-too-long
  pooler: str = 'token'  # 'token', 'gap' or 'none'. If set to 'token', an extra classification token is added to sequence.
  # pylint: enable=line-too-long
  representation_size: int = 0
  hidden_size: int = 1
  patch_size: int = 16
  transformer: Transformer = dataclasses.field(default_factory=Transformer)
  init_stochastic_depth_rate: float = 0.0
  original_init: bool = True
  pos_embed_shape: Optional[Tuple[int, int]] = None
  # If output encoded tokens sequence when pooler is `none`.
  output_encoded_tokens: bool = True
  # If output encoded tokens 2D feature map.
  output_2d_feature_maps: bool = False

  # Adding Layerscale to each Encoder block https://arxiv.org/abs/2204.07118
  layer_scale_init_value: float = 0.0
  # Transformer encoder spatial partition dimensions.
  transformer_partition_dims: Optional[Tuple[int, int, int, int]] = None
  # If True, output attention scores.
  output_attention_scores: bool = False


@dataclasses.dataclass
class ResNet(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50
  depth_multiplier: float = 1.0
  stem_type: str = 'v0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  scale_stem: bool = True
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False
  bn_trainable: bool = True


@dataclasses.dataclass
class DilatedResNet(hyperparams.Config):
  """DilatedResNet config."""
  model_id: int = 50
  output_stride: int = 16
  multigrid: Optional[List[int]] = None
  stem_type: str = 'v0'
  last_stage_repeats: int = 1
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False


@dataclasses.dataclass
class EfficientNet(hyperparams.Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class MobileNet(hyperparams.Config):
  """Mobilenet config."""
  model_id: str = 'MobileNetV2'
  filter_size_scale: float = 1.0
  stochastic_depth_drop_rate: float = 0.0
  # Whether to apply a fixed and common stochastic depth drop rate to all
  # blocks, instead to linearly scale it from zero to maximum value (standard
  # behaviour for stochastic depth). Set to True for backward compatibility.
  flat_stochastic_depth_drop_rate: bool = True
  output_stride: Optional[int] = None
  output_intermediate_endpoints: bool = False


@dataclasses.dataclass
class SpineNet(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  min_level: int = 3
  max_level: int = 7


@dataclasses.dataclass
class SpineNetMobile(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.2
  expand_ratio: int = 6
  min_level: int = 3
  max_level: int = 7
  # If use_keras_upsampling_2d is True, model uses UpSampling2D keras layer
  # instead of optimized custom TF op. It makes model be more keras style. We
  # set this flag to True when we apply QAT from model optimization toolkit
  # that requires the model should use keras layers.
  use_keras_upsampling_2d: bool = False


@dataclasses.dataclass
class RevNet(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: int = 56


@dataclasses.dataclass
class MobileDet(hyperparams.Config):
  """Mobiledet config."""
  model_id: str = 'MobileDetCPU'
  filter_size_scale: float = 1.0


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    resnet: resnet backbone config.
    dilated_resnet: dilated resnet backbone for semantic segmentation config.
    revnet: revnet backbone config.
    efficientnet: efficientnet backbone config.
    spinenet: spinenet backbone config.
    spinenet_mobile: mobile spinenet backbone config.
    mobilenet: mobilenet backbone config.
    mobiledet: mobiledet backbone config.
    vit: vision transformer backbone config.
  """
  type: Optional[str] = None
  resnet: ResNet = dataclasses.field(default_factory=ResNet)
  dilated_resnet: DilatedResNet = dataclasses.field(
      default_factory=DilatedResNet
  )
  revnet: RevNet = dataclasses.field(default_factory=RevNet)
  efficientnet: EfficientNet = dataclasses.field(default_factory=EfficientNet)
  spinenet: SpineNet = dataclasses.field(default_factory=SpineNet)
  spinenet_mobile: SpineNetMobile = dataclasses.field(
      default_factory=SpineNetMobile
  )
  mobilenet: MobileNet = dataclasses.field(default_factory=MobileNet)
  mobiledet: MobileDet = dataclasses.field(default_factory=MobileDet)
  vit: VisionTransformer = dataclasses.field(default_factory=VisionTransformer)
