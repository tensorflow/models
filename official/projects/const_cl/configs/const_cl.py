# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Video classification configuration definition."""

import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.const_cl.configs import backbones_3d as backbones_3d_cfg
from official.projects.const_cl.configs import head as head_cfg
from official.vision.configs import common
from official.vision.configs import video_classification


VideoClassificationTask = video_classification.VideoClassificationTask


@dataclasses.dataclass
class ConstCLPretrainTask(VideoClassificationTask):
  pass


@dataclasses.dataclass
class DataConfig(video_classification.DataConfig):
  """The base configuration for building datasets."""
  zero_centering_image: bool = True
  is_ssl: bool = False
  num_instances: int = 8


@dataclasses.dataclass
class ConstCLModel(hyperparams.Config):
  """The model config."""
  model_type: str = 'video_classification'
  backbone: backbones_3d_cfg.Backbone3D = dataclasses.field(
      default_factory=lambda: backbones_3d_cfg.Backbone3D(  # pylint: disable=g-long-lambda
          type='resnet_3dy', resnet_3dy=backbones_3d_cfg.ResNet3DY50()
      )
  )
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          use_sync_bn=False, norm_momentum=0.9, norm_epsilon=1e-5
      )
  )
  global_head: head_cfg.MLP = dataclasses.field(
      default_factory=lambda: head_cfg.MLP(  # pylint: disable=g-long-lambda
          use_sync_bn=False, normalize_inputs=False, norm_momentum=0.9
      )
  )
  local_head: head_cfg.InstanceReconstructor = dataclasses.field(
      default_factory=head_cfg.InstanceReconstructor
  )


@dataclasses.dataclass
class ConstCLLosses(hyperparams.Config):
  """The config for ConST-CL losses."""
  normalize_inputs: bool = True
  global_temperature: float = 0.1
  local_temperature: float = 0.2
  global_weight: float = 1.0
  local_weight: float = 0.001
  l2_weight_decay: float = 0.0


@exp_factory.register_config_factory('const_cl_pretrain_kinetics400')
def const_cl_pretrain_kinetics400() -> cfg.ExperimentConfig:
  """Pretrains SSL Video classification on Kinectics 400 with ResNet."""
  exp = video_classification.video_classification_kinetics400()
  exp.task = ConstCLPretrainTask(**exp.task.as_dict())
  exp.task.train_data.zero_centering_image = True
  exp.task.train_data = DataConfig(is_ssl=True, **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (16, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  exp.task.train_data.aug_min_area_ratio = 0.3
  exp.task.model = ConstCLModel()
  exp.task.model.model_type = 'const_cl_model'
  exp.task.losses = ConstCLLosses()
  return exp


@exp_factory.register_config_factory('const_cl_pretrain_kinetics600')
def const_cl_pretrain_kinetics600() -> cfg.ExperimentConfig:
  """Pretrains SSL Video classification on Kinectics 400 with ResNet."""
  exp = video_classification.video_classification_kinetics600()
  exp.task = ConstCLPretrainTask(**exp.task.as_dict())
  exp.task.train_data.zero_centering_image = True
  exp.task.train_data = DataConfig(is_ssl=True, **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (16, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  exp.task.train_data.aug_min_area_ratio = 0.3
  exp.task.model = ConstCLModel()
  exp.task.model.model_type = 'const_cl_model'
  exp.task.losses = ConstCLLosses()
  return exp
