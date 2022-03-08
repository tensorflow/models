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

"""Multi-task SimCLR configs."""

import dataclasses
from typing import List, Tuple

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling.multitask import configs as multitask_configs
from official.vision.beta.projects.simclr.configs import simclr as simclr_configs
from official.vision.beta.projects.simclr.modeling import simclr_model
from official.vision.configs import backbones
from official.vision.configs import common


@dataclasses.dataclass
class SimCLRMTHeadConfig(hyperparams.Config):
  """Per-task specific configs."""
  task_name: str = 'task_name'
  # Supervised head is required for finetune, but optional for pretrain.
  supervised_head: simclr_configs.SupervisedHead = simclr_configs.SupervisedHead(
      num_classes=1001)
  mode: str = simclr_model.PRETRAIN


@dataclasses.dataclass
class SimCLRMTModelConfig(hyperparams.Config):
  """Model config for multi-task SimCLR model."""
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  backbone_trainable: bool = True
  projection_head: simclr_configs.ProjectionHead = simclr_configs.ProjectionHead(
      proj_output_dim=128, num_proj_layers=3, ft_proj_idx=1)
  norm_activation: common.NormActivation = common.NormActivation(
      norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)
  heads: Tuple[SimCLRMTHeadConfig, ...] = ()
  # L2 weight decay is used in the model, not in task.
  # Note that this can not be used together with lars optimizer.
  l2_weight_decay: float = 0.0
  init_checkpoint: str = ''
  # backbone_projection or backbone
  init_checkpoint_modules: str = 'backbone_projection'


@exp_factory.register_config_factory('multitask_simclr')
def multitask_simclr() -> multitask_configs.MultiTaskExperimentConfig:
  return multitask_configs.MultiTaskExperimentConfig(
      task=multitask_configs.MultiTaskConfig(
          model=SimCLRMTModelConfig(
              heads=(SimCLRMTHeadConfig(
                  task_name='pretrain_simclr', mode=simclr_model.PRETRAIN),
                     SimCLRMTHeadConfig(
                         task_name='finetune_simclr',
                         mode=simclr_model.FINETUNE))),
          task_routines=(multitask_configs.TaskRoutine(
              task_name='pretrain_simclr',
              task_config=simclr_configs.SimCLRPretrainTask(),
              task_weight=2.0),
                         multitask_configs.TaskRoutine(
                             task_name='finetune_simclr',
                             task_config=simclr_configs.SimCLRFinetuneTask(),
                             task_weight=1.0))),
      trainer=multitask_configs.MultiTaskTrainerConfig())
