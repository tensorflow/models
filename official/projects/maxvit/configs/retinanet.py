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

"""RetinaNet configuration definition."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.maxvit.configs import backbones
from official.vision.configs import retinanet


@exp_factory.register_config_factory('retinanet_maxvit_coco')
def retinanet_maxvit_coco() -> cfg.ExperimentConfig:
  """COCO object detection with RetinaNet using MaxViT backbone."""
  config = retinanet.retinanet_resnetfpn_coco()
  config.task.model.backbone = backbones.Backbone(
      type='maxvit', maxvit=backbones.MaxViT(
          model_name='maxvit-base',
          window_size=20,
          grid_size=20,
          scale_ratio='20/7',
          survival_prob=0.7,
      )
  )
  config.task.validation_data.global_batch_size = 32
  config.trainer.validation_steps = 156
  config.trainer.validation_interval = 1560
  return config
