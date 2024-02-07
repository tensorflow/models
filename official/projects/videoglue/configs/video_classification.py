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

"""Video classification configuration definition."""
import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.videoglue.configs import backbones_3d
from official.projects.videoglue.configs import dataset
from official.vision.configs import common
from official.vision.configs import video_classification

VideoClassificationTask = video_classification.VideoClassificationTask
DataConfig = video_classification.DataConfig


@dataclasses.dataclass
class MultiHeadVideoClassificationModel(
    video_classification.VideoClassificationModel):
  """The model config."""
  model_type: str = 'mh_video_classification'
  backbone: backbones_3d.Backbone3D = dataclasses.field(
      default_factory=lambda: backbones_3d.Backbone3D(  # pylint: disable=g-long-lambda
          type='vit_3d', vit_3d=backbones_3d.VisionTransformer3D()
      )
  )
  classifier_type: str = 'pooler'  # 'linear' or 'pooler'
  # only useful when classifier_type == 'pooler'
  attention_num_heads: int = 6
  attention_hidden_size: int = 768
  attention_dropout_rate: float = 0.0
  add_temporal_pos_emb_pooler: bool = False


@dataclasses.dataclass
class MultiHeadVideoClassificationTask(VideoClassificationTask):
  """The task config."""
  _MHVCM = MultiHeadVideoClassificationModel
  model: _MHVCM = dataclasses.field(default_factory=_MHVCM)
  train_data: dataset.DataConfig = dataclasses.field(
      default_factory=lambda: dataset.DataConfig(  # pylint: disable=g-long-lambda
          is_training=True,
          data_augmentation=dataset.DataAugmentation(type='inception'),
      )
  )
  validation_data: dataset.DataConfig = dataclasses.field(
      default_factory=lambda: dataset.DataConfig(is_training=False)
  )


@exp_factory.register_config_factory('mh_video_classification')
def mh_video_classification() -> cfg.ExperimentConfig:
  """Multi-head video classification."""
  exp = video_classification.video_classification_kinetics400()
  task = MultiHeadVideoClassificationTask()
  exp.task = task
  return exp


@exp_factory.register_config_factory('mh_video_classification_strong_aug')
def mh_video_classification_strong_aug() -> cfg.ExperimentConfig:
  """Multi-head video classification with strong augmentation."""
  exp = video_classification.video_classification_kinetics400()
  task = MultiHeadVideoClassificationTask()
  task.train_data = dataset.DataConfig(
      is_training=True,
      data_augmentation=dataset.DataAugmentation(type='inception'),
      randaug=common.RandAugment(magnitude=9, magnitude_std=0.5),
      mixup_cutmix=common.MixupAndCutmix())
  exp.task = task
  return exp
