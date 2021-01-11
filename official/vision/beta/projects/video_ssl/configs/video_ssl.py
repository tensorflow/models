# Lint as: python3
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
# ==============================================================================
"""Video classification configuration definition."""


import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.vision.beta.configs import video_classification


Losses = video_classification.Losses
VideoClassificationModel = video_classification.VideoClassificationModel
VideoClassificationTask = video_classification.VideoClassificationTask


@dataclasses.dataclass
class DataConfig(video_classification.DataConfig):
  """The base configuration for building datasets."""
  is_ssl: bool = False


@exp_factory.register_config_factory('video_ssl_pretrain_kinetics400')
def video_ssl_pretrain_kinetics400() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics400()
  exp.task.train_data = DataConfig(is_ssl=True, **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (16, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  return exp


@exp_factory.register_config_factory('video_ssl_linear_eval_kinetics400')
def video_ssl_linear_eval_kinetics400() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics400()
  exp.task.train_data = DataConfig(is_ssl=False,
                                   **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (32, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  exp.task.validation_data.feature_shape = (32, 256, 256, 3)
  exp.task.validation_data.temporal_stride = 2
  exp.task.validation_data = DataConfig(is_ssl=False,
                                        **exp.task.validation_data.as_dict())
  exp.task.validation_data.min_image_size = 256
  exp.task.validation_data.num_test_clips = 10
  return exp


@exp_factory.register_config_factory('video_ssl_pretrain_kinetics600')
def video_ssl_pretrain_kinetics600() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics600()
  exp.task.train_data = DataConfig(is_ssl=True, **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (16, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  return exp


@exp_factory.register_config_factory('video_ssl_linear_eval_kinetics600')
def video_ssl_linear_eval_kinetics600() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics600()
  exp.task.train_data = DataConfig(is_ssl=False,
                                   **exp.task.train_data.as_dict())
  exp.task.train_data.feature_shape = (32, 224, 224, 3)
  exp.task.train_data.temporal_stride = 2
  exp.task.validation_data = DataConfig(is_ssl=False,
                                        **exp.task.validation_data.as_dict())
  exp.task.validation_data.feature_shape = (32, 256, 256, 3)
  exp.task.validation_data.temporal_stride = 2
  exp.task.validation_data.min_image_size = 256
  exp.task.validation_data.num_test_clips = 10
  return exp
