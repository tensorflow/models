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
from official.vision.configs import common
from official.vision.configs import video_classification


Losses = video_classification.Losses
VideoClassificationModel = video_classification.VideoClassificationModel
VideoClassificationTask = video_classification.VideoClassificationTask


@dataclasses.dataclass
class DataConfig(video_classification.DataConfig):
  """The base configuration for building datasets."""
  is_ssl: bool = False
  is_training: bool = True
  drop_remainder: bool = True


@dataclasses.dataclass
class VideoSSLModel(VideoClassificationModel):
  """The model config."""
  normalize_feature: bool = False
  hidden_dim: int = 2048
  hidden_layer_num: int = 3
  projection_dim: int = 128
  hidden_norm_activation: common.NormActivation = common.NormActivation(
      use_sync_bn=False, norm_momentum=0.997, norm_epsilon=1.0e-05)


@dataclasses.dataclass
class SSLLosses(Losses):
  normalize_hidden: bool = True
  temperature: float = 0.1


@dataclasses.dataclass
class VideoSSLPretrainTask(VideoClassificationTask):
  model: VideoSSLModel = VideoSSLModel()
  losses: SSLLosses = SSLLosses()
  train_data: DataConfig = DataConfig(is_training=True, drop_remainder=True)
  validation_data: DataConfig = DataConfig(
      is_training=False, drop_remainder=False)
  losses: SSLLosses = SSLLosses()


@dataclasses.dataclass
class VideoSSLEvalTask(VideoClassificationTask):
  model: VideoSSLModel = VideoSSLModel()
  train_data: DataConfig = DataConfig(is_training=True, drop_remainder=True)
  validation_data: DataConfig = DataConfig(
      is_training=False, drop_remainder=False)
  losses: SSLLosses = SSLLosses()


@exp_factory.register_config_factory('video_ssl_pretrain_kinetics400')
def video_ssl_pretrain_kinetics400() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics400()
  task = VideoSSLPretrainTask()
  task.override(exp.task)
  task.train_data.is_ssl = True
  task.train_data.feature_shape = (16, 224, 224, 3)
  task.train_data.temporal_stride = 2
  task.model.model_type = 'video_ssl_model'
  exp.task = task
  return exp


@exp_factory.register_config_factory('video_ssl_linear_eval_kinetics400')
def video_ssl_linear_eval_kinetics400() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics400()
  task = VideoSSLEvalTask()  # Replaces the task type.
  task.override(exp.task)
  task.train_data.is_ssl = False
  task.train_data.feature_shape = (32, 224, 224, 3)
  task.train_data.temporal_stride = 2
  task.validation_data.is_ssl = False
  task.validation_data.feature_shape = (32, 256, 256, 3)
  task.validation_data.temporal_stride = 2
  task.validation_data.min_image_size = 256
  task.validation_data.num_test_clips = 10
  task.validation_data.num_test_crops = 3
  task.model.model_type = 'video_ssl_model'
  task.model.normalize_feature = True
  task.model.hidden_layer_num = 0
  task.model.projection_dim = 600
  exp.task = task
  return exp


@exp_factory.register_config_factory('video_ssl_pretrain_kinetics600')
def video_ssl_pretrain_kinetics600() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics600()
  task = VideoSSLPretrainTask()
  task.override(exp.task)
  task.train_data.is_ssl = True
  task.train_data.feature_shape = (16, 224, 224, 3)
  task.train_data.temporal_stride = 2
  task.model.model_type = 'video_ssl_model'
  exp.task = task
  return exp


@exp_factory.register_config_factory('video_ssl_linear_eval_kinetics600')
def video_ssl_linear_eval_kinetics600() -> cfg.ExperimentConfig:
  """Pretrain SSL Video classification on Kinectics 400 with resnet."""
  exp = video_classification.video_classification_kinetics600()
  task = VideoSSLEvalTask()  # Replaces the task type.
  task.override(exp.task)
  task.train_data.is_ssl = False
  task.train_data.feature_shape = (32, 224, 224, 3)
  task.train_data.temporal_stride = 2
  task.validation_data.is_ssl = False
  task.validation_data.feature_shape = (32, 256, 256, 3)
  task.validation_data.temporal_stride = 2
  task.validation_data.min_image_size = 256
  task.validation_data.num_test_clips = 10
  task.validation_data.num_test_crops = 3
  task.model.model_type = 'video_ssl_model'
  task.model.normalize_feature = True
  task.model.hidden_layer_num = 0
  task.model.projection_dim = 600
  exp.task = task
  return exp
