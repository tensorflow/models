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

# pylint: disable=line-too-long
# type: ignore
"""Configuration definitions for MobilenetEdgeTPU losses, learning rates, optimizers, and training."""
import dataclasses
import os
from typing import Any, Mapping, Optional

# Import libraries

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import common
from official.vision.beta.configs import image_classification as base_config


@dataclasses.dataclass
class MobilenetEdgeTPUModelConfig(base_config.ImageClassificationModel):
  """Configuration for the MobilenetEdgeTPU model.

  Attributes:
    name: The name of the model. Defaults to 'MobilenetEdgeTPU'.
    model_params: A dictionary that represents the parameters of the
      EfficientNet model. These will be passed in to the "from_name" function.
  """
  model_params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {  # pylint: disable=g-long-lambda
          'model_name': 'mobilenet_edgetpu_v2_xs',
          'model_weights_path': '',
          'checkpoint_format': 'tf_checkpoint',
          'overrides': {
              'batch_norm': 'tpu',
              'num_classes': 1001,
              'rescale_input': False,
              'dtype': 'bfloat16'
          }
      })


@dataclasses.dataclass
class MobilenetEdgeTPUTaskConfig(base_config.ImageClassificationTask):
  """Task defination for MobileNetEdgeTPU.

  Attributes:
    model: A `ModelConfig` instance.
    saved_model_path: Instead of initializing a model from the model config,
      the model can be loaded from a file path.
  """
  model: MobilenetEdgeTPUModelConfig = MobilenetEdgeTPUModelConfig()
  saved_model_path: Optional[str] = None


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


def mobilenet_edgetpu_base_experiment_config(
    model_name: str) -> cfg.ExperimentConfig:
  """Image classification on imagenet with mobilenet_edgetpu.

  Experiment config common across all mobilenet_edgetpu variants.
  Args:
    model_name: Name of the mobilenet_edgetpu model variant
  Returns:
    ExperimentConfig
  """
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  mobilenet_edgetpu_config = MobilenetEdgeTPUModelConfig(
      num_classes=1001, input_size=[224, 224, 3])
  mobilenet_edgetpu_config.model_params.model_name = model_name
  config = cfg.ExperimentConfig(
      task=MobilenetEdgeTPUTaskConfig(
          model=mobilenet_edgetpu_config,
          losses=base_config.Losses(label_smoothing=0.1),
          train_data=base_config.DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              dtype='bfloat16',
              aug_type=common.Augmentation(type='autoaug')),
          validation_data=base_config.DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              dtype='bfloat16',
              drop_remainder=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch * 5,
          max_to_keep=10,
          train_steps=550 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'rmsprop',
                  'rmsprop': {
                      'rho': 0.9,
                      'momentum': 0.9,
                      'epsilon': 0.001,
                  }
              },
              'ema': {
                  'average_decay': 0.99,
                  'trainable_weights_only': False,
              },
              'learning_rate': {
                  'type': 'exponential',
                  'exponential': {
                      'initial_learning_rate':
                          0.008 * (train_batch_size // 128),
                      'decay_steps':
                          int(2.4 * steps_per_epoch),
                      'decay_rate':
                          0.97,
                      'staircase':
                          True
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


# Registration for MobileNet-EdgeTPU-Search models.
# When this config is used, users need to specify the saved model path via
# --params_override=task.saved_model_path='your/saved_model/path/'.
@exp_factory.register_config_factory('mobilenet_edgetpu_search')
def mobilenet_edgetpu_search() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_search')


# Registration for MobileNet-EdgeTPU-V2 models.
@exp_factory.register_config_factory('mobilenet_edgetpu_v2_tiny')
def mobilenet_edgetpu_v2_tiny() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_v2_tiny')


# Registration for MobileNet-EdgeTPU-V2 models.
@exp_factory.register_config_factory('mobilenet_edgetpu_v2_xs')
def mobilenet_edgetpu_v2_xs() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_v2_xs')


@exp_factory.register_config_factory('mobilenet_edgetpu_v2_s')
def mobilenet_edgetpu_v2_s() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_v2_s')


@exp_factory.register_config_factory('mobilenet_edgetpu_v2_m')
def mobilenet_edgetpu_v2_m() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_v2_m')


@exp_factory.register_config_factory('mobilenet_edgetpu_v2_l')
def mobilenet_edgetpu_v2_l() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_v2_l')


# Registration for MobileNet-EdgeTPU-V1 models.
@exp_factory.register_config_factory('mobilenet_edgetpu')
def mobilenet_edgetpu() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu')


# Registration for MobileNet-EdgeTPU-V1 models.
# We use 'depth_multiplier' to scale the models.
# E.g. dm0p75 implies depth multiplier of 0.75x
@exp_factory.register_config_factory('mobilenet_edgetpu_dm0p75')
def mobilenet_edgetpu_dm0p75() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_dm0p75')


@exp_factory.register_config_factory('mobilenet_edgetpu_dm1p25')
def mobilenet_edgetpu_dm1p25() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_dm1p25')


@exp_factory.register_config_factory('mobilenet_edgetpu_dm1p5')
def mobilenet_edgetpu_dm1p5() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_dm1p5')


@exp_factory.register_config_factory('mobilenet_edgetpu_dm1p75')
def mobilenet_edgetpu_dm1p75() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('mobilenet_edgetpu_dm1p75')


# Registration for AutoSeg-EdgeTPU backbones
@exp_factory.register_config_factory('autoseg_edgetpu_backbone_xs')
def autoseg_edgetpu_backbone_xs() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('autoseg_edgetpu_backbone_xs')


@exp_factory.register_config_factory('autoseg_edgetpu_backbone_s')
def autoseg_edgetpu_backbone_s() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('autoseg_edgetpu_backbone_s')


@exp_factory.register_config_factory('autoseg_edgetpu_backbone_m')
def autoseg_edgetpu_backbone_m() -> cfg.ExperimentConfig:
  return mobilenet_edgetpu_base_experiment_config('autoseg_edgetpu_backbone_m')
