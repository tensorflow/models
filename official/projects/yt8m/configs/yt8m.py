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

"""Video classification configuration definition."""
import dataclasses
from typing import Optional, Tuple
from absl import flags

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common

FLAGS = flags.FLAGS

YT8M_TRAIN_EXAMPLES = 3888919
YT8M_VAL_EXAMPLES = 1112356
# 2/frame -> frame level
# 3/frame -> segment level
YT8M_TRAIN_PATH = 'gs://youtube8m-ml/2/frame/train/train*.tfrecord'
YT8M_VAL_PATH = 'gs://youtube8m-ml/3/frame/validate/validate*.tfrecord'


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets."""
  name: Optional[str] = 'yt8m'
  split: Optional[str] = None
  feature_sizes: Tuple[int, ...] = (1024, 128)
  feature_names: Tuple[str, ...] = ('rgb', 'audio')
  segment_size: int = 1
  segment_labels: bool = False
  temporal_stride: int = 1
  max_frames: int = 300
  num_frames: int = 300  # set smaller to allow random sample (Parser)
  num_classes: int = 3862
  num_devices: int = 1
  input_path: str = ''
  is_training: bool = True
  random_seed: int = 123
  num_examples: int = -1


def yt8m(is_training):
  """YT8M dataset configs."""
  return DataConfig(
      num_frames=30,
      temporal_stride=1,
      segment_labels=False,
      segment_size=5,
      is_training=is_training,
      split='train' if is_training else 'valid',
      num_examples=YT8M_TRAIN_EXAMPLES if is_training else YT8M_VAL_EXAMPLES,
      input_path=YT8M_TRAIN_PATH if is_training else YT8M_VAL_PATH)


@dataclasses.dataclass
class MoeModel(hyperparams.Config):
  """The model config."""
  num_mixtures: int = 5
  l2_penalty: float = 1e-5
  use_input_context_gate: bool = False
  use_output_context_gate: bool = False


@dataclasses.dataclass
class DbofModel(hyperparams.Config):
  """The model config."""
  cluster_size: int = 3000
  hidden_size: int = 2000
  add_batch_norm: bool = True
  sample_random_frames: bool = True
  use_context_gate_cluster_layer: bool = False
  context_gate_cluster_bottleneck_size: int = 0
  pooling_method: str = 'average'
  yt8m_agg_classifier_model: str = 'MoeModel'
  agg_model: hyperparams.Config = MoeModel()
  norm_activation: common.NormActivation = common.NormActivation(
      activation='relu', use_sync_bn=False)


@dataclasses.dataclass
class Losses(hyperparams.Config):
  name: str = 'binary_crossentropy'
  from_logits: bool = False
  label_smoothing: float = 0.0
  l2_weight_decay: float = 1e-5


@dataclasses.dataclass
class YT8MTask(cfg.TaskConfig):
  """The task config."""
  model: DbofModel = DbofModel()
  train_data: DataConfig = yt8m(is_training=True)
  validation_data: DataConfig = yt8m(is_training=False)
  losses: Losses = Losses()
  gradient_clip_norm: float = 1.0
  num_readers: int = 8
  top_k: int = 20
  top_n: Optional[int] = None


def add_trainer(
    experiment: cfg.ExperimentConfig,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float = 0.0001,
    train_epochs: int = 50,
):
  """Add and config a trainer to the experiment config."""
  if YT8M_TRAIN_EXAMPLES <= 0:
    raise ValueError('Wrong train dataset size {!r}'.format(
        experiment.task.train_data))
  if YT8M_VAL_EXAMPLES <= 0:
    raise ValueError('Wrong validation dataset size {!r}'.format(
        experiment.task.validation_data))
  experiment.task.train_data.global_batch_size = train_batch_size
  experiment.task.validation_data.global_batch_size = eval_batch_size
  steps_per_epoch = YT8M_TRAIN_EXAMPLES // train_batch_size
  steps_per_loop = 30
  experiment.trainer = cfg.TrainerConfig(
      steps_per_loop=steps_per_loop,
      summary_interval=steps_per_loop,
      checkpoint_interval=steps_per_loop,
      train_steps=train_epochs * steps_per_epoch,
      validation_steps=YT8M_VAL_EXAMPLES // eval_batch_size,
      validation_interval=steps_per_loop,
      optimizer_config=optimization.OptimizationConfig({
          'optimizer': {
              'type': 'adam',
              'adam': {}
          },
          'learning_rate': {
              'type': 'exponential',
              'exponential': {
                  'initial_learning_rate': learning_rate,
                  'decay_rate': 0.95,
                  'decay_steps': int(steps_per_epoch * 1.5),
                  'offset': 500,
              }
          },
          'warmup': {
              'linear': {
                  'name': 'linear',
                  'warmup_learning_rate': 0,
                  'warmup_steps': 500,
              },
              'type': 'linear',
          }
      }))
  return experiment


@exp_factory.register_config_factory('yt8m_experiment')
def yt8m_experiment() -> cfg.ExperimentConfig:
  """Video classification general."""
  exp_config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YT8MTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
          'task.train_data.feature_sizes != None',
          'task.train_data.feature_names != None',
      ])

  # Per TPUv3 Core batch size 16GB HBM. `factor` in range(1, 26)
  factor = 1
  num_cores = 32  # for TPU 4x4
  train_per_core_bs = 32 * factor
  train_bs = train_per_core_bs * num_cores
  eval_per_core_bs = 32 * 50  # multiplier<=100
  eval_bs = eval_per_core_bs * num_cores
  # based lr=0.0001 for bs=512
  return add_trainer(
      exp_config,
      train_batch_size=train_bs,
      eval_batch_size=eval_bs,
      learning_rate=0.0001 * (train_bs / 512),
      train_epochs=100)
