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
from typing import Optional, Tuple

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common


YT8M_TRAIN_EXAMPLES = 3888919
YT8M_VAL_EXAMPLES = 1112356
# 2/frame -> frame level
# 3/frame -> segment level
YT8M_TRAIN_PATH = 'gs://youtube8m-ml/2/frame/train/train*.tfrecord'
YT8M_VAL_PATH = 'gs://youtube8m-ml/3/frame/validate/validate*.tfrecord'


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets.

  Attributes:
    name: Dataset name.
    split: dataset split, 'train' or 'valid'.
    feature_sizes: shape(length) of each feature specified in the feature_names.
    feature_names: names of the features in the tf.SequenceExample.
    feature_sources: if the feature from 'context' or 'features'.
    feature_dtypes: dtype of decoded feature.
    feature_from_bytes: decode feature from bytes or as dtype list.
    label_fields: name of field to read from tf.SequenceExample.
    segment_size: Number of frames in each segment.
    segment_labels: Use segment level label. Default: False, video level label.
    include_video_id: `True` means include video id (string) in the input to the
      model.
    temporal_stride: Not used. Need to deprecated.
    max_frames: Maxim Number of frames in a input example. It is used to crop
      the input in the temporal dimension.
    sample_random_frames: If sample random frames or random sequence.
    num_sample_frames: Number of frames to sample for each input example. No
      frame sampling if None.
    num_classes: Number of classes to classify. Assuming it is a classification
      task.
    num_devices: Not used. To be deprecated.
    input_path: The path to the input.
    is_training: Whether this data is used for training or not.
    num_examples: Number of examples in the dataset. It is used to compute the
      steps for train or eval. set the value to `-1` to make the experiment run
      until the end of dataset.
    file_type: type of input files.
  """
  name: Optional[str] = 'yt8m'
  split: Optional[str] = None
  feature_sizes: Tuple[int, ...] = (1024, 128)
  feature_names: Tuple[str, ...] = ('rgb', 'audio')
  feature_sources: Tuple[str, ...] = ('feature', 'feature')
  feature_dtypes: Tuple[str, ...] = ('uint8', 'uint8')
  feature_from_bytes: Tuple[bool, ...] = (True, True)
  label_field: str = 'labels'
  segment_size: int = 1
  segment_labels: bool = False
  include_video_id: bool = False
  temporal_stride: int = 1
  max_frames: int = 300  # Cap input frames.
  sample_random_frames: bool = True
  # Sample random frames if not None. No sampling in inference.
  num_sample_frames: Optional[int] = 300
  input_per_feature_l2_norm: bool = False
  prefetch_buffer_size: int = 100
  shuffle_buffer_size: int = 100
  num_classes: int = 3862
  num_devices: int = 1
  input_path: str = ''
  is_training: bool = True
  num_examples: int = -1
  file_type: str = 'tfrecord'


def yt8m(is_training):
  """YT8M dataset configs."""
  # pylint: disable=unexpected-keyword-arg
  return DataConfig(
      temporal_stride=1,
      segment_labels=False,
      segment_size=5,
      is_training=is_training,
      split='train' if is_training else 'valid',
      drop_remainder=is_training,  # pytype: disable=wrong-keyword-args
      num_examples=YT8M_TRAIN_EXAMPLES if is_training else YT8M_VAL_EXAMPLES,
      input_path=YT8M_TRAIN_PATH if is_training else YT8M_VAL_PATH)
  # pylint: enable=unexpected-keyword-arg


@dataclasses.dataclass
class DbofModel(hyperparams.Config):
  """The model config."""
  cluster_size: int = 3000
  hidden_size: int = 2000
  add_batch_norm: bool = True
  pooling_method: str = 'average'
  use_context_gate_cluster_layer: bool = False
  context_gate_cluster_bottleneck_size: int = 0


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    dbof: dbof backbone config.
  """
  type: Optional[str] = None
  dbof: DbofModel = dataclasses.field(default_factory=DbofModel)


@dataclasses.dataclass
class MoeModel(hyperparams.Config):
  """The MoE model config."""

  num_mixtures: int = 5
  vocab_as_last_dim: bool = False
  use_input_context_gate: bool = False
  use_output_context_gate: bool = False


@dataclasses.dataclass
class LogisticModel(hyperparams.Config):
  """The logistic model config."""
  return_logits: bool = False


@dataclasses.dataclass
class Head(hyperparams.OneOfConfig):
  """Configuration for aggreagation heads.

  Attributes:
    type: 'str', type of head be used, one of the fields below.
    moe: MoE head config.
    logistic: Logistic head config.
  """
  type: Optional[str] = None
  moe: MoeModel = dataclasses.field(default_factory=MoeModel)
  logistic: LogisticModel = dataclasses.field(default_factory=LogisticModel)


@dataclasses.dataclass
class VideoClassificationModel(hyperparams.Config):
  """The classifier model config."""
  backbone: Backbone = dataclasses.field(
      default_factory=lambda: Backbone(type='dbof')
  )
  head: Head = dataclasses.field(default_factory=lambda: Head(type='moe'))
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          activation='relu', use_sync_bn=False
      )
  )


@dataclasses.dataclass
class Losses(hyperparams.Config):
  name: str = 'binary_crossentropy'
  from_logits: bool = False
  label_smoothing: float = 0.0
  l2_weight_decay: float = 1e-5


@dataclasses.dataclass
class AveragePrecisionConfig(hyperparams.Config):
  top_k: int = 20
  top_n: Optional[int] = None
  return_per_class_ap: bool = False


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  average_precision: Optional[AveragePrecisionConfig] = None


@dataclasses.dataclass
class YT8MTask(cfg.TaskConfig):
  """The task config."""
  model: VideoClassificationModel = dataclasses.field(
      default_factory=VideoClassificationModel
  )
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: yt8m(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: yt8m(is_training=False)
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  evaluation: Evaluation = dataclasses.field(
      default_factory=lambda: Evaluation(  # pylint: disable=g-long-lambda
          average_precision=AveragePrecisionConfig()
      )
  )
  gradient_clip_norm: float = 1.0


def add_trainer(
    experiment: cfg.ExperimentConfig,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float = 0.0001,
    train_epochs: int = 50,
    num_train_examples: int = YT8M_TRAIN_EXAMPLES,
    num_val_examples: int = YT8M_VAL_EXAMPLES,
) -> cfg.ExperimentConfig:
  """Adds and config a trainer to the experiment config."""
  if num_train_examples <= 0:
    raise ValueError('Wrong train dataset size {!r}'.format(
        experiment.task.train_data))
  if num_val_examples <= 0:
    raise ValueError('Wrong validation dataset size {!r}'.format(
        experiment.task.validation_data))
  experiment.task.train_data.global_batch_size = train_batch_size
  experiment.task.validation_data.global_batch_size = eval_batch_size
  steps_per_epoch = num_train_examples // train_batch_size
  steps_per_loop = 500
  experiment.trainer = cfg.TrainerConfig(
      steps_per_loop=steps_per_loop,
      summary_interval=steps_per_loop,
      checkpoint_interval=steps_per_loop,
      train_steps=train_epochs * steps_per_epoch,
      validation_steps=num_val_examples // eval_batch_size,
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
          'task.train_data.feature_sources != None',
          'task.train_data.feature_dtypes != None',
      ])

  # Per TPUv3 Core batch size 16GB HBM. `factor` in range(1, 26)
  factor = 1
  num_cores = 32  # for TPUv3 4x4
  train_per_core_bs = 32 * factor
  train_bs = train_per_core_bs * num_cores
  eval_per_core_bs = 4 * 50  # multiplier<=100
  eval_bs = eval_per_core_bs * num_cores
  # based lr=0.0001 for bs=512
  return add_trainer(
      exp_config,
      train_batch_size=train_bs,
      eval_batch_size=eval_bs,
      learning_rate=0.0001 * (train_bs / 512),
      train_epochs=100)
