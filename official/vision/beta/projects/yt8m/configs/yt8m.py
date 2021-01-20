"""Video classification configuration definition."""
from typing import Optional, Tuple
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common
from absl import flags
FLAGS = flags.FLAGS

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets."""
  name: Optional[str] = None
  file_type: Optional[str] = 'tfrecord'
  split: str = 'train'
  feature_sizes: Tuple[int, ...] = None
  feature_names: Tuple[str, ...] = None
  segment_size: int = 1
  segment_labels: bool = False
  temporal_stride: int = 1
  max_frames: int = -1
  num_classes: int = -1
  num_channels: int = 3
  num_devices: int = 1
  dtype: str = 'float32'
  input_path: str = ''
  is_training: bool = True


def yt8m(is_training):
  """YT8M dataset configs."""
  return DataConfig(
    name='yt8m',
    num_classes=3862,
    feature_sizes=[1024, 128],
    feature_names=["rgb", "audio"],
    max_frames=300,
    segment_labels=False,
    segment_size=5,
    is_training=is_training,
    split='train' if is_training else 'valid',
  )

YT8M_TRAIN_EXAMPLES = 48800 #TODO: get actual numbers
YT8M_VAL_EXAMPLES = 12200

@dataclasses.dataclass
class YT8MModel(hyperparams.Config):
  """The model config."""
  iterations : int = 30
  cluster_size : int = 8192
  hidden_size : int = 1024
  add_batch_norm : bool = True
  sample_random_frames : bool = True
  is_training : bool = True
  activation : str = "sigmoid"
  pooling_method : str = "max"
  yt8m_agg_classifier_model : str = "MoeModel"
  frame_features : bool = False
  segment_labels : bool = False
  start_new_model : bool = True

@dataclasses.dataclass
class Losses(hyperparams.Config):
  name = 'binary_crossentropy'
  from_logits: bool = False
  label_smoothing: float = 0.0

@dataclasses.dataclass
class YT8MTask(cfg.TaskConfig):
  """The task config."""
  model: YT8MModel = YT8MModel()
  train_data: DataConfig = yt8m(is_training=True)
  validation_data: DataConfig = yt8m(is_training=False)
  gradient_clip_norm: float = 1.0
  losses: Losses = Losses()
  num_readers: int = 8
  top_k: int = 20
  top_n: int = None

def add_trainer(experiment: cfg.ExperimentConfig,
                train_batch_size: int,
                eval_batch_size: int,
                learning_rate: float = 0.01,
                train_epochs: int = 44,
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
  experiment.trainer = cfg.TrainerConfig(
    steps_per_loop=steps_per_epoch,
    summary_interval=steps_per_epoch,
    checkpoint_interval=steps_per_epoch,
    train_steps=train_epochs * steps_per_epoch,
    validation_steps=YT8M_VAL_EXAMPLES //
                     eval_batch_size,
    validation_interval=steps_per_epoch,
    optimizer_config=optimization.OptimizationConfig({
      'optimizer': {
        'type': 'adam',
        'adam': {
        }
      },
      'learning_rate': {
        'type': 'exponential',
        'exponential': {
          'initial_learning_rate': learning_rate,
          'decay_rate': 0.95,
          'decay_steps': 4000000,
        }
      },
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
      'task.train_data.feature_names != None'
    ])

  return add_trainer(exp_config, train_batch_size=1024, eval_batch_size=1024)


