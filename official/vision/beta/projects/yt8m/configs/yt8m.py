"""Video classification configuration definition."""
from typing import Optional, Tuple
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common

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
    split='train' if is_training else 'valid'
  )

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

class Losses(hyperparams.Config):
  name = 'binary_crossentropy'
  from_logits: bool = False
  label_smoothing: float = 0.0


class YT8MTask(cfg.TaskConfig):
  """The task config."""
  model: YT8MModel = YT8MModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()

def add_trainer(experiment: cfg.ExperimentConfig,
                train_batch_size: int,
                eval_batch_size: int,
                learning_rate: float = 1.6,
                train_epochs: int = 44,
                warmup_epochs: int = 5):
  """Add and config a trainer to the experiment config."""
  if experiment.task.train_data.num_examples <= 0:
    raise ValueError('Wrong train dataset size {!r}'.format(
      experiment.task.train_data))
  if experiment.task.validation_data.num_examples <= 0:
    raise ValueError('Wrong validation dataset size {!r}'.format(
      experiment.task.validation_data))
  experiment.task.train_data.global_batch_size = train_batch_size
  experiment.task.validation_data.global_batch_size = eval_batch_size
  steps_per_epoch = experiment.task.train_data.num_examples // train_batch_size
  experiment.trainer = cfg.TrainerConfig(
    steps_per_loop=steps_per_epoch,
    summary_interval=steps_per_epoch,
    checkpoint_interval=steps_per_epoch,
    train_steps=train_epochs * steps_per_epoch,
    validation_steps=experiment.task.validation_data.num_examples //
                     eval_batch_size,
    validation_interval=steps_per_epoch,
    optimizer_config=optimization.OptimizationConfig({
      'optimizer': {
        'type': 'sgd',
        'sgd': {
          'momentum': 0.9,
          'nesterov': True,
        }
      },
      'learning_rate': {
        'type': 'cosine',
        'cosine': {
          'initial_learning_rate': learning_rate,
          'decay_steps': train_epochs * steps_per_epoch,
        }
      },
      'warmup': {
        'type': 'linear',
        'linear': {
          'warmup_steps': warmup_epochs * steps_per_epoch,
          'warmup_learning_rate': 0
        }
      }
    }))
  return experiment

@exp_factory.register_config_factory('yt8m_experiment')
def yt8m_experiment() -> cfg.ExperimentConfig:
  """Video classification general."""
  return cfg.ExperimentConfig(
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


