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
    split='train' if is_training else 'valid',
  )

class YT8MModel(hyperparams.Config):
  """The model config."""
  num_classes : int = 3862, #TODO: get from reader, should be removed
  num_frames : int = 32,    #TODO: get from reader, should be removed
  iterations : int = 30
  cluster_size : int = 8192
  hidden_size : int = 1024
  add_batch_norm : bool = True
  sample_random_frame : bool = True
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



