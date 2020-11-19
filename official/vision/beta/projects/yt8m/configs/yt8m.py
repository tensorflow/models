"""Video classification configuration definition."""
from typing import Optional, Tuple
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import backbones_3d
from official.vision.beta.configs import common

class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets."""
  name: Optional[str] = None
  file_type: Optional[str] = 'tfrecord'
  compressed_input: bool = False
  split: str = 'train'
  feature_shape: Tuple[int, ...] = (64, 224, 224, 3)
  temporal_stride: int = 1
  num_test_clips: int = 1
  num_classes: int = -1
  num_channels: int = 3
  num_examples: int = -1
  global_batch_size: int = 128
  num_devices: int = 1
  data_format: str = 'channels_last'
  dtype: str = 'float32'
  one_hot: bool = True
  shuffle_buffer_size: int = 64
  cache: bool = False
  input_path: str = ''
  is_training: bool = True
  cycle_length: int = 10
  min_image_size: int = 256

class YT8MTask(Task):
  """The task config."""



class DbofModel(Config):
  """The model config."""