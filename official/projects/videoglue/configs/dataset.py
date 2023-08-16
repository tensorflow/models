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

"""Video coarse classification configuration definition."""

import dataclasses
from typing import Optional, Tuple, Union, List

from official.modeling import hyperparams
from official.vision.configs import common as common_cfg


@dataclasses.dataclass
class VGG(hyperparams.Config):
  """Config for VGG style data augmentation."""
  pass


@dataclasses.dataclass
class Inception(hyperparams.Config):
  """Config for Inception style data augmentation."""
  min_aspect_ratio: float = 0.5
  max_aspect_ratio: float = 2.0
  min_area_ratio: float = 0.3
  max_area_ratio: float = 1.0


@dataclasses.dataclass
class AVA(hyperparams.Config):
  """Config for AVA style data augmentation."""
  scale_min: float = 0.5
  scale_max: float = 2.0


@dataclasses.dataclass
class DataAugmentation(hyperparams.OneOfConfig):
  """Configuration for data augmentation.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    inception: resnet backbone config.
    vgg: dilated resnet backbone for semantic segmentation config.
    ava: revnet backbone config.
  """
  type: Optional[str] = None
  vgg: VGG = dataclasses.field(default_factory=VGG)
  inception: Inception = dataclasses.field(default_factory=Inception)
  ava: AVA = dataclasses.field(default_factory=AVA)


@dataclasses.dataclass
class DataConfig(hyperparams.Config):
  """The base configuration for building datasets."""
  name: str = 'some_dataset'
  is_training: bool = False
  num_classes: Union[int, List[int]] = 1
  label_names: Union[str, List[str]] = 'label'
  num_examples: int = 10000
  global_batch_size: int = 128
  feature_shape: Tuple[int, ...] = (64, 224, 224, 3)
  min_resize: int = 256
  temporal_stride: int = 5
  sample_from_segments: bool = False
  zero_centering_image: bool = True
  random_flip_image: bool = True
  num_test_clips: int = 1
  num_test_crops: int = 1
  data_augmentation: DataAugmentation = dataclasses.field(
      default_factory=lambda: DataAugmentation(type='vgg')
  )
  randaug: Optional[common_cfg.RandAugment] = None
  autoaug: Optional[common_cfg.AutoAugment] = None
  mixup_cutmix: Optional[common_cfg.MixupAndCutmix] = None
  is_multilabel: bool = False
  # Pipeline parameters.
  drop_remainder: bool = True
  dtype: str = 'float32'
  prefetch_buffer_size: int = 512
  shuffle_buffer_size: int = 256
  num_process_threads: int = 32
  num_parallel_calls_interleave: int = 32
  cycle_length: int = 32
  block_length: int = 1
  cache: bool = False
  use_tf_data_service: bool = False
