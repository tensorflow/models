# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from dataclasses import dataclass

from official.vision.image_classification import dataset_factory


@dataclass
class ImageNetteConfig(dataset_factory.DatasetConfig):
  """The base ImageNette dataset config.

  Imagenette is a subset of 10 easily classified classes from the Imagenet
  dataset. It was originally prepared by Jeremy Howard of FastAI.
  The objective behind putting together a small version of the Imagenet dataset
  was mainly because running new ideas/algorithms/experiments on the whole
  Imagenet take a lot of time.
  """
  name: str = 'imagenette'
  image_size: int = 224
  num_channels: int = 3
  batch_size: int = 32
  num_classes: int = 10
  num_examples: int = 9469
  num_eval_examples: int = 3925
  mean_subtract: bool = True
  standardize: bool = True
  augmenter: dataset_factory.AugmentConfig = dataset_factory.AugmentConfig(
    name='autoaugment'
  )

  download: bool = True
  builder: str = 'tfds'


@dataclass
class ImageNetConfig(dataset_factory.DatasetConfig):
  """The base ImageNet dataset config."""
  name: str = 'imagenet2012'
  image_size: int = 224
  num_channels: int = 3
  batch_size: int = 32
  num_classes: int = 1000
  num_examples: int = 1281167
  num_eval_examples: int = 50000
  dtype: str = 'float32'
  mean_subtract: bool = True
  standardize: bool = True
  augmenter: dataset_factory.AugmentConfig = dataset_factory.AugmentConfig(
    name='autoaugment'
  )

  download: bool = False
  builder: str = 'records'
  data_dir: str = 'gs://'
  shuffle_buffer_size: int = 10000
  # each file is around 130M, and in total 1000 files
  file_shuffle_buffer_size: int = 100


@dataclass
class MNISTConfig(dataset_factory.DatasetConfig):
  """The base MNIST dataset config."""
  name: str = 'mnist'
  image_size: int = 28
  num_channels: int = 1
  batch_size: int = 128
  num_classes: int = 10
  num_examples: int = 60000
  download: bool = True
  builder: str = 'tfds'
  mean_subtract: bool = True
  standardize: bool = True