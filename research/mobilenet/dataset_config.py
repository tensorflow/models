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


from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class DatasetConfig:
  """The base configuration for building datasets.

  Attributes:
    name: The name of the Dataset. Usually should correspond to a TFDS dataset.
    data_dir: The path where the dataset files are stored, if available.
    filenames: Optional list of strings representing the TFRecord names.
    builder: The builder type used to load the dataset. Value should be one of
      'tfds' (load using TFDS), 'records' (load from TFRecords), or 'synthetic'
      (generate dummy synthetic data without reading from files).
    split: The split of the dataset. Usually 'train', 'validation', or 'test'.
    image_size: The size of the image in the dataset. This assumes that
      `width` == `height`. Set to 'infer' to infer the image size from TFDS
      info. This requires `name` to be a registered dataset in TFDS.
    num_classes: The number of classes given by the dataset. Set to 'infer'
      to infer the image size from TFDS info. This requires `name` to be a
      registered dataset in TFDS.
    num_channels: The number of channels given by the dataset. Set to 'infer'
      to infer the image size from TFDS info. This requires `name` to be a
      registered dataset in TFDS.
    num_examples: The number of examples given by the dataset. Set to 'infer'
      to infer the image size from TFDS info. This requires `name` to be a
      registered dataset in TFDS.
    batch_size: The base batch size for the dataset.
    use_per_replica_batch_size: Whether to scale the batch size based on
      available resources. If set to `True`, the dataset builder will return
      batch_size multiplied by `num_devices`, the number of device replicas
      (e.g., the number of GPUs or TPU cores). This setting should be `True` if
      the strategy argument is passed to `build()` and `num_devices > 1`.
    num_devices: The number of replica devices to use. This should be set by
      `strategy.num_replicas_in_sync` when using a distribution strategy.
    dtype: The desired dtype of the dataset. This will be set during
      preprocessing.
    one_hot: Whether to apply one hot encoding. Set to `True` to be able to use
      label smoothing.
    download: Whether to download data using TFDS.
    shuffle_buffer_size: The buffer size used for shuffling training data.
    file_shuffle_buffer_size: The buffer size used for shuffling raw training
      files.
    skip_decoding: Whether to skip image decoding when loading from TFDS.
    deterministic_train: Whether the examples in the training set should output
      in a deterministic order.
    use_slack: whether to introduce slack in the last prefetch. This may reduce
      CPU contention at the start of a training step.
    cache: whether to cache to dataset examples. Can be used to avoid re-reading
      from disk on the second epoch. Requires significant memory overhead.
    mean_subtract: whether or not to apply mean subtraction to the dataset.
    standardize: whether or not to apply standardization to the dataset.
  """
  name: Optional[str] = None
  data_dir: Optional[str] = None
  filenames: Optional[List[str]] = None
  builder: str = 'tfds'
  split: str = 'train'
  image_size: Union[int, str] = 'infer'
  num_classes: Union[int, str] = 'infer'
  num_channels: Union[int, str] = 3
  num_examples: Union[int, str] = 'infer'
  batch_size: int = 128
  use_per_replica_batch_size: bool = True
  num_devices: int = 1
  dtype: str = 'float32'
  one_hot: bool = True
  download: bool = False
  shuffle_buffer_size: int = 10000
  file_shuffle_buffer_size: int = 1024
  skip_decoding: bool = True
  deterministic_train: bool = False
  use_slack: bool = True
  cache: bool = False
  mean_subtract: bool = False
  standardize: bool = False

  @property
  def has_data(self):
    """Whether this dataset is has any data associated with it."""
    return self.name or self.data_dir or self.filenames


@dataclass
class ImageNetTEConfig(DatasetConfig):
  """The base ImageNette dataset config."""
  name: str = 'imagenette'
  image_size: int = 224
  batch_size: int = 32
  num_classes: int = 10
  num_examples: int = 9469
  download: bool = True


@dataclass
class ImageNetConfig(DatasetConfig):
  """The base ImageNet dataset config."""
  name: str = 'imagenet2012'
  image_size: int = 224
  batch_size: int = 32
  num_classes: int = 1000
  num_examples: int = 1281167
  download: bool = False


@dataclass
class MNISTConfig(DatasetConfig):
  """The base MNIST dataset config."""
  name: str = 'mnist'
  image_size: int = 28
  batch_size: int = 128
  num_classes: int = 10
  num_examples: int = 60000
  download: bool = True
