# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Dataset utilities for vision tasks using TFDS and tf.data.Dataset."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from typing import Any, List, Optional, Tuple, Mapping, Union
from absl import logging
from dataclasses import dataclass
import tensorflow as tf
import tensorflow_datasets as tfds

from official.modeling.hyperparams import base_config
from official.vision.image_classification import augment
from official.vision.image_classification import preprocessing


AUGMENTERS = {
    'autoaugment': augment.AutoAugment,
    'randaugment': augment.RandAugment,
}


@dataclass
class AugmentConfig(base_config.Config):
  """Configuration for image augmenters.

  Attributes:
    name: The name of the image augmentation to use. Possible options are
      None (default), 'autoaugment', or 'randaugment'.
    params: Any paramaters used to initialize the augmenter.
  """
  name: Optional[str] = None
  params: Optional[Mapping[str, Any]] = None

  def build(self) -> augment.ImageAugment:
    """Build the augmenter using this config."""
    params = self.params or {}
    augmenter = AUGMENTERS.get(self.name, None)
    return augmenter(**params) if augmenter is not None else None


@dataclass
class DatasetConfig(base_config.Config):
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
    augmenter: The augmenter config to use. No augmentation is used by default.
    download: Whether to download data using TFDS.
    shuffle_buffer_size: The buffer size used for shuffling training data.
    file_shuffle_buffer_size: The buffer size used for shuffling raw training
      files.
    skip_decoding: Whether to skip image decoding when loading from TFDS.
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
  num_channels: Union[int, str] = 'infer'
  num_examples: Union[int, str] = 'infer'
  batch_size: int = 128
  use_per_replica_batch_size: bool = True
  num_devices: int = 1
  dtype: str = 'float32'
  one_hot: bool = True
  augmenter: AugmentConfig = AugmentConfig()
  download: bool = False
  shuffle_buffer_size: int = 10000
  file_shuffle_buffer_size: int = 1024
  skip_decoding: bool = True
  cache: bool = False
  mean_subtract: bool = False
  standardize: bool = False

  @property
  def has_data(self):
    """Whether this dataset is has any data associated with it."""
    return self.name or self.data_dir or self.filenames


@dataclass
class ImageNetConfig(DatasetConfig):
  """The base ImageNet dataset config."""
  name: str = 'imagenet2012'
  # Note: for large datasets like ImageNet, using records is faster than tfds
  builder: str = 'records'
  image_size: int = 224
  batch_size: int = 128


@dataclass
class Cifar10Config(DatasetConfig):
  """The base CIFAR-10 dataset config."""
  name: str = 'cifar10'
  image_size: int = 224
  batch_size: int = 128
  download: bool = True
  cache: bool = True


class DatasetBuilder:
  """An object for building datasets.

  Allows building various pipelines fetching examples, preprocessing, etc.
  Maintains additional state information calculated from the dataset, i.e.,
  training set split, batch size, and number of steps (batches).
  """

  def __init__(self, config: DatasetConfig, **overrides: Any):
    """Initialize the builder from the config."""
    self.config = config.replace(**overrides)
    self.builder_info = None

    if self.config.augmenter is not None:
      logging.info('Using augmentation: %s', self.config.augmenter.name)
      self.augmenter = self.config.augmenter.build()
    else:
      self.augmenter = None

  @property
  def is_training(self) -> bool:
    """Whether this is the training set."""
    return self.config.split == 'train'

  @property
  def batch_size(self) -> int:
    """The batch size, multiplied by the number of replicas (if configured)."""
    if self.config.use_per_replica_batch_size:
      return self.config.batch_size * self.config.num_devices
    else:
      return self.config.batch_size

  @property
  def global_batch_size(self):
    """The global batch size across all replicas."""
    return self.batch_size

  @property
  def local_batch_size(self):
    """The base unscaled batch size."""
    if self.config.use_per_replica_batch_size:
      return self.config.batch_size
    else:
      return self.config.batch_size // self.config.num_devices

  @property
  def num_steps(self) -> int:
    """The number of steps (batches) to exhaust this dataset."""
    # Always divide by the global batch size to get the correct # of steps
    return self.num_examples // self.global_batch_size

  @property
  def dtype(self) -> tf.dtypes.DType:
    """Converts the config's dtype string to a tf dtype.

    Returns:
      A mapping from string representation of a dtype to the `tf.dtypes.DType`.

    Raises:
      ValueError if the config's dtype is not supported.

    """
    dtype_map = {
        'float32': tf.float32,
        'bfloat16': tf.bfloat16,
        'float16': tf.float16,
        'fp32': tf.float32,
        'bf16': tf.bfloat16,
    }
    try:
      return dtype_map[self.config.dtype]
    except:
      raise ValueError('Invalid DType provided. Supported types: {}'.format(
          dtype_map.keys()))

  @property
  def image_size(self) -> int:
    """The size of each image (can be inferred from the dataset)."""

    if self.config.image_size == 'infer':
      return self.info.features['image'].shape[0]
    else:
      return int(self.config.image_size)

  @property
  def num_channels(self) -> int:
    """The number of image channels (can be inferred from the dataset)."""
    if self.config.num_channels == 'infer':
      return self.info.features['image'].shape[-1]
    else:
      return int(self.config.num_channels)

  @property
  def num_examples(self) -> int:
    """The number of examples (can be inferred from the dataset)."""
    if self.config.num_examples == 'infer':
      return self.info.splits[self.config.split].num_examples
    else:
      return int(self.config.num_examples)

  @property
  def num_classes(self) -> int:
    """The number of classes (can be inferred from the dataset)."""
    if self.config.num_classes == 'infer':
      return self.info.features['label'].num_classes
    else:
      return int(self.config.num_classes)

  @property
  def info(self) -> tfds.core.DatasetInfo:
    """The TFDS dataset info, if available."""
    if self.builder_info is None:
      self.builder_info = tfds.builder(self.config.name).info
    return self.builder_info

  def build(self, strategy: tf.distribute.Strategy = None) -> tf.data.Dataset:
    """Construct a dataset end-to-end and return it using an optional strategy.

    Args:
      strategy: a strategy that, if passed, will distribute the dataset
        according to that strategy. If passed and `num_devices > 1`,
        `use_per_replica_batch_size` must be set to `True`.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if strategy:
      if strategy.num_replicas_in_sync != self.config.num_devices:
        logging.warn('Passed a strategy with %d devices, but expected'
                     '%d devices.',
                     strategy.num_replicas_in_sync,
                     self.config.num_devices)

      dataset = strategy.experimental_distribute_datasets_from_function(
          self._build)
    else:
      dataset = self._build()

    return dataset

  def _build(self, input_context: tf.distribute.InputContext = None
             ) -> tf.data.Dataset:
    """Construct a dataset end-to-end and return it.

    Args:
      input_context: An optional context provided by `tf.distribute` for
        cross-replica training.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    builders = {
        'tfds': self.load_tfds,
        'records': self.load_records,
        'synthetic': self.load_synthetic,
    }

    builder = builders.get(self.config.builder, None)

    if builder is None:
      raise ValueError('Unknown builder type {}'.format(self.config.builder))

    dataset = builder()
    dataset = self.pipeline(dataset, input_context)

    return dataset

  def load_tfds(self) -> tf.data.Dataset:
    """Return a dataset loading files from TFDS."""

    logging.info('Using TFDS to load data.')

    builder = tfds.builder(self.config.name,
                           data_dir=self.config.data_dir)

    if self.config.download:
      builder.download_and_prepare()

    decoders = {}

    if self.config.skip_decoding:
      decoders['image'] = tfds.decode.SkipDecoding()

    read_config = tfds.ReadConfig(
        interleave_cycle_length=64,
        interleave_block_length=1)

    dataset = builder.as_dataset(
        split=self.config.split,
        as_supervised=True,
        shuffle_files=True,
        decoders=decoders,
        read_config=read_config)

    return dataset

  def load_records(self) -> tf.data.Dataset:
    """Return a dataset loading files with TFRecords."""
    logging.info('Using TFRecords to load data.')

    if self.config.filenames is None:
      if self.config.data_dir is None:
        raise ValueError('Dataset must specify a path for the data files.')

      file_pattern = os.path.join(self.config.data_dir,
                                  '{}*'.format(self.config.split))
      dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    else:
      dataset = tf.data.Dataset.from_tensor_slices(self.config.filenames)
      if self.is_training:
        # Shuffle the input files.
        dataset.shuffle(buffer_size=self.config.file_shuffle_buffer_size)

    return dataset

  def load_synthetic(self) -> tf.data.Dataset:
    """Return a dataset generating dummy synthetic data."""
    logging.info('Generating a synthetic dataset.')

    def generate_data(_):
      image = tf.zeros([self.image_size, self.image_size, self.num_channels],
                       dtype=self.dtype)
      label = tf.zeros([1], dtype=tf.int32)
      return image, label

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(generate_data,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def pipeline(self,
               dataset: tf.data.Dataset,
               input_context: tf.distribute.InputContext = None
              ) -> tf.data.Dataset:
    """Build a pipeline fetching, shuffling, and preprocessing the dataset.

    Args:
      dataset: A `tf.data.Dataset` that loads raw files.
      input_context: An optional context provided by `tf.distribute` for
        cross-replica training. If set with more than one replica, this
        function assumes `use_per_replica_batch_size=True`.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if input_context and input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    if self.is_training and not self.config.cache:
      dataset = dataset.repeat()

    if self.config.builder == 'records':
      # Read the data from disk in parallel
      buffer_size = 8 * 1024 * 1024  # Use 8 MiB per file
      dataset = dataset.interleave(
          lambda name: tf.data.TFRecordDataset(name, buffer_size=buffer_size),
          cycle_length=16,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.config.cache:
      dataset = dataset.cache()

    if self.is_training:
      dataset = dataset.shuffle(self.config.shuffle_buffer_size)
      dataset = dataset.repeat()

    # Parse, pre-process, and batch the data in parallel
    if self.config.builder == 'records':
      preprocess = self.parse_record
    else:
      preprocess = self.preprocess
    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if input_context and self.config.num_devices > 1:
      if not self.config.use_per_replica_batch_size:
        raise ValueError(
            'The builder does not support a global batch size with more than '
            'one replica. Got {} replicas. Please set a '
            '`per_replica_batch_size` and enable '
            '`use_per_replica_batch_size=True`.'.format(
                self.config.num_devices))

      # The batch size of the dataset will be multiplied by the number of
      # replicas automatically when strategy.distribute_datasets_from_function
      # is called, so we use local batch size here.
      dataset = dataset.batch(self.local_batch_size,
                              drop_remainder=self.is_training)
    else:
      dataset = dataset.batch(self.global_batch_size,
                              drop_remainder=self.is_training)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def parse_record(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(record, keys_to_features)

    label = tf.reshape(parsed['image/class/label'], shape=[1])
    label = tf.cast(label, dtype=tf.int32)

    # Subtract one so that labels are in [0, 1000)
    label -= 1

    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image, label = self.preprocess(image_bytes, label)

    return image, label

  def preprocess(self, image: tf.Tensor, label: tf.Tensor
                ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply image preprocessing and augmentation to the image and label."""
    if self.is_training:
      image = preprocessing.preprocess_for_train(
          image,
          image_size=self.image_size,
          mean_subtract=self.config.mean_subtract,
          standardize=self.config.standardize,
          dtype=self.dtype,
          augmenter=self.augmenter)
    else:
      image = preprocessing.preprocess_for_eval(
          image,
          image_size=self.image_size,
          num_channels=self.num_channels,
          mean_subtract=self.config.mean_subtract,
          standardize=self.config.standardize,
          dtype=self.dtype)

    label = tf.cast(label, tf.int32)
    if self.config.one_hot:
      label = tf.one_hot(label, self.num_classes)
      label = tf.reshape(label, [self.num_classes])

    return image, label

  @classmethod
  def from_params(cls, *args, **kwargs):
    """Construct a dataset builder from a default config and any overrides."""
    config = DatasetConfig.from_args(*args, **kwargs)
    return cls(config)
