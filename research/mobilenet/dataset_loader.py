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

import logging
from typing import Text, Tuple, Mapping
from functools import partial

from dataclasses import dataclass

import tensorflow_datasets as tfds
import tensorflow as tf

from official.vision.image_classification.dataset_factory import DatasetConfig
from research.mobilenet import dataset_preprocessing


@dataclass
class ImageNetConfig(DatasetConfig):
  """The base ImageNet dataset config."""
  name: str = 'imagenet2012'
  image_size: int = 224
  batch_size: int = 32


def _get_dtype_map() -> Mapping[str, tf.dtypes.DType]:
  """Returns the mapping from dtype string representations to TF dtypes."""
  return {
    'float32': tf.float32,
    'bfloat16': tf.bfloat16,
    'float16': tf.float16,
    'fp32': tf.float32,
    'bf16': tf.bfloat16,
  }


def _preprocess(image: tf.Tensor,
                label: tf.Tensor,
                config: DatasetConfig,
                is_training: bool = True
                ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply image preprocessing and augmentation to the image and label."""
  if is_training:
    image = dataset_preprocessing.preprocess_for_train(
      image,
      image_size=config.image_size,
      mean_subtract=config.mean_subtract,
      standardize=config.standardize,
      dtype=_get_dtype_map()[config.dtype])
  else:
    image = dataset_preprocessing.preprocess_for_eval(
      image,
      image_size=config.image_size,
      num_channels=config.num_channels,
      mean_subtract=config.config.mean_subtract,
      standardize=config.config.standardize,
      dtype=_get_dtype_map()[config.dtype])

  label = tf.cast(label, tf.int32)
  if config.one_hot:
    label = tf.one_hot(label, config.num_classes)
    label = tf.reshape(label, [config.num_classes])

  return image, label


def load_tfds(
    dataset_name: Text,
    data_dir: Text,
    download: bool,
    skip_decoding: bool,
    split: Text,
) -> tf.data.Dataset:
  """Return a dataset loading files from TFDS."""

  logging.info('Using TFDS to load data.')

  builder = tfds.builder(name=dataset_name,
                         data_dir=data_dir)

  if download:
    builder.download_and_prepare()

  decoders = {}

  if skip_decoding:
    decoders['image'] = tfds.decode.SkipDecoding()

  read_config = tfds.ReadConfig(
    interleave_parallel_reads=8,
    interleave_block_length=1)

  dataset = builder.as_dataset(
    split=split,
    as_supervised=True,
    shuffle_files=True,
    decoders=decoders,
    read_config=read_config)

  return dataset


def pipeline(dataset: tf.data.Dataset,
             config: DatasetConfig,
             is_training: bool = True,
             strategy: tf.distribute.Strategy = None
             ) -> tf.data.Dataset:
  """Build a pipeline fetching, shuffling, and preprocessing the dataset.

  Args:
    dataset: A `tf.data.Dataset` that loads raw files.

  Returns:
    A TensorFlow dataset outputting batched images and labels.
  """

  if is_training and not config.cache:
    dataset = dataset.repeat()

  if config.cache:
    dataset = dataset.cache()

  if is_training:
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.repeat()

  # Parse, pre-process, and batch the data in parallel

  dataset = dataset.map(
    partial(_preprocess, config=config, is_training=is_training),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if strategy:
    num_devices = strategy.num_replicas_in_sync
    if not config.use_per_replica_batch_size:
      raise ValueError(
        'The builder does not support a global batch size with more than '
        'one replica. Got {} replicas. Please set a '
        '`per_replica_batch_size` and enable '
        '`use_per_replica_batch_size=True`.'.format(num_devices))
    global_batch_size = config.batch_size * num_devices
  else:
    global_batch_size = config.batch_size
  dataset = dataset.batch(global_batch_size, drop_remainder=is_training)

  if is_training:
    options = tf.data.Options()
    options.experimental_deterministic = config.deterministic_train
    options.experimental_slack = config.use_slack
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)

  # Prefetch overlaps in-feed with training
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
