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

import os
import logging
from typing import Text, Mapping, Tuple, List, Optional, Union
import functools

import tensorflow_datasets as tfds
import tensorflow as tf

from official.vision.image_classification import preprocessing
from official.vision.image_classification import dataset_factory

from research.mobilenet import preprocessing_slim


def _get_dtype_map() -> Mapping[Text, tf.dtypes.DType]:
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
                config: dataset_factory.DatasetConfig,
                is_training: bool = True,
                slim_preprocess: bool = False
                ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply image preprocessing and augmentation to the image and label.

  Args:
    image: a Tensor representing the image
    label: a Tensor containing the label
    config: a instant of DatasetConfig
    is_training: indicate whether the process involved is for training
    slim_preprocess: whether use the preprocessing function from slim

  Returns:
    A tuple with processed (image, label)
  """
  if is_training:
    if slim_preprocess:
      image = preprocessing_slim.preprocess_for_train(
        image=image,
        image_size=config.image_size)
    else:
      image = preprocessing.preprocess_for_train(
        image_bytes=image,
        image_size=config.image_size,
        mean_subtract=config.mean_subtract,
        standardize=config.standardize,
        dtype=_get_dtype_map()[config.dtype],
        augmenter=config.augmenter.build()
        if config.augmenter is not None else None)
  else:
    if slim_preprocess:
      image = preprocessing_slim.preprocess_for_eval(
        image=image,
        image_size=config.image_size)
    else:
      image = preprocessing.preprocess_for_eval(
        image_bytes=image,
        image_size=config.image_size,
        num_channels=config.num_channels,
        mean_subtract=config.mean_subtract,
        standardize=config.standardize,
        dtype=_get_dtype_map()[config.dtype])

  label = tf.cast(label, tf.int32)
  if config.one_hot:
    label = tf.one_hot(label, config.num_classes)
    label = tf.reshape(label, [config.num_classes])

  return image, label


def _get_tf_data_config() -> tf.data.Options:
  """Construct an `Options` object to control which graph
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations

  Args:

  Returns:
    An options instance for tf.data.Dataset.
  """
  options = tf.data.Options()
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.map_fusion = True
  options.experimental_optimization.map_vectorization.enabled = True
  options.experimental_optimization.map_parallelization = True
  return options


def parse_record(record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

  image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

  return image_bytes, label


def load_tfrecords(data_dir: Text,
                   split: Text,
                   file_shuffle_buffer_size: int
                   ) -> tf.data.Dataset:
  """

  Args:
    data_dir: path holding the dataset
    split: the split of data, `train` or `validation`
    file_shuffle_buffer_size: the number of elements from this dataset from
    which the new dataset will sample

  Returns:
    Return a dataset loading files with TFRecords.
  """
  logging.info('Using TFRecords to load data.')

  file_pattern = os.path.join(data_dir, '{}*'.format(split))
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  dataset.shuffle(buffer_size=file_shuffle_buffer_size)
  dataset = dataset.interleave(
    tf.data.TFRecordDataset,
    cycle_length=10,
    block_length=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(map_func=parse_record,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def load_tfds(dataset_name: Text,
              split: Union[Text, List[Text]],
              download: bool = True,
              data_dir: Optional[Text] = None
              ) -> tf.data.Dataset:
  """Load dataset using TFDS

  Args:
    dataset_name: name of the dataset.
    split: split to be loaded. The value could be `train` and `test`.
    download: whether download the dataset.
    data_dir: directory to read/write data.

  Returns:
    A tf.data.Dataset instance representing the target dataset.
  """

  logging.info('Using TFDS to load data.')

  # A temp fix as here: https://github.com/tensorflow/datasets/issues/1441
  # import resource
  # low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  # resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

  if download:
    data_dir = None

  dataset = tfds.load(
    name=dataset_name,
    data_dir=data_dir,
    split=split,
    download=download,
    as_supervised=True,
    shuffle_files=True)

  return dataset


def pipeline(dataset: tf.data.Dataset,
             config: dataset_factory.DatasetConfig,
             slim_preprocess: bool = False
             ) -> tf.data.Dataset:
  """Build a pipeline fetching, shuffling, and preprocessing the dataset.

  Args:
    dataset: A `tf.data.Dataset` that loads raw files.
    config: A subclass instance of DatasetConfig
    slim_preprocess: whether use the preprocessing function from slim

  Returns:
    A TensorFlow dataset outputting batched images and labels.
  """

  is_training = True if config.split == 'train' else False

  if is_training and not config.cache:
    dataset = dataset.repeat()

  if config.cache:
    dataset = dataset.cache()

  if is_training:
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.repeat()

  # Parse, pre-process, and batch the data in parallel
  preprocess_func = functools.partial(_preprocess,
                                      config=config,
                                      is_training=is_training,
                                      slim_preprocess=slim_preprocess)
  dataset = dataset.map(preprocess_func,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(config.batch_size, drop_remainder=is_training)

  if is_training:
    dataset = dataset.with_options(_get_tf_data_config())

  # Prefetch overlaps in-feed with training
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
