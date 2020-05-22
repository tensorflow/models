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
from typing import Text, Mapping, Type, Tuple, List, Optional, Union
import functools

import tensorflow_datasets as tfds
import tensorflow as tf

from research.mobilenet.configs import dataset as dataset_config
from official.vision.image_classification import preprocessing


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
                config: Type[dataset_config.DatasetConfig],
                is_training: bool = True
                ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply image preprocessing and augmentation to the image and label.

  Args:
    image: a Tensor representing the image
    label: a Tensor containing the label
    config: a instant of DatasetConfig
    is_training: indicate whether the process involved is for training

  Returns:
    A tuple with processed (image, label)
  """
  if is_training:
    image = preprocessing.preprocess_for_train(
      image,
      image_size=config.image_size,
      mean_subtract=config.mean_subtract,
      standardize=config.standardize,
      dtype=_get_dtype_map()[config.dtype])
  else:
    image = preprocessing.preprocess_for_eval(
      image,
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


def _get_tf_data_config(config: Type[dataset_config.DatasetConfig]
                        ) -> tf.data.Options:
  """Construct an `Options` object to control which graph
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations

  Args:
    config: A subclass instance of DatasetConfig.

  Returns:
    An options instance for tf.data.Dataset.
  """
  options = tf.data.Options()
  options.experimental_deterministic = config.deterministic_train
  options.experimental_slack = config.use_slack
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.map_fusion = True
  options.experimental_optimization.map_vectorization.enabled = True
  options.experimental_optimization.map_parallelization = True
  return options


def load_tfds(dataset_name: Text,
              split: Union[Text, List[Text]],
              download: bool = True,
              data_dir: Optional[Text] = None,
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

  dataset = tfds.load(
    name=dataset_name,
    data_dir=data_dir,
    split=split,
    download=download,
    as_supervised=True,
    shuffle_files=True)

  return dataset


def pipeline(dataset: tf.data.Dataset,
             config: Type[dataset_config.DatasetConfig],
             ) -> tf.data.Dataset:
  """Build a pipeline fetching, shuffling, and preprocessing the dataset.

  Args:
    dataset: A `tf.data.Dataset` that loads raw files.
    config: A subclass instance of DatasetConfig

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
  dataset = dataset.map(
    functools.partial(_preprocess, config=config, is_training=is_training),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(config.batch_size, drop_remainder=is_training)

  if is_training:
    dataset = dataset.with_options(_get_tf_data_config(config))

  # Prefetch overlaps in-feed with training
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
