# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""NCF model input pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# pylint: disable=g-bad-import-order
import tensorflow.compat.v2 as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants as rconst
from official.recommendation import movielens
from official.recommendation import data_pipeline

NUM_SHARDS = 16


def create_dataset_from_tf_record_files(input_file_pattern,
                                        pre_batch_size,
                                        batch_size,
                                        is_training=True):
  """Creates dataset from (tf)records files for training/evaluation."""

  files = tf.data.Dataset.list_files(input_file_pattern, shuffle=is_training)

  def make_dataset(files_dataset, shard_index):
    """Returns dataset for sharded tf record files."""
    if pre_batch_size != batch_size:
      raise ValueError("Pre-batch ({}) size is not equal to batch "
                       "size ({})".format(pre_batch_size, batch_size))
    files_dataset = files_dataset.shard(NUM_SHARDS, shard_index)
    dataset = files_dataset.interleave(tf.data.TFRecordDataset)
    decode_fn = functools.partial(
        data_pipeline.DatasetManager.deserialize,
        batch_size=pre_batch_size,
        is_training=is_training)
    dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  dataset = tf.data.Dataset.range(NUM_SHARDS)
  map_fn = functools.partial(make_dataset, files)
  dataset = dataset.interleave(
      map_fn,
      cycle_length=NUM_SHARDS,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def create_dataset_from_data_producer(producer, params):
  """Return dataset online-generating data."""

  def preprocess_train_input(features, labels):
    """Pre-process the training data.

    This is needed because
    - The label needs to be extended to be used in the loss fn
    - We need the same inputs for training and eval so adding fake inputs
      for DUPLICATE_MASK in training data.

    Args:
      features: Dictionary of features for training.
      labels: Training labels.

    Returns:
      Processed training features.
    """
    fake_dup_mask = tf.zeros_like(features[movielens.USER_COLUMN])
    features[rconst.DUPLICATE_MASK] = fake_dup_mask
    features[rconst.TRAIN_LABEL_KEY] = labels
    return features

  train_input_fn = producer.make_input_fn(is_training=True)
  train_input_dataset = train_input_fn(params).map(preprocess_train_input)

  def preprocess_eval_input(features):
    """Pre-process the eval data.

    This is needed because:
    - The label needs to be extended to be used in the loss fn
    - We need the same inputs for training and eval so adding fake inputs
      for VALID_PT_MASK in eval data.

    Args:
      features: Dictionary of features for evaluation.

    Returns:
      Processed evaluation features.
    """
    labels = tf.cast(tf.zeros_like(features[movielens.USER_COLUMN]), tf.bool)
    fake_valid_pt_mask = tf.cast(
        tf.zeros_like(features[movielens.USER_COLUMN]), tf.bool)
    features[rconst.VALID_POINT_MASK] = fake_valid_pt_mask
    features[rconst.TRAIN_LABEL_KEY] = labels
    return features

  eval_input_fn = producer.make_input_fn(is_training=False)
  eval_input_dataset = eval_input_fn(params).map(preprocess_eval_input)

  return train_input_dataset, eval_input_dataset


def create_ncf_input_data(params,
                          producer=None,
                          input_meta_data=None,
                          strategy=None):
  """Creates NCF training/evaluation dataset.

  Args:
    params: Dictionary containing parameters for train/evaluation data.
    producer: Instance of BaseDataConstructor that generates data online. Must
      not be None when params['train_dataset_path'] or
      params['eval_dataset_path'] is not specified.
    input_meta_data: A dictionary of input metadata to be used when reading data
      from tf record files. Must be specified when params["train_input_dataset"]
      is specified.
    strategy: Distribution strategy used for distributed training. If specified,
      used to assert that evaluation batch size is correctly a multiple of
      total number of devices used.

  Returns:
    (training dataset, evaluation dataset, train steps per epoch,
    eval steps per epoch)

  Raises:
    ValueError: If data is being generated online for when using TPU's.
  """
  # NCF evaluation metric calculation logic assumes that evaluation data
  # sample size are in multiples of (1 + number of negative samples in
  # evaluation) for each device. As so, evaluation batch size must be a
  # multiple of (number of replicas * (1 + number of negative samples)).
  num_devices = strategy.num_replicas_in_sync if strategy else 1
  if (params["eval_batch_size"] % (num_devices *
                                   (1 + rconst.NUM_EVAL_NEGATIVES))):
    raise ValueError("Evaluation batch size must be divisible by {} "
                     "times {}".format(num_devices,
                                       (1 + rconst.NUM_EVAL_NEGATIVES)))

  if params["train_dataset_path"]:
    assert params["eval_dataset_path"]

    train_dataset = create_dataset_from_tf_record_files(
        params["train_dataset_path"],
        input_meta_data["train_prebatch_size"],
        params["batch_size"],
        is_training=True)
    eval_dataset = create_dataset_from_tf_record_files(
        params["eval_dataset_path"],
        input_meta_data["eval_prebatch_size"],
        params["eval_batch_size"],
        is_training=False)

    num_train_steps = int(input_meta_data["num_train_steps"])
    num_eval_steps = int(input_meta_data["num_eval_steps"])
  else:
    if params["use_tpu"]:
      raise ValueError("TPU training does not support data producer yet. "
                       "Use pre-processed data.")

    assert producer
    # Start retrieving data from producer.
    train_dataset, eval_dataset = create_dataset_from_data_producer(
        producer, params)
    num_train_steps = producer.train_batches_per_epoch
    num_eval_steps = producer.eval_batches_per_epoch

  return train_dataset, eval_dataset, num_train_steps, num_eval_steps
