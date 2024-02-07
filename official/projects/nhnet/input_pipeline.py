# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Input pipelines."""

import tensorflow as tf, tf_keras


def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def process_singledoc_dataset(dataset, batch_size, params):
  """Parses and batches single-doc dataset."""
  name_to_features = {
      "input_ids_a": tf.io.FixedLenFeature([params.len_title], tf.int64),
      "input_ids_b": tf.io.FixedLenFeature([params.len_passage], tf.int64),
      "input_mask_b": tf.io.FixedLenFeature([params.len_passage], tf.int64),
      "segment_ids_b": tf.io.FixedLenFeature([params.len_passage], tf.int64),
  }
  decode_fn = lambda record: decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    return {
        "input_ids": record["input_ids_b"],
        "input_mask": record["input_mask_b"],
        "segment_ids": record["segment_ids_b"],
        "target_ids": record["input_ids_a"],
    }

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def decode_sparse_record(record, name_to_features):
  """Decodes a sparse record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = tf.sparse.to_dense(t)

  return example


def _filter_max_length(example, max_title_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.size(example["targets"]) <= max_title_length


def process_singledoc_transformer_dataset(dataset, batch_size, params):
  """Parses, batches and pads single-doc dataset."""
  name_to_features = {
      "inputs": tf.io.VarLenFeature(tf.int64),
      "targets": tf.io.VarLenFeature(tf.int64),
  }
  decode_fn = lambda record: decode_sparse_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    input_ids = record["inputs"][:params.len_passage]
    target_ids = record["targets"]
    input_mask = tf.ones_like(input_ids)
    segment_ids = tf.zeros_like(input_ids)
    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "target_ids": target_ids,
    }

  dataset = dataset.filter(lambda x: _filter_max_length(x, params.len_title))

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.padded_batch(
      batch_size, {
          "input_ids": [params.len_passage],
          "input_mask": [params.len_passage],
          "segment_ids": [params.len_passage],
          "target_ids": [params.len_title],
      },
      padding_values={
          "input_ids": params.pad_token_id,
          "input_mask": 0,
          "segment_ids": 0,
          "target_ids": params.pad_token_id,
      },
      drop_remainder=True)

  return dataset


def multidoc_parse_spec(params, training=True):
  """Gets the mutli-doc tf.Example parsing spec."""
  len_p = params.len_passage
  name_to_features = {}
  feature_list = ["input_ids", "input_mask", "segment_ids"]
  for idx in params.passage_list:
    for feature in feature_list:
      name_to_features["%s_%s" % (feature, idx)] = tf.io.FixedLenFeature(
          [len_p], tf.int64)
  if training:
    # Cluster title.
    name_to_features["input_ids_a"] = tf.io.FixedLenFeature([params.len_title],
                                                            tf.int64)
  return name_to_features, feature_list


def process_multidoc_dataset(dataset, batch_size, params):
  """Parses, organizes and batches multi-doc dataset."""
  name_to_features, feature_list = multidoc_parse_spec(params)
  decode_fn = lambda record: decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    features = {"target_ids": record["input_ids_a"]}
    for feature in feature_list:
      tensors = [record["%s_%s" % (feature, i)] for i in params.passage_list]
      features[feature] = tf.stack(tensors)
    return features

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def create_dataset(file_paths,
                   batch_size,
                   params,
                   is_training=True,
                   input_pipeline_context=None):
  """Creates input dataset from (tf)records files for pretraining."""
  dataset = tf.data.Dataset.list_files(file_paths, shuffle=is_training)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    if not is_training or params.input_sharding:
      dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                              input_pipeline_context.input_pipeline_id)

  if is_training:
    dataset = dataset.repeat()
    # We set shuffle buffer to exactly match total number of
    # training files to ensure that training data is well shuffled.
    dataset = dataset.shuffle(len(file_paths))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=8,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(100)

  if params.get("multi_channel_cross_attention", value=False):
    dataset = process_multidoc_dataset(dataset, batch_size, params)
  else:
    if not params.input_data_not_padded:
      dataset = process_singledoc_dataset(dataset, batch_size, params)
    else:
      dataset = process_singledoc_transformer_dataset(dataset, batch_size,
                                                      params)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def get_input_dataset(input_file_pattern,
                      batch_size,
                      params,
                      is_training,
                      strategy=None):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.TPUStrategy)
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          "Batch size must be divisible by number of replicas : {}".format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    input_files = []
    for input_pattern in input_file_pattern.split(","):
      input_files.extend(tf.io.gfile.glob(input_pattern))

    return create_dataset(
        input_files,
        batch_size,
        params,
        is_training=is_training,
        input_pipeline_context=ctx)

  if use_dataset_fn:
    return strategy.distribute_datasets_from_function(_dataset_fn)
  else:
    return strategy.experimental_distribute_dataset(_dataset_fn())
