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
"""BERT model input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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


def file_based_input_fn_builder(input_file, name_to_features):
  """Creates an `input_fn` closure to be passed for BERT custom training."""

  def input_fn():
    """Returns dataset for training/evaluation."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: decode_record(record, name_to_features))

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_file, str) or len(input_file) == 1:
      options = tf.data.Options()
      options.experimental_distribute.auto_shard = False
      d = d.with_options(options)
    return d

  return input_fn


def create_pretrain_dataset(file_paths,
                            seq_length,
                            max_predictions_per_seq,
                            batch_size,
                            is_training=True,
                            input_pipeline_context=None):
  """Creates input dataset from (tf)records files for pretraining."""
  name_to_features = {
      'input_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'masked_lm_positions':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_ids':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_weights':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
      'next_sentence_labels':
          tf.io.FixedLenFeature([1], tf.int64),
  }

  dataset = tf.data.Dataset.list_files(file_paths, shuffle=is_training)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  dataset = dataset.repeat()
  # We set shuffle buffer to exactly match total number of
  # training files to ensure that training data is well shuffled.
  dataset = dataset.shuffle(len(file_paths))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset, cycle_length=8,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  decode_fn = lambda record: decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
        'masked_lm_positions': record['masked_lm_positions'],
        'masked_lm_ids': record['masked_lm_ids'],
        'masked_lm_weights': record['masked_lm_weights'],
        'next_sentence_labels': record['next_sentence_labels'],
    }

    y = record['masked_lm_weights']

    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(100)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)
  return dataset


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              drop_remainder=True):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64),
      'is_real_example': tf.io.FixedLenFeature([], tf.int64),
  }
  input_fn = file_based_input_fn_builder(file_path, name_to_features)
  dataset = input_fn()

  def _select_data_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    return (x, y)

  dataset = dataset.map(_select_data_from_record)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(1024)
  return dataset


def create_squad_dataset(file_path, seq_length, batch_size, is_training=True):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'unique_ids': tf.io.FixedLenFeature([], tf.int64),
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
  }
  if is_training:
    name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)

  input_fn = file_based_input_fn_builder(file_path, name_to_features)
  dataset = input_fn()

  def _select_data_from_record(record):
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions'):
        y[name] = tensor
      else:
        x[name] = tensor
    return (x, y)

  dataset = dataset.map(_select_data_from_record)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)
  return dataset
