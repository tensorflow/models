# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""tf.data.Dataset builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import tensorflow.compat.v1 as tf

from object_detection.builders import decoder_builder
from object_detection.protos import input_reader_pb2


def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def _read_dataset_internal(file_read_func,
                           input_files,
                           num_readers,
                           config,
                           filename_shard_fn=None):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf_data.parallel_interleave, to read
      every individual file into a tf.data.Dataset.
    input_files: A list of file paths to read.
    num_readers: Number of readers to use.
    config: A input_reader_builder.InputReader object.
    filename_shard_fn: optional, A function used to shard filenames across
      replicas. This function takes as input a TF dataset of filenames and is
      expected to return its sharded version. It is useful when the dataset is
      being loaded on one of possibly many replicas and we want to evenly shard
      the files between the replicas.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.

  Raises:
    RuntimeError: If no files are found at the supplied path(s).
  """
  filenames = tf.gfile.Glob(input_files)
  tf.logging.info('Reading record datasets for input file: %s' % input_files)
  tf.logging.info('Number of filenames to read: %s' % len(filenames))
  if not filenames:
    raise RuntimeError('Did not find any input files matching the glob pattern '
                       '{}'.format(input_files))
  if num_readers > len(filenames):
    num_readers = len(filenames)
    tf.logging.warning('num_readers has been reduced to %d to match input file '
                       'shards.' % num_readers)
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if config.shuffle:
    filename_dataset = filename_dataset.shuffle(
        config.filenames_shuffle_buffer_size)
  elif num_readers > 1:
    tf.logging.warning('`shuffle` is false, but the input data stream is '
                       'still slightly shuffled since `num_readers` > 1.')
  if filename_shard_fn:
    filename_dataset = filename_shard_fn(filename_dataset)

  filename_dataset = filename_dataset.repeat(config.num_epochs or None)
  records_dataset = filename_dataset.apply(
      tf.data.experimental.parallel_interleave(
          file_read_func,
          cycle_length=num_readers,
          block_length=config.read_block_length,
          sloppy=config.shuffle))
  if config.shuffle:
    records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
  return records_dataset


def read_dataset(file_read_func, input_files, config, filename_shard_fn=None):
  """Reads multiple datasets with sampling.

  Args:
    file_read_func: Function to use in tf_data.parallel_interleave, to read
      every individual file into a tf.data.Dataset.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.
    filename_shard_fn: optional, A function used to shard filenames across
      replicas. This function takes as input a TF dataset of filenames and is
      expected to return its sharded version. It is useful when the dataset is
      being loaded on one of possibly many replicas and we want to evenly shard
      the files between the replicas.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.

  Raises:
    RuntimeError: If no files are found at the supplied path(s).
  """
  if config.sample_from_datasets_weights:
    tf.logging.info('Reading weighted datasets: %s' % input_files)
    if len(input_files) != len(config.sample_from_datasets_weights):
      raise ValueError('Expected the number of input files to be the same as '
                       'the number of dataset sample weights. But got '
                       '[input_files, sample_from_datasets_weights]: [' +
                       input_files + ', ' +
                       str(config.sample_from_datasets_weights) + ']')
    tf.logging.info('Sampling from datasets %s with weights %s' %
                    (input_files, config.sample_from_datasets_weights))
    records_datasets = []
    dataset_weights = []
    for i, input_file in enumerate(input_files):
      weight = config.sample_from_datasets_weights[i]
      num_readers = math.ceil(config.num_readers *
                              weight /
                              sum(config.sample_from_datasets_weights))
      tf.logging.info(
          'Num readers for dataset [%s]: %d', input_file, num_readers)
      if num_readers == 0:
        tf.logging.info('Skipping dataset due to zero weights: %s', input_file)
        continue
      tf.logging.info(
          'Num readers for dataset [%s]: %d', input_file, num_readers)
      records_dataset = _read_dataset_internal(file_read_func, [input_file],
                                               num_readers, config,
                                               filename_shard_fn)
      dataset_weights.append(weight)
      records_datasets.append(records_dataset)
    return tf.data.experimental.sample_from_datasets(records_datasets,
                                                     dataset_weights)
  else:
    tf.logging.info('Reading unweighted datasets: %s' % input_files)
    return _read_dataset_internal(file_read_func, input_files,
                                  config.num_readers, config, filename_shard_fn)


def shard_function_for_context(input_context):
  """Returns a function that shards filenames based on the input context."""

  if input_context is None:
    return None

  def shard_fn(dataset):
    return dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id)

  return shard_fn


def build(input_reader_config, batch_size=None, transform_input_data_fn=None,
          input_context=None, reduce_to_frame_fn=None):
  """Builds a tf.data.Dataset.

  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    batch_size: Batch size. If batch size is None, no batching is performed.
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.
    input_context: optional, A tf.distribute.InputContext object used to
      shard filenames and compute per-replica batch_size when this function
      is being called per-replica.
    reduce_to_frame_fn: Function that extracts frames from tf.SequenceExample
      type input data.

  Returns:
    A tf.data.Dataset based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  decoder = decoder_builder.build(input_reader_config)

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')
    def dataset_map_fn(dataset, fn_to_map, batch_size=None,
                       input_reader_config=None):
      """Handles whether or not to use the legacy map function.

      Args:
        dataset: A tf.Dataset.
        fn_to_map: The function to be mapped for that dataset.
        batch_size: Batch size. If batch size is None, no batching is performed.
        input_reader_config: A input_reader_pb2.InputReader object.

      Returns:
        A tf.data.Dataset mapped with fn_to_map.
      """
      if hasattr(dataset, 'map_with_legacy_function'):
        if batch_size:
          num_parallel_calls = batch_size * (
              input_reader_config.num_parallel_batches)
        else:
          num_parallel_calls = input_reader_config.num_parallel_map_calls
        dataset = dataset.map_with_legacy_function(
            fn_to_map, num_parallel_calls=num_parallel_calls)
      else:
        dataset = dataset.map(fn_to_map, tf.data.experimental.AUTOTUNE)
      return dataset
    shard_fn = shard_function_for_context(input_context)
    if input_context is not None:
      batch_size = input_context.get_per_replica_batch_size(batch_size)
    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], input_reader_config, filename_shard_fn=shard_fn)
    if input_reader_config.sample_1_of_n_examples > 1:
      dataset = dataset.shard(input_reader_config.sample_1_of_n_examples, 0)
    # TODO(rathodv): make batch size a required argument once the old binaries
    # are deleted.
    dataset = dataset_map_fn(dataset, decoder.decode, batch_size,
                             input_reader_config)
    if reduce_to_frame_fn:
      dataset = reduce_to_frame_fn(dataset, dataset_map_fn, batch_size,
                                   input_reader_config)
    if transform_input_data_fn is not None:
      dataset = dataset_map_fn(dataset, transform_input_data_fn,
                               batch_size, input_reader_config)
    if batch_size:
      dataset = dataset.batch(batch_size,
                              drop_remainder=input_reader_config.drop_remainder)
    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)
    return dataset

  raise ValueError('Unsupported input_reader_config.')
