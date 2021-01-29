# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""A common dataset reader."""
import random
from typing import Any, Callable, Optional

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.core import config_definitions as cfg


def _get_random_integer():
  return random.randint(0, (1 << 31) - 1)


class InputReader:
  """Input reader that returns a tf.data.Dataset instance."""

  # A static random number which is the same across different InputReader
  # instances.
  static_randnum = _get_random_integer()

  def __init__(self,
               params: cfg.DataConfig,
               dataset_fn=tf.data.TFRecordDataset,
               decoder_fn: Optional[Callable[..., Any]] = None,
               sample_fn: Optional[Callable[..., Any]] = None,
               parser_fn: Optional[Callable[..., Any]] = None,
               transform_and_batch_fn: Optional[Callable[
                   [tf.data.Dataset, Optional[tf.distribute.InputContext]],
                   tf.data.Dataset]] = None,
               postprocess_fn: Optional[Callable[..., Any]] = None):
    """Initializes an InputReader instance.

    Args:
      params: A config_definitions.DataConfig object.
      dataset_fn: A `tf.data.Dataset` that consumes the input files. For
        example, it can be `tf.data.TFRecordDataset`.
      decoder_fn: An optional `callable` that takes the serialized data string
        and decodes them into the raw tensor dictionary.
      sample_fn: An optional `callable` that takes a `tf.data.Dataset` object as
        input and outputs the transformed dataset. It performs sampling on the
        decoded raw tensors dict before the parser_fn.
      parser_fn: An optional `callable` that takes the decoded raw tensors dict
        and parse them into a dictionary of tensors that can be consumed by the
        model. It will be executed after decoder_fn.
      transform_and_batch_fn: An optional `callable` that takes a
        `tf.data.Dataset` object and an optional `tf.distribute.InputContext` as
        input, and returns a `tf.data.Dataset` object. It will be executed after
        `parser_fn` to transform and batch the dataset; if None, after
        `parser_fn` is executed, the dataset will be batched into per-replica
        batch size.
      postprocess_fn: A optional `callable` that processes batched tensors. It
        will be executed after batching.
    """
    if params.input_path and params.tfds_name:
      raise ValueError('At most one of `input_path` and `tfds_name` can be '
                       'specified, but got %s and %s.' %
                       (params.input_path, params.tfds_name))
    self._tfds_builder = None
    self._matched_files = []
    if params.input_path:
      # Read dataset from files.
      usage = ('`input_path` should be either (1) a str indicating a file '
               'path/pattern, or (2) a str indicating multiple file '
               'paths/patterns separated by comma (e.g "a, b, c" or no spaces '
               '"a,b,c", or (3) a list of str, each of which is a file '
               'path/pattern or multiple file paths/patterns separated by '
               'comma, but got: %s')
      if isinstance(params.input_path, str):
        input_path_list = [params.input_path]
      elif isinstance(params.input_path, (list, tuple)):
        if any(not isinstance(x, str) for x in params.input_path):
          raise ValueError(usage % params.input_path)
        input_path_list = params.input_path
      else:
        raise ValueError(usage % params.input_path)

      for input_path in input_path_list:
        input_patterns = input_path.strip().split(',')
        for input_pattern in input_patterns:
          input_pattern = input_pattern.strip()
          if not input_pattern:
            continue
          if '*' in input_pattern or '?' in input_pattern:
            tmp_matched_files = tf.io.gfile.glob(input_pattern)
            if not tmp_matched_files:
              raise ValueError('%s does not match any files.' % input_pattern)
            self._matched_files.extend(tmp_matched_files)
          else:
            self._matched_files.append(input_pattern)

      if not self._matched_files:
        raise ValueError('%s does not match any files.' % params.input_path)
    else:
      # Read dataset from TFDS.
      if not params.tfds_split:
        raise ValueError(
            '`tfds_name` is %s, but `tfds_split` is not specified.' %
            params.tfds_name)
      self._tfds_builder = tfds.builder(
          params.tfds_name, data_dir=params.tfds_data_dir)

    self._global_batch_size = params.global_batch_size
    self._is_training = params.is_training
    self._drop_remainder = params.drop_remainder
    self._shuffle_buffer_size = params.shuffle_buffer_size
    self._cache = params.cache
    self._cycle_length = params.cycle_length
    self._block_length = params.block_length
    self._deterministic = params.deterministic
    self._sharding = params.sharding
    self._tfds_split = params.tfds_split
    self._tfds_download = params.tfds_download
    self._tfds_as_supervised = params.tfds_as_supervised
    self._tfds_skip_decoding_feature = params.tfds_skip_decoding_feature

    self._dataset_fn = dataset_fn
    self._decoder_fn = decoder_fn
    self._sample_fn = sample_fn
    self._parser_fn = parser_fn
    self._transform_and_batch_fn = transform_and_batch_fn
    self._postprocess_fn = postprocess_fn
    self._seed = _get_random_integer()

    self._enable_tf_data_service = (
        params.enable_tf_data_service and params.tf_data_service_address)
    self._tf_data_service_address = params.tf_data_service_address
    if self._enable_tf_data_service:
      # Add a random seed as the tf.data service job name suffix, so tf.data
      # service doesn't reuse the previous state if TPU worker gets preempted.
      self._tf_data_service_job_name = (
          params.tf_data_service_job_name + str(self.static_randnum))
      self._enable_round_robin_tf_data_service = params.get(
          'enable_round_robin_tf_data_service', False)

  def _shard_files_then_read(
      self, input_context: Optional[tf.distribute.InputContext] = None):
    """Shards the data files and then sent a split to every worker to read."""
    dataset = tf.data.Dataset.from_tensor_slices(self._matched_files)

    # Shuffle and repeat at file level.
    if self._is_training:
      dataset = dataset.shuffle(
          len(self._matched_files),
          seed=self._seed,
          reshuffle_each_iteration=True)

    # Do not enable sharding if tf.data service is enabled, as sharding will be
    # handled inside tf.data service.
    if self._sharding and input_context and (
        input_context.num_input_pipelines > 1 and
        not self._enable_tf_data_service):
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if self._is_training:
      dataset = dataset.repeat()

    dataset = dataset.interleave(
        map_func=self._dataset_fn,
        cycle_length=self._cycle_length,
        block_length=self._block_length,
        num_parallel_calls=(self._cycle_length if self._cycle_length else
                            tf.data.experimental.AUTOTUNE),
        deterministic=self._deterministic)
    return dataset

  def _read_files_then_shard(
      self, input_context: Optional[tf.distribute.InputContext] = None):
    """Sends all data files to every worker and then shard by data."""
    dataset = self._dataset_fn(self._matched_files)

    # When `input_file` is a path to a single file or the number of files is
    # less than the number of input pipelines, disable auto sharding
    # so that same input file is sent to all workers.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    dataset = dataset.with_options(options)
    # Do not enable sharding if tf.data service is enabled, as sharding will be
    # handled inside tf.data service.
    if self._sharding and input_context and (
        input_context.num_input_pipelines > 1 and
        not self._enable_tf_data_service):
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if self._is_training:
      dataset = dataset.repeat()
    return dataset

  def _read_tfds(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Reads a dataset from tfds."""
    if self._tfds_download:
      self._tfds_builder.download_and_prepare()

    read_config = tfds.ReadConfig(
        interleave_cycle_length=self._cycle_length,
        interleave_block_length=self._block_length,
        input_context=input_context)
    decoders = {}
    if self._tfds_skip_decoding_feature:
      for skip_feature in self._tfds_skip_decoding_feature.split(','):
        decoders[skip_feature.strip()] = tfds.decode.SkipDecoding()
    dataset = self._tfds_builder.as_dataset(
        split=self._tfds_split,
        shuffle_files=self._is_training,
        as_supervised=self._tfds_as_supervised,
        decoders=decoders,
        read_config=read_config)

    if self._is_training:
      dataset = dataset.repeat()
    return dataset

  @property
  def tfds_info(self) -> tfds.core.DatasetInfo:
    """Returns TFDS dataset info, if available."""
    if self._tfds_builder:
      return self._tfds_builder.info
    else:
      raise ValueError('tfds_info is not available, because the dataset '
                       'is not loaded from tfds.')

  def read(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Generates a tf.data.Dataset object."""
    if self._tfds_builder:
      dataset = self._read_tfds(input_context)
    elif len(self._matched_files) > 1:
      if input_context and (len(self._matched_files) <
                            input_context.num_input_pipelines):
        logging.warn(
            'The number of files %d is less than the number of input pipelines '
            '%d. We will send all input files to every worker. '
            'Please consider sharding your data into more files.',
            len(self._matched_files), input_context.num_input_pipelines)
        dataset = self._read_files_then_shard(input_context)
      else:
        dataset = self._shard_files_then_read(input_context)
    elif len(self._matched_files) == 1:
      dataset = self._read_files_then_shard(input_context)
    else:
      raise ValueError('It is unexpected that `tfds_builder` is None and '
                       'there is also no `matched_files`.')

    if self._cache:
      dataset = dataset.cache()

    if self._is_training:
      dataset = dataset.shuffle(self._shuffle_buffer_size)

    def maybe_map_fn(dataset, fn):
      return dataset if fn is None else dataset.map(
          fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = maybe_map_fn(dataset, self._decoder_fn)
    if self._sample_fn is not None:
      dataset = dataset.apply(self._sample_fn)
    dataset = maybe_map_fn(dataset, self._parser_fn)

    if self._transform_and_batch_fn is not None:
      dataset = self._transform_and_batch_fn(dataset, input_context)
    else:
      per_replica_batch_size = input_context.get_per_replica_batch_size(
          self._global_batch_size) if input_context else self._global_batch_size
      dataset = dataset.batch(
          per_replica_batch_size, drop_remainder=self._drop_remainder)

    dataset = maybe_map_fn(dataset, self._postprocess_fn)

    if self._enable_tf_data_service and input_context:
      if self._enable_round_robin_tf_data_service:
        replicas_per_input_pipeline = input_context.num_replicas_in_sync // (
            input_context.num_input_pipelines)
        base_consumer_index = input_context.input_pipeline_id * (
            replicas_per_input_pipeline)
        num_consumers = input_context.num_input_pipelines * (
            replicas_per_input_pipeline)
        range_dataset = tf.data.Dataset.range(replicas_per_input_pipeline)
        dataset = range_dataset.map(lambda i: dataset.apply(  # pylint: disable=g-long-lambda
            tf.data.experimental.service.distribute(
                processing_mode='parallel_epochs',
                service=self._tf_data_service_address,
                job_name=self._tf_data_service_job_name,
                consumer_index=base_consumer_index + i,
                num_consumers=num_consumers)))
        # Use parallel interleave to read multiple batches from a tf.data
        # service worker in parallel.
        dataset = dataset.interleave(
            lambda x: x,
            cycle_length=replicas_per_input_pipeline,
            num_parallel_calls=replicas_per_input_pipeline,
            deterministic=True)
      else:
        dataset = dataset.apply(
            tf.data.experimental.service.distribute(
                processing_mode='parallel_epochs',
                service=self._tf_data_service_address,
                job_name=self._tf_data_service_job_name))

    if self._deterministic is not None:
      options = tf.data.Options()
      options.experimental_deterministic = self._deterministic
      dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
