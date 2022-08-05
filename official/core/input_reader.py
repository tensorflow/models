# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Text, Union

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.core import config_definitions as cfg


def _get_random_integer():
  return random.randint(0, (1 << 31) - 1)


def _maybe_map_fn(dataset: tf.data.Dataset,
                  fn: Optional[Callable[..., Any]] = None) -> tf.data.Dataset:
  """Calls dataset.map if a valid function is passed in."""
  return dataset if fn is None else dataset.map(
      fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def match_files(input_path: Union[Sequence[str], str]) -> List[str]:
  """Matches files from an input_path."""
  matched_files = []
  # Read dataset from files.
  usage = ('`input_path` should be either (1) a str indicating a file '
           'path/pattern, or (2) a str indicating multiple file '
           'paths/patterns separated by comma (e.g "a, b, c" or no spaces '
           '"a,b,c", or (3) a list of str, each of which is a file '
           'path/pattern or multiple file paths/patterns separated by '
           'comma, but got: %s')
  if isinstance(input_path, str):
    input_path_list = [input_path]
  elif isinstance(input_path, (list, tuple)):
    if any(not isinstance(x, str) for x in input_path):
      raise ValueError(usage % input_path)
    input_path_list = input_path
  else:
    raise ValueError(usage % input_path)

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
        matched_files.extend(tmp_matched_files)
      else:
        matched_files.append(input_pattern)

  if not matched_files:
    raise ValueError('%s does not match any files.' % input_path)

  return matched_files


def _read_files_then_shard(matched_files: List[str],
                           dataset_fn,
                           input_context: Optional[
                               tf.distribute.InputContext] = None,
                           sharding: bool = False,
                           repeat: bool = False) -> tf.data.Dataset:
  """Sends all data files to every worker and then shard by data."""
  dataset = dataset_fn(matched_files)

  # When `input_file` is a path to a single file or the number of files is
  # less than the number of input pipelines, disable auto sharding
  # so that same input file is sent to all workers.
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.OFF)
  dataset = dataset.with_options(options)
  # Do not enable sharding if tf.data service is enabled, as sharding will be
  # handled inside tf.data service.
  if sharding and input_context and (input_context.num_input_pipelines > 1):
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  if repeat:
    dataset = dataset.repeat()
  return dataset


def _shard_files_then_read(matched_files: List[str],
                           dataset_fn,
                           input_context: Optional[
                               tf.distribute.InputContext] = None,
                           seed: Optional[Union[int, tf.Tensor]] = None,
                           is_training: bool = False,
                           sharding: bool = False,
                           cache: bool = False,
                           cycle_length: Optional[int] = None,
                           block_length: Optional[int] = None,
                           deterministic: bool = False) -> tf.data.Dataset:
  """Shards the data files and then sent a split to every worker to read."""
  dataset = tf.data.Dataset.from_tensor_slices(matched_files)

  # Shuffle and repeat at file level.
  # If cache is enabled, `reshuffle_each_iteration` is set to False,
  # because we will read the same cached data in every iteration anyway.
  if is_training:
    # We need a seed to shuffle the files so that when each TPU workers gets
    # its own shard the files do not overlap.
    if sharding and seed is None:
      seed = _get_random_integer()
    dataset = dataset.shuffle(
        len(matched_files),
        seed=seed,
        reshuffle_each_iteration=True if not cache else False)

  # Do not enable sharding if tf.data service is enabled, as sharding will be
  # handled inside tf.data service.
  if sharding and input_context and (input_context.num_input_pipelines > 1):
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  # If cache is enabled, we will call `repeat()` later after `cache()`.
  if is_training and not cache:
    dataset = dataset.repeat()

  dataset = dataset.interleave(
      map_func=dataset_fn,
      cycle_length=cycle_length,
      block_length=block_length,
      num_parallel_calls=(cycle_length
                          if cycle_length else tf.data.experimental.AUTOTUNE),
      deterministic=deterministic)
  return dataset


def _read_tfds(tfds_builder: tfds.core.DatasetBuilder,
               tfds_split: Text,
               tfds_skip_decoding_feature: Text,
               tfds_as_supervised: bool,
               input_context: Optional[tf.distribute.InputContext] = None,
               seed: Optional[Union[int, tf.Tensor]] = None,
               is_training: bool = False,
               cache: bool = False,
               cycle_length: Optional[int] = None,
               block_length: Optional[int] = None) -> tf.data.Dataset:
  """Reads a dataset from tfds."""
  # No op if exist.
  tfds_builder.download_and_prepare()
  decoders = {}
  if tfds_skip_decoding_feature:
    for skip_feature in tfds_skip_decoding_feature.split(','):
      decoders[skip_feature.strip()] = tfds.decode.SkipDecoding()
  if tfds_builder.info.splits:
    num_shards = len(tfds_builder.info.splits[tfds_split].file_instructions)
  else:
    # The tfds mock path often does not provide splits.
    num_shards = 1
  if input_context and num_shards < input_context.num_input_pipelines:
    # The number of files in the dataset split is smaller than the number of
    # input pipelines. We read the entire dataset first and then shard in the
    # host memory.
    read_config = tfds.ReadConfig(
        interleave_cycle_length=cycle_length,
        interleave_block_length=block_length,
        input_context=None,
        shuffle_seed=seed)
    dataset = tfds_builder.as_dataset(
        split=tfds_split,
        shuffle_files=is_training,
        as_supervised=tfds_as_supervised,
        decoders=decoders,
        read_config=read_config)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
  else:
    read_config = tfds.ReadConfig(
        interleave_cycle_length=cycle_length,
        interleave_block_length=block_length,
        input_context=input_context,
        shuffle_seed=seed)
    dataset = tfds_builder.as_dataset(
        split=tfds_split,
        shuffle_files=is_training,
        as_supervised=tfds_as_supervised,
        decoders=decoders,
        read_config=read_config)

  if is_training and not cache:
    dataset = dataset.repeat()
  return dataset


class InputReader:
  """Input reader that returns a tf.data.Dataset instance."""

  # A static random number which is the same across different InputReader
  # instances.
  static_randnum = _get_random_integer()

  def __init__(self,
               params: cfg.DataConfig,
               dataset_fn=tf.data.TFRecordDataset,
               decoder_fn: Optional[Callable[..., Any]] = None,
               combine_fn: Optional[Callable[..., Any]] = None,
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
      combine_fn: An optional `callable` that takes a dictionarty of
        `tf.data.Dataset` objects as input and outputs a combined dataset. It
        will be executed after the decoder_fn and before the sample_fn.
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

    if isinstance(params.input_path,
                  cfg.base_config.Config) and combine_fn is None:
      raise ValueError(
          'A `combine_fn` is required if the `input_path` is a dictionary.')

    self._tfds_builder = None
    self._matched_files = None
    if not params.input_path:
      # Read dataset from TFDS.
      if not params.tfds_split:
        raise ValueError(
            '`tfds_name` is %s, but `tfds_split` is not specified.' %
            params.tfds_name)
      self._tfds_builder = tfds.builder(
          params.tfds_name, data_dir=params.tfds_data_dir)
    else:
      self._matched_files = self.get_files(params.input_path)

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
    self._tfds_as_supervised = params.tfds_as_supervised
    self._tfds_skip_decoding_feature = params.tfds_skip_decoding_feature

    self._dataset_fn = dataset_fn
    self._decoder_fn = decoder_fn
    self._combine_fn = combine_fn
    self._sample_fn = sample_fn
    self._parser_fn = parser_fn
    self._transform_and_batch_fn = transform_and_batch_fn
    self._postprocess_fn = postprocess_fn
    self._seed = params.seed
    self._prefetch_buffer_size = (
        params.prefetch_buffer_size or tf.data.experimental.AUTOTUNE)

    # When tf.data service is enabled, each data service worker should get
    # different random seeds. Thus, we set `seed` to None.
    # Sharding should also be disabled because tf data service handles how
    # each worker shard data with `processing_mode` in distribute method.
    if params.enable_tf_data_service:
      self._seed = None
      self._sharding = False

    self._enable_tf_data_service = (
        params.enable_tf_data_service and params.tf_data_service_address)
    self._tf_data_service_address = params.tf_data_service_address
    self._enable_shared_tf_data_service_between_parallel_trainers = (
        params.enable_shared_tf_data_service_between_parallel_trainers)
    self._apply_tf_data_service_before_batching = (
        params.apply_tf_data_service_before_batching)
    self._trainer_id = params.trainer_id
    if self._enable_tf_data_service:
      # Add a random seed as the tf.data service job name suffix, so tf.data
      # service doesn't reuse the previous state if TPU worker gets preempted.
      # It's necessary to add global batch size into the tf data service job
      # name because when tuning batch size with vizier and tf data service is
      # also enable, the tf data servce job name should be different for
      # different vizier trials since once batch size is changed, from the
      # tf.data perspective, the dataset is a different instance, and a
      # different job name should be used for tf data service. Otherwise, the
      # model would read tensors from the incorrect tf data service job, which
      # would causes dimension mismatch on the batch size dimension.
      self._tf_data_service_job_name = (
          f'{params.tf_data_service_job_name}_bs{params.global_batch_size}_'
          f'{self.static_randnum}')
      self._enable_round_robin_tf_data_service = params.get(
          'enable_round_robin_tf_data_service', False)
      if self._enable_shared_tf_data_service_between_parallel_trainers:
        # When shared tf.data service is enabled, only a single tf.data service
        # instance should be created and shared between parallel trainers. If
        # the global batch size is different across trainers,
        # params.apply_tf_data_service_before_batching should be set to true
        # because tf.data service with different batch sizes will be considered
        # separate tf.data service instances.
        self._tf_data_service_job_name = (
            f'{params.tf_data_service_job_name}_{self.static_randnum}')

  @property
  def tfds_info(self) -> tfds.core.DatasetInfo:
    """Returns TFDS dataset info, if available."""
    if self._tfds_builder:
      return self._tfds_builder.info
    else:
      raise ValueError('tfds_info is not available, because the dataset '
                       'is not loaded from tfds.')

  def get_files(self, input_path):
    """Gets matched files. Can be overridden by subclasses."""
    if not input_path:
      return None
    # we want to combine / mix datasets
    if isinstance(input_path, cfg.base_config.Config):
      matched_files = {}
      for k, v in input_path.as_dict().items():
        matched_files[k] = match_files(v)
    # single dataset
    else:
      matched_files = match_files(input_path)
    return matched_files

  def _read_data_source(
      self,
      matched_files: Union[Dict[str, List[str]], List[str]],
      dataset_fn,
      input_context: Optional[tf.distribute.InputContext] = None,
      tfds_builder: Optional[tfds.core.DatasetBuilder] = None):
    """Reads the data source (files/tfds) to a dataset."""

    def _files_to_dataset(files: List[str]) -> tf.data.Dataset:
      if len(files) > 1:
        if input_context and (len(files) < input_context.num_input_pipelines):
          logging.warn(
              'The number of files %d is less than the number of input pipelines '
              '%d. We will send all input files to every worker. '
              'Please consider sharding your data into more files.', len(files),
              input_context.num_input_pipelines)
          return _read_files_then_shard(
              files,
              dataset_fn,
              input_context,
              sharding=self._sharding,
              repeat=self._is_training and not self._cache)
        else:
          return _shard_files_then_read(
              files,
              dataset_fn,
              input_context,
              seed=self._seed,
              is_training=self._is_training,
              sharding=self._sharding,
              cache=self._cache,
              cycle_length=self._cycle_length,
              block_length=self._block_length,
              deterministic=self._deterministic)
      elif len(files) == 1:
        return _read_files_then_shard(
            files,
            dataset_fn,
            input_context,
            sharding=self._sharding,
            repeat=self._is_training and not self._cache)
      else:
        raise ValueError('It is unexpected that `tfds_builder` is None and '
                         'there is also no `files`.')

    if tfds_builder:
      dataset = _read_tfds(
          tfds_builder=self._tfds_builder,
          tfds_split=self._tfds_split,
          tfds_skip_decoding_feature=self._tfds_skip_decoding_feature,
          tfds_as_supervised=self._tfds_as_supervised,
          input_context=input_context,
          seed=self._seed,
          is_training=self._is_training,
          cache=self._cache,
          cycle_length=self._cycle_length,
          block_length=self._block_length)
    elif isinstance(matched_files, (list, tuple)):
      dataset = _files_to_dataset(matched_files)
    elif isinstance(matched_files, dict):
      dataset = {}
      for k, fs in matched_files.items():
        dataset[k] = _files_to_dataset(fs)
    else:
      raise ValueError('`matched_files` should be a list or dict.')

    return dataset

  def _decode_and_parse_dataset(
      self,
      dataset: Union[tf.data.Dataset, Dict[Text, tf.data.Dataset]],
      batch_size: int,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Returns a tf.data.Dataset object after shuffling, decoding, and parsing."""

    def _shuffle_and_decode(ds):
      # If cache is enabled, we will call `shuffle()` later after `cache()`.
      if self._is_training and not self._cache:
        ds = ds.shuffle(self._shuffle_buffer_size, seed=self._seed)
      # Decode
      ds = _maybe_map_fn(ds, self._decoder_fn)
      return ds

    dataset = tf.nest.map_structure(_shuffle_and_decode, dataset)
    if tf.nest.is_nested(dataset):
      dataset = self._combine_fn(dataset)

    if self._sample_fn is not None:
      dataset = dataset.apply(self._sample_fn)
    dataset = _maybe_map_fn(dataset, self._parser_fn)

    if self._cache:
      dataset = dataset.cache()
      if self._is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self._shuffle_buffer_size, seed=self._seed)

    # Applies tf.data service before batching operations. This is useful when
    # tf.data service is shared between parallel trainers, and batch size is
    # changing between parallel trainers. Then batch size is changing, tf.data
    # services will be considered different instances if applied after batching
    # operations, which make it difficult to share between parallel trainers.
    # However, if there are additional expensive operations in
    # self._transform_and_batch_fn and self._postprocess_fn, the entire tf.data
    # pipeline could be slowed down. In this case, try to move these dataset
    # operations into early stages if possible.
    if (self._enable_shared_tf_data_service_between_parallel_trainers and
        self._apply_tf_data_service_before_batching):
      dataset = self._maybe_apply_data_service(dataset, input_context)

    if self._transform_and_batch_fn is not None:
      dataset = self._transform_and_batch_fn(dataset, input_context)
    else:
      per_replica_batch_size = input_context.get_per_replica_batch_size(
          batch_size) if input_context else batch_size
      dataset = dataset.batch(
          per_replica_batch_size, drop_remainder=self._drop_remainder)

    return dataset

  def _maybe_apply_data_service(
      self,
      dataset: tf.data.Dataset,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Potentially distributes a dataset."""
    if self._enable_tf_data_service and input_context:
      if self._enable_round_robin_tf_data_service:
        replicas_per_input_pipeline = input_context.num_replicas_in_sync // (
            input_context.num_input_pipelines)
        base_consumer_index = input_context.input_pipeline_id * (
            replicas_per_input_pipeline)
        num_consumers = input_context.num_input_pipelines * (
            replicas_per_input_pipeline)
        range_dataset = tf.data.Dataset.range(replicas_per_input_pipeline)
        tfds_kwargs = {
            'processing_mode': 'parallel_epochs',
            'service': self._tf_data_service_address,
            'job_name': self._tf_data_service_job_name,
            'num_consumers': num_consumers
        }
        if self._enable_shared_tf_data_service_between_parallel_trainers:
          raise ValueError('Shared tf.data service does not support round-robin'
                           ' tf.data service.')
        dataset = range_dataset.map(lambda i: dataset.apply(  # pylint: disable=g-long-lambda
            tf.data.experimental.service.distribute(
                consumer_index=base_consumer_index + i, **tfds_kwargs)))
        # Use parallel interleave to read multiple batches from a tf.data
        # service worker in parallel.
        dataset = dataset.interleave(
            lambda x: x,
            cycle_length=replicas_per_input_pipeline,
            num_parallel_calls=replicas_per_input_pipeline,
            deterministic=True)
      else:
        tfds_kwargs = {
            'processing_mode': 'parallel_epochs',
            'service': self._tf_data_service_address,
            'job_name': self._tf_data_service_job_name,
        }
        if self._enable_shared_tf_data_service_between_parallel_trainers:
          tfds_kwargs.update({
              'processing_mode':
                  tf.data.experimental.service.ShardingPolicy.OFF,
              'cross_trainer_cache':
                  tf.data.experimental.service.CrossTrainerCache(
                      trainer_id=self._trainer_id)
          })
        dataset = dataset.apply(
            tf.data.experimental.service.distribute(**tfds_kwargs))
    return dataset

  def read(self,
           input_context: Optional[tf.distribute.InputContext] = None,
           dataset: Optional[tf.data.Dataset] = None) -> tf.data.Dataset:
    """Generates a tf.data.Dataset object."""
    if dataset is None:
      dataset = self._read_data_source(self._matched_files, self._dataset_fn,
                                       input_context, self._tfds_builder)
    dataset = self._decode_and_parse_dataset(dataset, self._global_batch_size,
                                             input_context)
    dataset = _maybe_map_fn(dataset, self._postprocess_fn)
    if not (self._enable_shared_tf_data_service_between_parallel_trainers and
            self._apply_tf_data_service_before_batching):
      dataset = self._maybe_apply_data_service(dataset, input_context)

    if self._deterministic is not None:
      options = tf.data.Options()
      options.experimental_deterministic = self._deterministic
      dataset = dataset.with_options(options)
    return dataset.prefetch(self._prefetch_buffer_size)
