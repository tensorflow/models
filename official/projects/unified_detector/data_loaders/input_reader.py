# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Input data reader.

Creates a tf.data.Dataset object from multiple input sstables and use a
provided data parser function to decode the serialized tf.Example and optionally
run data augmentation.
"""

import os
from typing import Any, Callable, List, Optional, Sequence, Union

import gin
from six.moves import map
import tensorflow as tf

from official.common import dataset_fn
from research.object_detection.utils import label_map_util
from official.core import config_definitions as cfg
from official.projects.unified_detector.data_loaders import universal_detection_parser  # pylint: disable=unused-import

FuncType = Callable[..., Any]


@gin.configurable(denylist=['is_training'])
class InputFn(object):
  """Input data reader class.

  Creates a tf.data.Dataset object from multiple datasets  (optionally performs
  weighted sampling between different datasets), parses the tf.Example message
  using `parser_fn`. The datasets can either be stored in SSTable or TfRecord.
  """

  def __init__(self,
               is_training: bool,
               batch_size: Optional[int] = None,
               data_root: str = '',
               input_paths: List[str] = gin.REQUIRED,
               dataset_type: str = 'tfrecord',
               use_sampling: bool = False,
               sampling_weights: Optional[Sequence[Union[int, float]]] = None,
               cycle_length: Optional[int] = 64,
               shuffle_buffer_size: Optional[int] = 512,
               parser_fn: Optional[FuncType] = None,
               parser_num_parallel_calls: Optional[int] = 64,
               max_intra_op_parallelism: Optional[int] = None,
               label_map_proto_path: Optional[str] = None,
               input_filter_fns: Optional[List[FuncType]] = None,
               input_training_filter_fns: Optional[Sequence[FuncType]] = None,
               dense_to_ragged_batch: bool = False,
               data_validator_fn: Optional[Callable[[Sequence[str]],
                                                    None]] = None):
    """Input reader constructor.

    Args:
      is_training: Boolean indicating TRAIN or EVAL.
      batch_size: Input data batch size. Ignored if batch size is passed through
        params. In that case, this can be None.
      data_root: All the relative input paths are based on this location.
      input_paths: Input file patterns.
      dataset_type: Can be 'sstable' or 'tfrecord'.
      use_sampling: Whether to perform weighted sampling between different
        datasets.
      sampling_weights: Unnormalized sampling weights. The length should be
        equal to `input_paths`.
      cycle_length: The number of input Datasets to interleave from in parallel.
        If set to None tf.data experimental autotuning is used.
      shuffle_buffer_size: The random shuffle buffer size.
      parser_fn: The function to run decoding and data augmentation. The
        function takes `is_training` as an input, which is passed from here.
      parser_num_parallel_calls: The number of parallel calls for `parser_fn`.
        The number of CPU cores is the suggested value. If set to None tf.data
        experimental autotuning is used.
      max_intra_op_parallelism: if set limits the max intra op parallelism of
        functions run on slices of the input.
      label_map_proto_path: Path to a StringIntLabelMap which will be used to
        decode the input data.
      input_filter_fns: A list of functions on the dataset points which returns
        true for valid data.
      input_training_filter_fns: A list of functions on the dataset points which
        returns true for valid data used only for training.
      dense_to_ragged_batch: Whether to use ragged batching for MPNN format.
      data_validator_fn: If not None, used to validate the data specified by
        input_paths.

    Raises:
      ValueError for invalid input_paths.
    """
    self._is_training = is_training

    if data_root:
      # If an input path is absolute this does not change it.
      input_paths = [os.path.join(data_root, value) for value in input_paths]

    self._input_paths = input_paths
    # Disables datasets sampling during eval.
    self._batch_size = batch_size
    if is_training:
      self._use_sampling = use_sampling
    else:
      self._use_sampling = False
    self._sampling_weights = sampling_weights
    self._cycle_length = (cycle_length if cycle_length else tf.data.AUTOTUNE)
    self._shuffle_buffer_size = shuffle_buffer_size
    self._parser_num_parallel_calls = (
        parser_num_parallel_calls
        if parser_num_parallel_calls else tf.data.AUTOTUNE)
    self._max_intra_op_parallelism = max_intra_op_parallelism
    self._label_map_proto_path = label_map_proto_path
    if label_map_proto_path:
      name_to_id = label_map_util.get_label_map_dict(label_map_proto_path)
      self._lookup_str_keys = list(name_to_id.keys())
      self._lookup_int_values = list(name_to_id.values())
    self._parser_fn = parser_fn
    self._input_filter_fns = input_filter_fns or []
    if is_training and input_training_filter_fns:
      self._input_filter_fns.extend(input_training_filter_fns)
    self._dataset_type = dataset_type
    self._dense_to_ragged_batch = dense_to_ragged_batch

    if data_validator_fn is not None:
      data_validator_fn(self._input_paths)

  @property
  def batch_size(self):
    return self._batch_size

  def __call__(
      self,
      params: cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Read and parse input datasets, return a tf.data.Dataset object."""
    # TPUEstimator passes the batch size through params.
    if params is not None and 'batch_size' in params:
      batch_size = params['batch_size']
    else:
      batch_size = self._batch_size

    per_replica_batch_size = input_context.get_per_replica_batch_size(
        batch_size) if input_context else batch_size

    with tf.name_scope('input_reader'):
      dataset = self._build_dataset_from_records()
      dataset_parser_fn = self._build_dataset_parser_fn()

      dataset = dataset.map(
          dataset_parser_fn, num_parallel_calls=self._parser_num_parallel_calls)
      for filter_fn in self._input_filter_fns:
        dataset = dataset.filter(filter_fn)

    if self._dense_to_ragged_batch:
      dataset = dataset.apply(
          tf.data.experimental.dense_to_ragged_batch(
              batch_size=per_replica_batch_size, drop_remainder=True))
    else:
      dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  def _fetch_dataset(self, filename: str) -> tf.data.Dataset:
    """Fetch dataset depending on type.

    Args:
      filename: Location of dataset.

    Returns:
      Tf Dataset.
    """

    data_cls = dataset_fn.pick_dataset_fn(self._dataset_type)

    data = data_cls([filename])
    return data

  def _build_dataset_parser_fn(self) -> Callable[..., tf.Tensor]:
    """Depending on label_map and storage type, build a parser_fn."""
    # Parse the fetched records to input tensors for model function.
    if self._label_map_proto_path:
      lookup_initializer = tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(self._lookup_str_keys, dtype=tf.string),
          values=tf.constant(self._lookup_int_values, dtype=tf.int32))
      name_to_id_table = tf.lookup.StaticHashTable(
          initializer=lookup_initializer, default_value=0)
      parser_fn = self._parser_fn(
          is_training=self._is_training, label_lookup_table=name_to_id_table)
    else:
      parser_fn = self._parser_fn(is_training=self._is_training)

    return parser_fn

  def _build_dataset_from_records(self) -> tf.data.Dataset:
    """Build a tf.data.Dataset object from input SSTables.

    If the input data come from multiple SSTables, use the user defined sampling
    weights to perform sampling. For example, if the sampling weights is
    [1., 2.], the second dataset will be sampled twice more often than the first
    one.

    Returns:
      Dataset built from SSTables.
    Raises:
      ValueError for inability to find SSTable files.
    """
    all_file_patterns = []
    if self._use_sampling:
      for file_pattern in self._input_paths:
        all_file_patterns.append([file_pattern])
      # Normalize sampling probabilities.
      total_weight = sum(self._sampling_weights)
      sampling_probabilities = [
          float(w) / total_weight for w in self._sampling_weights
      ]
    else:
      all_file_patterns.append(self._input_paths)

    datasets = []
    for file_pattern in all_file_patterns:
      filenames = sum(list(map(tf.io.gfile.glob, file_pattern)), [])
      if not filenames:
        raise ValueError(
            f'Error trying to read input files for file pattern {file_pattern}')
      # Create a dataset of filenames and shuffle the files. In each epoch,
      # the file order is shuffled again. This may help if
      # per_host_input_for_training = false on TPU.
      dataset = tf.data.Dataset.list_files(
          file_pattern, shuffle=self._is_training)

      if self._is_training:
        dataset = dataset.repeat()

      if self._max_intra_op_parallelism:
        # Disable intra-op parallelism to optimize for throughput instead of
        # latency.
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)

      dataset = dataset.interleave(
          self._fetch_dataset,
          cycle_length=self._cycle_length,
          num_parallel_calls=self._cycle_length,
          deterministic=(not self._is_training))

      if self._is_training:
        dataset = dataset.shuffle(self._shuffle_buffer_size)

      datasets.append(dataset)

    if self._use_sampling:
      assert len(datasets) == len(sampling_probabilities)
      dataset = tf.data.experimental.sample_from_datasets(
          datasets, sampling_probabilities)
    else:
      dataset = datasets[0]

    return dataset
