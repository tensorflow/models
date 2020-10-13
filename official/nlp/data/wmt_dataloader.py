# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Input pipeline for the transformer model to read, filter, and batch examples.

1. Batching scheme

   The examples encoded in the TFRecord files contain data in the format:
     {'inputs': [variable length array of integers],
      'targets': [variable length array of integers]}
   Where integers in the arrays refer to tokens in the English and German vocab
   file (named `vocab.ende.32768`).

   Prior to batching, elements in the dataset are grouped by length (max between
   'inputs' and 'targets' length). Each group is then batched such that:
     group_batch_size * length <= batch_size.

   Another way to view batch_size is the maximum number of tokens in each batch.

   Once batched, each element in the dataset will have the shape:
     {'inputs': [group_batch_size, padded_input_length],
      'targets': [group_batch_size, padded_target_length]}
   Lengths are padded to the longest 'inputs' or 'targets' sequence in the batch
   (padded_input_length and padded_target_length can be different).

   This batching scheme decreases the fraction of padding tokens per training
   batch, thus improving the training speed significantly.
"""
from typing import Optional

import dataclasses
import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

# Buffer size for reading records from a TFRecord file. Each training file is
# 7.2 MB, so 8 MB allows an entire file to be kept in memory.
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.logical_and(
      tf.size(example[0]) <= max_length,
      tf.size(example[1]) <= max_length)


def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
  return length


def _create_min_max_boundaries(max_length,
                               min_boundary=_MIN_BOUNDARY,
                               boundary_scale=_BOUNDARY_SCALE):
  """Create min and max boundary lists up to max_length.

  For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
  returned values will be:
    buckets_min = [0, 4, 8, 16, 24]
    buckets_max = [4, 8, 16, 24, 25]

  Args:
    max_length: The maximum length of example in dataset.
    min_boundary: Minimum length in boundary.
    boundary_scale: Amount to scale consecutive boundaries in the list.

  Returns:
    min and max boundary lists

  """
  # Create bucket boundaries list by scaling the previous boundary or adding 1
  # (to ensure increasing boundary sizes).
  bucket_boundaries = []
  x = min_boundary
  while x < max_length:
    bucket_boundaries.append(x)
    x = max(x + 1, int(x * boundary_scale))

  # Create min and max boundary lists from the initial list.
  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length + 1]
  return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
  """Group examples by similar lengths, and return batched dataset.

  Each batch of similar-length examples are padded to the same length, and may
  have different number of elements in each batch, such that:
    group_batch_size * padded_length <= batch_size.

  This decreases the number of padding tokens per batch, which improves the
  training speed.

  Args:
    dataset: Dataset of unbatched examples.
    batch_size: Max number of tokens per batch of examples.
    max_length: Max number of tokens in an example input or target sequence.

  Returns:
    Dataset of batched examples with similar lengths.
  """
  # Get min and max boundary lists for each example. These are used to calculate
  # the `bucket_id`, which is the index at which:
  # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
  # Note that using both min and max lists improves the performance.
  buckets_min, buckets_max = _create_min_max_boundaries(max_length)

  # Create list of batch sizes for each bucket_id, so that
  # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
  bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]

  # Validates bucket batch sizes.
  if any([batch_size <= 0 for batch_size in bucket_batch_sizes]):
    raise ValueError(
        'The token budget, global batch size, is too small to yeild 0 bucket '
        'window: %s' % str(bucket_batch_sizes))

  # bucket_id will be a tensor, so convert this list to a tensor as well.
  bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

  def example_to_bucket_id(example):
    """Return int64 bucket id for this example, calculated based on length."""
    example_input = example['inputs']
    example_target = example['targets']
    seq_length = _get_example_length((example_input, example_target))

    conditions_c = tf.logical_and(
        tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
    bucket_id = tf.reduce_min(tf.where(conditions_c))
    return bucket_id

  def window_size_fn(bucket_id):
    """Return number of examples to be grouped when given a bucket id."""
    return bucket_batch_sizes[bucket_id]

  def batching_fn(bucket_id, grouped_dataset):
    """Batch and add padding to a dataset of elements with similar lengths."""
    bucket_batch_size = window_size_fn(bucket_id)

    # Batch the dataset and add padding so that all input sequences in the
    # examples have the same length, and all target sequences have the same
    # lengths as well. Resulting lengths of inputs and targets can differ.
    padded_shapes = dict([
        (name, [None] * len(spec.shape))
        for name, spec in grouped_dataset.element_spec.items()
    ])
    return grouped_dataset.padded_batch(bucket_batch_size, padded_shapes)

  return dataset.apply(
      tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn))


@dataclasses.dataclass
class WMTDataConfig(cfg.DataConfig):
  """Data config for WMT translation."""
  max_seq_length: int = 64
  static_batch: bool = False
  vocab_file: str = ''


@data_loader_factory.register_data_loader_cls(WMTDataConfig)
class WMTDataLoader(data_loader.DataLoader):
  """A class to load dataset for WMT translation task."""

  def __init__(self, params: WMTDataConfig):
    self._params = params
    self._max_seq_length = params.max_seq_length
    self._static_batch = params.static_batch
    self._global_batch_size = params.global_batch_size

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    if self._params.is_training:
      name_to_features = {
          'inputs': tf.io.VarLenFeature(tf.int64),
          'targets': tf.io.VarLenFeature(tf.int64)
      }
      example = tf.io.parse_single_example(record, name_to_features)
      example['inputs'] = tf.sparse.to_dense(example['inputs'])
      example['targets'] = tf.sparse.to_dense(example['targets'])
    else:
      name_to_features = {
          'inputs': tf.io.VarLenFeature(tf.int64),
          'unique_id': tf.io.FixedLenFeature([], tf.int64)
      }
      example = tf.io.parse_single_example(record, name_to_features)
      example['inputs'] = tf.sparse.to_dense(example['inputs'])
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t
    return example

  def _bucketize_and_batch(
      self,
      dataset,
      input_context: Optional[tf.distribute.InputContext] = None):
    # pylint: disable=g-long-lambda
    dataset = dataset.filter(lambda x: _filter_max_length(
        (x['inputs'], x['targets']), self._max_seq_length))
    # pylint: enable=g-long-lambda
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size
    if self._static_batch:
      padded_shapes = dict([(name, [self._max_seq_length])
                            for name, _ in dataset.element_spec.items()])
      dataset = dataset.padded_batch(
          int(per_replica_batch_size // self._max_seq_length),
          padded_shapes,
          drop_remainder=True)
    else:
      # Group and batch such that each batch has examples of similar length.
      dataset = _batch_examples(dataset, per_replica_batch_size,
                                self._max_seq_length)
    # Prefetch the next element to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

  def _inference_padded_batch(
      self,
      dataset,
      input_context: Optional[tf.distribute.InputContext] = None):
    padded_shapes = {}
    for name, _ in dataset.element_spec.items():
      if name == 'unique_id':
        padded_shapes[name] = []
      else:
        padded_shapes[name] = [self._max_seq_length
                              ] if self._static_batch else [None]
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size
    return dataset.padded_batch(
        per_replica_batch_size, padded_shapes, drop_remainder=True)

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        decoder_fn=self._decode,
        transform_and_batch_fn=self._bucketize_and_batch
        if self._params.is_training else self._inference_padded_batch)
    return reader.read(input_context)
