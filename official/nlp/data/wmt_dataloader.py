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

"""Input pipeline for the transformer model to read, filter, and batch examples.

Batching scheme

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
from typing import Dict, Optional

import dataclasses
import tensorflow as tf
import tensorflow_text as tftxt
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


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
    buckets_min = [0, 4, 8, 16]
    buckets_max = [4, 8, 16, 25]

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
  sentencepiece_model_path: str = ''
  src_lang: str = ''
  tgt_lang: str = ''
  transform_and_batch: bool = True
  has_unique_id: bool = False


@data_loader_factory.register_data_loader_cls(WMTDataConfig)
class WMTDataLoader(data_loader.DataLoader):
  """A class to load dataset for WMT translation task."""

  def __init__(self, params: WMTDataConfig):
    self._params = params
    self._max_seq_length = params.max_seq_length
    self._static_batch = params.static_batch
    self._global_batch_size = params.global_batch_size
    if self._params.transform_and_batch:
      self._tokenizer = tftxt.SentencepieceTokenizer(
          model=tf.io.gfile.GFile(params.sentencepiece_model_path, 'rb').read(),
          add_eos=True)

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        self._params.src_lang: tf.io.FixedLenFeature([], tf.string),
        self._params.tgt_lang: tf.io.FixedLenFeature([], tf.string),
    }
    if self._params.has_unique_id:
      name_to_features['unique_id'] = tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t
    return example

  def _tokenize(self, inputs) -> Dict[str, tf.Tensor]:
    tokenized_inputs = {}
    for k, v in inputs.items():
      if k == self._params.src_lang:
        tokenized_inputs['inputs'] = self._tokenizer.tokenize(v)
      elif k == self._params.tgt_lang:
        tokenized_inputs['targets'] = self._tokenizer.tokenize(v)
      else:
        tokenized_inputs[k] = v
    print(tokenized_inputs)
    return tokenized_inputs

  def _filter_max_length(self, inputs):
    # return tf.constant(True)
    return tf.logical_and(
        tf.shape(inputs['inputs'])[0] <= self._max_seq_length,
        tf.shape(inputs['targets'])[0] <= self._max_seq_length)

  def _maybe_truncate(self, inputs):
    truncated_inputs = {}
    for k, v in inputs.items():
      if k == 'inputs' or k == 'targets':
        truncated_inputs[k] = tf.pad(
            v[:self._max_seq_length - 1], [[0, 1]],
            constant_values=1) if tf.shape(v)[0] > self._max_seq_length else v
      else:
        truncated_inputs[k] = v
    return truncated_inputs

  def _tokenize_bucketize_and_batch(
      self,
      dataset,
      input_context: Optional[tf.distribute.InputContext] = None):
    dataset = dataset.map(
        self._tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._params.is_training:
      dataset = dataset.filter(self._filter_max_length)
    else:
      dataset = dataset.map(
          self._maybe_truncate,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size
    if self._static_batch:
      padded_shapes = {}
      for name, _ in dataset.element_spec.items():
        if name == 'unique_id':
          padded_shapes[name] = []
        else:
          padded_shapes[name] = [self._max_seq_length
                                ] if self._static_batch else [None]
      batch_size = per_replica_batch_size
      if self._params.is_training:
        batch_size = int(batch_size // self._max_seq_length)
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes,
          drop_remainder=True)
    else:
      # Group and batch such that each batch has examples of similar length.
      dataset = _batch_examples(dataset, per_replica_batch_size,
                                self._max_seq_length)
    # Prefetch the next element to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    decoder_fn = None
    # Only decode for TFRecords.
    if self._params.input_path:
      decoder_fn = self._decode

    def _identity(
        dataset, input_context: Optional[tf.distribute.InputContext] = None):
      del input_context
      return dataset

    transform_and_batch_fn = _identity
    if self._params.transform_and_batch:
      transform_and_batch_fn = self._tokenize_bucketize_and_batch

    reader = input_reader.InputReader(
        params=self._params,
        decoder_fn=decoder_fn,
        transform_and_batch_fn=transform_and_batch_fn)
    return reader.read(input_context)
