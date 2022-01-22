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

"""Dataset loader for the pre-training with dynamic sequence length."""
from typing import Optional, Tuple

import dataclasses
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader_factory
from official.nlp.data import pretrain_dataloader


@dataclasses.dataclass
class BertPretrainDataConfig(cfg.DataConfig):
  """Data config for BERT pretraining task (tasks/masked_lm)."""
  input_path: str = ''
  global_batch_size: int = 512
  is_training: bool = True
  seq_bucket_lengths: Tuple[int, ...] = (128, 256, 384, 512,)
  # TODO(rxsang): `seq_bucket_window_scale` is only useful when round robin
  # tf.data service is disabled. Deprecate this flag once we always enable round
  # robin tf.data service.
  seq_bucket_window_scale: int = 8
  use_next_sentence_label: bool = True
  use_position_id: bool = False
  deterministic: bool = False
  enable_tf_data_service: bool = False
  enable_round_robin_tf_data_service: bool = False
  tf_data_service_job_name: str = 'bert_pretrain'
  use_v2_feature_names: bool = False


@data_loader_factory.register_data_loader_cls(BertPretrainDataConfig)
class PretrainingDynamicDataLoader(pretrain_dataloader.BertPretrainDataLoader):
  """Dataset loader for bert-style pretraining with dynamic sequenece length.

  Bucketizes the input id features by the seq_bucket_lengths and features are
  padded to the bucket boundaries. The mask features are usually short than
  input id features and can also be dynamic. We require the mask feature lengths
  within a bucket must be the same. For example, with [128, 256] buckets,
  the mask features for bucket 128 should always have the length as X and
  features for bucket 256 should always have the length as Y.

  The dataloader does not filter out empty masks. Make sure to handle this
  in the model.
  """

  def __init__(self, params):
    self._params = params
    if len(params.seq_bucket_lengths) < 1:
      raise ValueError('The seq_bucket_lengths cannot be empty.')
    self._seq_bucket_lengths = params.seq_bucket_lengths
    self._seq_bucket_window_scale = params.seq_bucket_window_scale
    self._global_batch_size = params.global_batch_size
    self._use_next_sentence_label = params.use_next_sentence_label
    self._use_position_id = params.use_position_id
    self._drop_remainder = params.drop_remainder
    self._enable_tf_data_service = params.enable_tf_data_service
    self._enable_round_robin_tf_data_service = (
        params.enable_round_robin_tf_data_service)
    self._mask_keys = [
        'masked_lm_positions', 'masked_lm_ids', 'masked_lm_weights'
    ]

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        'input_mask': tf.io.VarLenFeature(tf.int64),
        'masked_lm_positions': tf.io.VarLenFeature(tf.int64),
        'masked_lm_ids': tf.io.VarLenFeature(tf.int64),
        'masked_lm_weights': tf.io.VarLenFeature(tf.float32),
    }
    if self._params.use_v2_feature_names:
      input_ids_key = 'input_word_ids'
      segment_key = 'input_type_ids'
      name_to_features.update({
          input_ids_key: tf.io.VarLenFeature(tf.int64),
          segment_key: tf.io.VarLenFeature(tf.int64),
      })
    else:
      input_ids_key = 'input_ids'
      segment_key = 'segment_ids'
      name_to_features.update({
          input_ids_key: tf.io.VarLenFeature(tf.int64),
          segment_key: tf.io.VarLenFeature(tf.int64),
      })
    if self._use_next_sentence_label:
      name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1],
                                                                       tf.int64)
    dynamic_keys = [input_ids_key, 'input_mask', segment_key]
    if self._use_position_id:
      name_to_features['position_ids'] = tf.io.VarLenFeature(tf.int64)
      dynamic_keys.append('position_ids')

    example = tf.io.parse_single_example(record, name_to_features)
    for key in dynamic_keys + self._mask_keys:
      example[key] = tf.sparse.to_dense(example[key])

    # Truncate padded data after the first non pad in the
    # sequence length dimension.
    # Pad before the first non pad from the back should not be removed.
    mask = tf.math.greater(
        tf.math.cumsum(example[input_ids_key], reverse=True), 0)
    for key in dynamic_keys:
      example[key] = tf.boolean_mask(example[key], mask)

    # masked_lm_ids should be 0 padded.
    # Change mask features to -1 padding so that we can differentiate
    # padding from data or from bucketizing.
    mask = tf.math.not_equal(example['masked_lm_ids'], 0)
    example['masked_lm_ids'] = tf.where(
        mask, example['masked_lm_ids'],
        -tf.ones(tf.shape(example['masked_lm_ids']), dtype=example[key].dtype))

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    # tf.data service uses dataset graph fingerprint to distinguish input
    # pipeline jobs, thus we sort the keys here to make sure they are generated
    # in a deterministic order each time the dataset function is traced.
    for name in sorted(list(example.keys())):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _bucketize_and_batch(
      self,
      dataset,
      input_context: Optional[tf.distribute.InputContext] = None):
    """Bucketize by sequence length and batch the datasets."""
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size

    def element_length_func(example, seq_len_dim):
      return tf.shape(example['input_word_ids'])[seq_len_dim]

    bucket_boundaries = [length + 1 for length in self._seq_bucket_lengths]
    bucket_batch_sizes = [per_replica_batch_size] * (len(bucket_boundaries) + 1)

    # Bucketize and batch the dataset with per replica batch size first.
    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            lambda example: tf.cast(element_length_func(example, 0), tf.int32),
            bucket_boundaries,
            bucket_batch_sizes,
            pad_to_bucket_boundary=True,
            drop_remainder=self._drop_remainder))
    if input_context:
      window_size = input_context.num_replicas_in_sync
      if self._enable_tf_data_service and (
          not self._enable_round_robin_tf_data_service):
        # If tf.data service is enabled but round-robin behavior is not enabled,
        # different TPU workers may fetch data from one tf.data service worker
        # in different speed. We set the window size to be
        # `seq_bucket_window_scale` larger to leave buffer if some workers are
        # fetching data faster than others, so all the data within the same
        # global batch can still have more chances to be in the same bucket.
        window_size *= self._seq_bucket_window_scale

      # Group `num_replicas_in_sync` batches from same bucket together, so all
      # replicas can get the same sequence length for one global step.
      dataset = dataset.apply(
          tf.data.experimental.group_by_window(
              key_func=lambda example: tf.cast(  # pylint: disable=g-long-lambda
                  element_length_func(example, 1), tf.int64),
              reduce_func=lambda _, x: tf.data.Dataset.from_tensors(x),
              window_size=window_size))
      dataset = dataset.flat_map(lambda x: x)

    def _remove_pads_from_bucketize(features):
      # All mask features must have the same effective length.
      # The real masked ids padding token is -1 and 0 comes from
      # bucket_by_sequence_length.
      mask = tf.math.not_equal(features['masked_lm_ids'], 0)

      mask_per_example = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=1)
      normalized = tf.cast(
          mask_per_example / tf.math.reduce_max(mask_per_example), tf.int32)
      assert_op = tf.debugging.assert_equal(
          tf.math.reduce_sum(normalized), per_replica_batch_size,
          'Number of non padded mask tokens is not the same for each example '
          'in the same sequence length.')
      with tf.control_dependencies([assert_op]):
        for key in self._mask_keys:
          features[key] = tf.reshape(
              tf.boolean_mask(
                  features[key], mask), [per_replica_batch_size, -1])
      # Revert masked_lm_ids to be 0-padded.
      mask = tf.math.not_equal(features['masked_lm_ids'], -1)
      features['masked_lm_ids'] = tf.where(
          mask, features['masked_lm_ids'],
          tf.zeros(
              tf.shape(features['masked_lm_ids']),
              dtype=features['masked_lm_ids'].dtype))
      return features

    dataset = dataset.map(_remove_pads_from_bucketize)
    return dataset

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        decoder_fn=self._decode,
        parser_fn=self._parse,
        transform_and_batch_fn=self._bucketize_and_batch)
    return reader.read(input_context)
