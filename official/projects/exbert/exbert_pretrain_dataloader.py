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

"""Loads dataset for the ExBERT teacher pretraining task."""
from typing import Mapping, Optional

import tensorflow as tf

from official.core import input_reader
from official.nlp.data import data_loader


class ExbertTeacherPretrainDataLoader(data_loader.DataLoader):
  """Class to load mixed-vocabulary dataset for Exbert teacher pretrain task."""

  def __init__(self, params):
    """Inits `ExbertTeacherPretrainDataLoader` class.

    Args:
      params: A `BertPretrainDataConfig` object.
    """
    self._params = params
    self._seq_length = params.seq_length
    self._max_predictions_per_seq = params.max_predictions_per_seq
    self._use_next_sentence_label = params.use_next_sentence_label
    self._use_position_id = params.use_position_id

  def _name_to_features(self):
    name_to_features = {
        'input_mask':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_mask_teacher':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_seg_vocab_ids':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'masked_lm_positions':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_positions_teacher_tvocab':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_positions_teacher_svocab':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_ids_teacher_tvocab':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_ids_teacher_svocab':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
    }
    if self._params.use_v2_feature_names:
      name_to_features.update({
          'input_word_ids':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'input_word_ids_teacher':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'input_type_ids':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'input_type_ids_teacher':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
      })
    else:
      name_to_features.update({
          'input_ids':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'input_ids_teacher':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'segment_ids':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'segment_ids_teacher':
              tf.io.FixedLenFeature([self._seq_length], tf.int64),
      })
    if self._use_next_sentence_label:
      name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1],
                                                                       tf.int64)
    if self._use_position_id:
      name_to_features['position_ids'] = tf.io.FixedLenFeature(
          [self._seq_length], tf.int64)
    return name_to_features

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = self._name_to_features()
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {
        'input_mask':
            record['input_mask'],
        'input_mask_teacher':
            record['input_mask_teacher'],
        'input_seg_vocab_ids':
            record['input_seg_vocab_ids'],
        'masked_lm_positions':
            record['masked_lm_positions'],
        'masked_lm_positions_teacher_tvocab':
            record['masked_lm_positions_teacher_tvocab'],
        'masked_lm_positions_teacher_svocab':
            record['masked_lm_positions_teacher_svocab'],
        'masked_lm_ids':
            record['masked_lm_ids'],
        'masked_lm_ids_teacher_tvocab':
            record['masked_lm_ids_teacher_tvocab'],
        'masked_lm_ids_teacher_svocab':
            record['masked_lm_ids_teacher_svocab'],
    }
    if self._params.use_v2_feature_names:
      x['input_word_ids'] = record['input_word_ids']
      x['input_word_ids_teacher'] = record['input_word_ids_teacher']
      x['input_type_ids'] = record['input_type_ids']
      x['input_type_ids_teacher'] = record['input_type_ids_teacher']
    else:
      x['input_word_ids'] = record['input_ids']
      x['input_word_ids_teacher'] = record['input_ids_teacher']
      x['input_type_ids'] = record['segment_ids']
      x['input_type_ids_teacher'] = record['segment_ids_teacher']
    if self._use_next_sentence_label:
      x['next_sentence_labels'] = record['next_sentence_labels']
    if self._use_position_id:
      x['position_ids'] = record['position_ids']

    return x

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)
