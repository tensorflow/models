# Lint as: python3
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
"""Loads dataset for the question answering (e.g, SQuAD) task."""
from typing import Mapping, Optional

import dataclasses
import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory


@dataclasses.dataclass
class QADataConfig(cfg.DataConfig):
  """Data config for question answering task (tasks/question_answering)."""
  # For training, `input_path` is expected to be a pre-processed TFRecord file,
  # while for evaluation, it is expected to be a raw JSON file (b/173814590).
  input_path: str = ''
  global_batch_size: int = 48
  is_training: bool = True
  seq_length: int = 384
  # Settings below are question answering specific.
  version_2_with_negative: bool = False
  # Settings below are only used for eval mode.
  input_preprocessed_data_path: str = ''
  doc_stride: int = 128
  query_length: int = 64
  # The path to the vocab file of word piece tokenizer or the
  # model of the sentence piece tokenizer.
  vocab_file: str = ''
  tokenization: str = 'WordPiece'  # WordPiece or SentencePiece
  do_lower_case: bool = True
  xlnet_format: bool = False


@data_loader_factory.register_data_loader_cls(QADataConfig)
class QuestionAnsweringDataLoader(data_loader.DataLoader):
  """A class to load dataset for sentence prediction (classification) task."""

  def __init__(self, params):
    self._params = params
    self._seq_length = params.seq_length
    self._is_training = params.is_training
    self._xlnet_format = params.xlnet_format

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
    }
    if self._xlnet_format:
      name_to_features['class_index'] = tf.io.FixedLenFeature([], tf.int64)
      name_to_features['paragraph_mask'] = tf.io.FixedLenFeature(
          [self._seq_length], tf.int64)
      if self._is_training:
        name_to_features['is_impossible'] = tf.io.FixedLenFeature([], tf.int64)

    if self._is_training:
      name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
      name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)
    else:
      name_to_features['unique_ids'] = tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions', 'is_impossible'):
        y[name] = tensor
      elif name == 'input_ids':
        x['input_word_ids'] = tensor
      elif name == 'segment_ids':
        x['input_type_ids'] = tensor
      else:
        x[name] = tensor
      if name == 'start_positions' and self._xlnet_format:
        x[name] = tensor
    return (x, y)

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)
