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
"""Loads dataset for the similarity comparison (classification) task."""

import dataclasses
from typing import Mapping, Optional, Tuple

import tensorflow as tf

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory


LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}


@dataclasses.dataclass
class DualEncoderDataConfig(cfg.DataConfig):
  """Data config for similarity comparison task."""

  input_path: str = ''
  global_batch_size: int = 32
  is_training: bool = True
  seq_length: int = 128
  label_type: str = 'int'
  # Whether to include the example id number.
  include_example_id: bool = False
  label_field: str = 'label_ids'
  # Maps the key in TfExample to feature name.
  # E.g 'label_ids' to 'next_sentence_labels'
  label_name: Optional[Tuple[str, str]] = None
  # Either tfrecord, sstable, or recordio.
  file_type: str = 'tfrecord'


@data_loader_factory.register_data_loader_cls(DualEncoderDataConfig)
class DualEncoderDataLoader(data_loader.DataLoader):
  """A class to load dataset for similarity comparison (classification) task."""

  def __init__(self, params):
    self._params = params
    self._seq_length = params.seq_length
    self._include_example_id = params.include_example_id
    self._label_field = params.label_field
    if params.label_name:
      self._label_name_mapping = dict([params.label_name])
    else:
      self._label_name_mapping = dict()

  def name_to_features_spec(self):
    """Defines features to decode. Subclass may override to append features."""
    label_type = LABEL_TYPES_MAP[self._params.label_type]
    name_to_features = {
        'left_word_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'left_mask': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'right_word_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'right_mask': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        self._label_field: tf.io.FixedLenFeature([], label_type),
    }
    if self._include_example_id:
      name_to_features['example_id'] = tf.io.FixedLenFeature([], tf.int64)

    return name_to_features

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    example = tf.io.parse_single_example(record, self.name_to_features_spec())

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
    key_mapping = {
        'left_ids': 'left_word_ids',
        'left_mask': 'left_mask',
        'right_ids': 'right_word_ids',
        'right_mask': 'right_mask',
    }
    ret = {}
    for record_key in record:
      if record_key in key_mapping:
        ret[key_mapping[record_key]] = record[record_key]
      else:
        ret[record_key] = record[record_key]

    if self._label_field in self._label_name_mapping:
      ret[self._label_name_mapping[self._label_field]] = record[
          self._label_field
      ]

    return ret

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        dataset_fn=dataset_fn.pick_dataset_fn(self._params.file_type),
        params=self._params,
        decoder_fn=self._decode,
        parser_fn=self._parse,
    )
    return reader.read(input_context)
