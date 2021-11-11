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

"""Loads dataset for classification tasks."""
from typing import Dict, Mapping, Optional, Tuple

import dataclasses
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader


@dataclasses.dataclass
class ClassificationExampleDataConfig(cfg.DataConfig):
  """Data config for token classification task."""
  seq_length: int = 128


class ClassificationDataLoader(data_loader.DataLoader):
  """A class to load dataset for sentence prediction (classification) task."""

  def __init__(self, params):
    self._params = params
    self._seq_length = params.seq_length

  def _decode(self, record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Decodes a serialized tf.Example."""

    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(
      self,
      record: Mapping[str,
                      tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }

    y = record['label_ids']
    return (x, y)

  def load(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)
