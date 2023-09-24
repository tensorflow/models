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

"""Loads dataset for the Pixel Classification task."""
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
class PixelDataConfig(cfg.DataConfig):
  """Data config for text classification task."""

  input_path: str = ''
  global_batch_size: int = 32
  is_training: bool = True
  label_type: str = 'int'
  num_channels: int = 3
  input_size: Tuple[int, int] = (16, 4096)
  patch_h: int = 16
  patch_w: int = 16
  # Whether to include the example id number.
  include_example_id: bool = False
  # Maps the key in TfExample to feature name.
  # Either tfrecord, sstable, or recordio.
  file_type: str = 'tfrecord'


@data_loader_factory.register_data_loader_cls(PixelDataConfig)
class PixelDataLoader(data_loader.DataLoader):
  """A class to load dataset for text classification task."""

  def __init__(self, params):
    self._params = params
    self._include_example_id = params.include_example_id

  def name_to_features_spec(self):
    """Defines features to decode. Subclass may override to append features."""
    h, w = self._params.input_size
    positions = h // self._params.patch_h * w // self._params.patch_w
    name_to_features = {
        'pixel_values': tf.io.FixedLenFeature(
            [self._params.num_channels, h, w], tf.float32
        ),
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([positions], tf.float32),
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
        'pixel_values': 'pixel_values',
        'label': 'label',
        'attention_mask': 'attention_mask',
    }
    ret = {}
    for record_key in record:
      if record_key in key_mapping:
        ret[key_mapping[record_key]] = record[record_key]
      else:
        ret[record_key] = record[record_key]
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
