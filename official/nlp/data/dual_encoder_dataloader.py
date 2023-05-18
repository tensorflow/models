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

"""Loads dataset for the dual encoder (retrieval) task."""
import dataclasses
import functools
import itertools
from typing import Iterable, Mapping, Optional, Tuple

import tensorflow as tf
import tensorflow_hub as hub

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory
from official.nlp.modeling import layers


@dataclasses.dataclass
class DualEncoderDataConfig(cfg.DataConfig):
  """Data config for dual encoder task (tasks/dual_encoder)."""
  # Either set `input_path`...
  input_path: str = ''
  # ...or `tfds_name` and `tfds_split` to specify input.
  tfds_name: str = ''
  tfds_split: str = ''
  global_batch_size: int = 32
  # Either build preprocessing with Python code by specifying these values...
  vocab_file: str = ''
  lower_case: bool = True
  # ...or load preprocessing from a SavedModel at this location.
  preprocessing_hub_module_url: str = ''

  left_text_fields: Tuple[str] = ('left_input',)
  right_text_fields: Tuple[str] = ('right_input',)
  is_training: bool = True
  seq_length: int = 128
  file_type: str = 'tfrecord'


@data_loader_factory.register_data_loader_cls(DualEncoderDataConfig)
class DualEncoderDataLoader(data_loader.DataLoader):
  """A class to load dataset for dual encoder task (tasks/dual_encoder)."""

  def __init__(self, params):
    if bool(params.tfds_name) == bool(params.input_path):
      raise ValueError('Must specify either `tfds_name` and `tfds_split` '
                       'or `input_path`.')
    if bool(params.vocab_file) == bool(params.preprocessing_hub_module_url):
      raise ValueError('Must specify exactly one of vocab_file (with matching '
                       'lower_case flag) or preprocessing_hub_module_url.')
    self._params = params
    self._seq_length = params.seq_length
    self._left_text_fields = params.left_text_fields
    self._right_text_fields = params.right_text_fields

    if params.preprocessing_hub_module_url:
      preprocessing_hub_module = hub.load(params.preprocessing_hub_module_url)
      self._tokenizer = preprocessing_hub_module.tokenize
      self._pack_inputs = functools.partial(
          preprocessing_hub_module.bert_pack_inputs,
          seq_length=params.seq_length)
    else:
      self._tokenizer = layers.BertTokenizer(
          vocab_file=params.vocab_file, lower_case=params.lower_case)
      self._pack_inputs = layers.BertPackInputs(
          seq_length=params.seq_length,
          special_tokens_dict=self._tokenizer.get_special_tokens_dict())

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        x: tf.io.FixedLenFeature([], tf.string)
        for x in itertools.chain(
            *[self._left_text_fields, self._right_text_fields])
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

  def _bert_tokenize(
      self, record: Mapping[str, tf.Tensor],
      text_fields: Iterable[str]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Tokenize the input in text_fields using BERT tokenizer.

    Args:
      record: A tfexample record contains the features.
      text_fields: A list of fields to be tokenzied.

    Returns:
      The tokenized features in a tuple of (input_word_ids, input_mask,
      input_type_ids).
    """
    segments_text = [record[x] for x in text_fields]
    segments_tokens = [self._tokenizer(s) for s in segments_text]
    segments = [tf.cast(x.merge_dims(1, 2), tf.int32) for x in segments_tokens]
    return self._pack_inputs(segments)

  def _bert_preprocess(
      self, record: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Perform the bert word piece tokenization for left and right inputs."""

    def _switch_prefix(string, old, new):
      if string.startswith(old): return new + string[len(old):]
      raise ValueError('Expected {} to start with {}'.format(string, old))

    def _switch_key_prefix(d, old, new):
      return {_switch_prefix(key, old, new): value for key, value in d.items()}  # pytype: disable=attribute-error  # trace-all-classes

    model_inputs = _switch_key_prefix(
        self._bert_tokenize(record, self._left_text_fields),
        'input_', 'left_')
    model_inputs.update(_switch_key_prefix(
        self._bert_tokenize(record, self._right_text_fields),
        'input_', 'right_'))
    return model_inputs

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        # Skip `decoder_fn` for tfds input.
        decoder_fn=self._decode if self._params.input_path else None,
        dataset_fn=dataset_fn.pick_dataset_fn(self._params.file_type),
        postprocess_fn=self._bert_preprocess)
    return reader.read(input_context)
