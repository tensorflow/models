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

"""Loads text dataset for the BERT pretraining task."""
import dataclasses
from typing import List, Mapping, Optional, Text

import tensorflow as tf, tf_keras
import tensorflow_text as tf_text

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory
from official.nlp.modeling.ops import segment_extractor


@dataclasses.dataclass
class BertPretrainTextDataConfig(cfg.DataConfig):
  """Data config for BERT pretraining task (tasks/masked_lm) from text."""
  input_path: str = ""
  doc_batch_size: int = 8
  global_batch_size: int = 512
  is_training: bool = True
  seq_length: int = 512
  max_predictions_per_seq: int = 76
  use_next_sentence_label: bool = True
  # The name of the text feature fields. The text features will be
  # concatenated in order.
  # Note: More than 1 field name is not compatible with NSP.
  text_field_names: Optional[List[str]] = dataclasses.field(
      default_factory=lambda: ["text"])
  vocab_file_path: str = ""
  masking_rate: float = 0.15
  use_whole_word_masking: bool = False
  file_type: str = "tfrecord"


_CLS_TOKEN = b"[CLS]"
_SEP_TOKEN = b"[SEP]"
_MASK_TOKEN = b"[MASK]"
_NUM_OOV_BUCKETS = 1
# Accounts for [CLS] and 2 x [SEP] tokens
_NUM_SPECIAL_TOKENS = 3


@data_loader_factory.register_data_loader_cls(BertPretrainTextDataConfig)
class BertPretrainTextDataLoader(data_loader.DataLoader):
  """A class to load text dataset for BERT pretraining task."""

  def __init__(self, params):
    """Inits `BertPretrainTextDataLoader` class.

    Args:
      params: A `BertPretrainTextDataConfig` object.
    """
    if len(params.text_field_names) > 1 and params.use_next_sentence_label:
      raise ValueError("Currently there is no support for more than text field "
                       "while generating next sentence labels.")

    self._params = params
    self._seq_length = params.seq_length
    self._max_predictions_per_seq = params.max_predictions_per_seq
    self._use_next_sentence_label = params.use_next_sentence_label
    self._masking_rate = params.masking_rate
    self._use_whole_word_masking = params.use_whole_word_masking

    lookup_table_init = tf.lookup.TextFileInitializer(
        params.vocab_file_path,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    self._vocab_lookup_table = tf.lookup.StaticVocabularyTable(
        lookup_table_init,
        num_oov_buckets=_NUM_OOV_BUCKETS,
        lookup_key_dtype=tf.string)

    self._cls_token = self._vocab_lookup_table.lookup(tf.constant(_CLS_TOKEN))
    self._sep_token = self._vocab_lookup_table.lookup(tf.constant(_SEP_TOKEN))
    self._mask_token = self._vocab_lookup_table.lookup(tf.constant(_MASK_TOKEN))

    # -_NUM_OOV_BUCKETS to offset unused OOV bucket.
    self._vocab_size = self._vocab_lookup_table.size() - _NUM_OOV_BUCKETS

  def _decode(self, record: tf.Tensor) -> Mapping[Text, tf.Tensor]:
    """Decodes a serialized tf.Example."""
    name_to_features = {}
    for text_field_name in self._params.text_field_names:
      name_to_features[text_field_name] = tf.io.FixedLenFeature([], tf.string)
    return tf.io.parse_single_example(record, name_to_features)

  def _tokenize(self, segments):
    """Tokenize the input segments."""
    # Tokenize segments
    tokenizer = tf_text.BertTokenizer(
        self._vocab_lookup_table, token_out_type=tf.int64)

    if self._use_whole_word_masking:
      # tokenize the segments which should have the shape:
      # [num_sentence, (num_words), (num_wordpieces)]
      segments = [tokenizer.tokenize(s) for s in segments]
    else:
      # tokenize the segments and merge out the token dimension so that each
      # segment has the shape: [num_sentence, (num_wordpieces)]
      segments = [tokenizer.tokenize(s).merge_dims(-2, -1) for s in segments]

    # Truncate inputs
    trimmer = tf_text.WaterfallTrimmer(
        self._seq_length - _NUM_SPECIAL_TOKENS, axis=-1)
    truncated_segments = trimmer.trim(segments)

    # Combine segments, get segment ids and add special tokens
    return tf_text.combine_segments(
        truncated_segments,
        start_of_sequence_id=self._cls_token,
        end_of_segment_id=self._sep_token)

  def _bert_preprocess(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    if self._use_next_sentence_label:
      input_text = record[self._params.text_field_names[0]]
      # Split sentences
      sentence_breaker = tf_text.RegexSplitter()
      sentences = sentence_breaker.split(input_text)

      # Extract next-sentence-prediction labels and segments
      next_or_random_segment, is_next = (
          segment_extractor.get_next_sentence_labels(sentences))
      # merge dims to change shape from [num_docs, (num_segments)] to
      # [total_num_segments]
      is_next = is_next.merge_dims(-2, -1)

      # construct segments with shape [(num_sentence)]
      segments = [
          sentences.merge_dims(-2, -1),
          next_or_random_segment.merge_dims(-2, -1)
      ]
    else:
      segments = [record[name] for name in self._params.text_field_names]

    segments_combined, segment_ids = self._tokenize(segments)

    # Dynamic masking
    item_selector = tf_text.RandomItemSelector(
        self._max_predictions_per_seq,
        selection_rate=self._masking_rate,
        unselectable_ids=[self._cls_token, self._sep_token],
        shuffle_fn=(tf.identity if self._params.deterministic else None))
    values_chooser = tf_text.MaskValuesChooser(
        vocab_size=self._vocab_size, mask_token=self._mask_token)
    masked_input_ids, masked_lm_positions, masked_lm_ids = (
        tf_text.mask_language_model(
            segments_combined,
            item_selector=item_selector,
            mask_values_chooser=values_chooser,
        ))

    # Pad out to fixed shape and get input mask.
    seq_lengths = {
        "input_word_ids": self._seq_length,
        "input_type_ids": self._seq_length,
        "masked_lm_positions": self._max_predictions_per_seq,
        "masked_lm_ids": self._max_predictions_per_seq,
    }
    model_inputs = {
        "input_word_ids": masked_input_ids,
        "input_type_ids": segment_ids,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
    }
    padded_inputs_and_mask = tf.nest.map_structure(tf_text.pad_model_inputs,
                                                   model_inputs, seq_lengths)
    model_inputs = {
        k: padded_inputs_and_mask[k][0] for k in padded_inputs_and_mask
    }
    model_inputs["masked_lm_weights"] = tf.cast(
        padded_inputs_and_mask["masked_lm_ids"][1], tf.float32)
    model_inputs["input_mask"] = padded_inputs_and_mask["input_word_ids"][1]

    if self._use_next_sentence_label:
      model_inputs["next_sentence_labels"] = is_next

    for name in model_inputs:
      t = model_inputs[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      model_inputs[name] = t

    return model_inputs

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""

    def _batch_docs(dataset, input_context):
      per_core_doc_batch_size = (
          input_context.get_per_replica_batch_size(self._params.doc_batch_size)
          if input_context else self._params.doc_batch_size)
      return dataset.batch(per_core_doc_batch_size)

    reader = input_reader.InputReader(
        params=self._params,
        dataset_fn=dataset_fn.pick_dataset_fn(self._params.file_type),
        decoder_fn=self._decode if self._params.input_path else None,
        transform_and_batch_fn=_batch_docs
        if self._use_next_sentence_label else None,
        postprocess_fn=self._bert_preprocess)
    transformed_inputs = reader.read(input_context)
    per_core_example_batch_size = (
        input_context.get_per_replica_batch_size(self._params.global_batch_size)
        if input_context else self._params.global_batch_size)
    batched_inputs = transformed_inputs.unbatch().batch(
        per_core_example_batch_size, self._params.drop_remainder)
    return batched_inputs.prefetch(tf.data.experimental.AUTOTUNE)
