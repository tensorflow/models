# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3
"""Utility functions for SQuAD v1.1/v2.0 datasets."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import json
import math
import re
import string
import sys
import modeling
import optimization
import tokenization
import numpy as np
import six
from six.moves import map
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import tpu as contrib_tpu

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "start_log_prob",
                                    "end_log_prob"])

RawResultV2 = collections.namedtuple(
    "RawResultV2",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               paragraph_len,
               p_mask=None,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index
    self.tok_end_to_orig_index = tok_end_to_orig_index
    self.token_is_max_context = token_is_max_context
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible
    self.p_mask = p_mask


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        orig_answer_text = None
        is_impossible = False

        if is_training:
          is_impossible = qa.get("is_impossible", False)
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            start_position = answer["answer_start"]
          else:
            start_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples


def _convert_index(index, pos, m=None, is_start=True):
  """Converts index."""
  if index[pos] is not None:
    return index[pos]
  n = len(index)
  rear = pos
  while rear < n - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if m is not None and index[front] < m - 1:
      if is_start:
        return index[front] + 1
      else:
        return m - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, do_lower_case):
  """Loads a data file into a list of `InputBatch`s."""

  cnt_pos, cnt_neg = 0, 0
  unique_id = 1000000000
  max_n, max_m = 1024, 1024
  f = np.zeros((max_n, max_m), dtype=np.float32)

  for (example_index, example) in enumerate(examples):

    if example_index % 100 == 0:
      tf.logging.info("Converting {}/{} pos {} neg {}".format(
          example_index, len(examples), cnt_pos, cnt_neg))

    query_tokens = tokenization.encode_ids(
        tokenizer.sp_model,
        tokenization.preprocess_text(
            example.question_text, lower=do_lower_case))

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    paragraph_text = example.paragraph_text
    para_tokens = tokenization.encode_pieces(
        tokenizer.sp_model,
        tokenization.preprocess_text(
            example.paragraph_text, lower=do_lower_case),
        return_unicode=False)

    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
    for i, token in enumerate(para_tokens):
      new_token = six.ensure_text(token).replace(
          tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
      chartok_to_tok_index.extend([i] * len(new_token))
      tok_start_to_chartok_index.append(char_cnt)
      char_cnt += len(new_token)
      tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = "".join(para_tokens).replace(
        tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
    n, m = len(paragraph_text), len(tok_cat_text)

    if n > max_n or m > max_m:
      max_n = max(n, max_n)
      max_m = max(m, max_m)
      f = np.zeros((max_n, max_m), dtype=np.float32)

    g = {}

    def _lcs_match(max_dist, n=n, m=m):
      """Longest-common-substring algorithm."""
      f.fill(0)
      g.clear()

      ### longest common sub sequence
      # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(n):

        # note(zhiliny):
        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          if j >= m or j < 0: continue

          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
          if (tokenization.preprocess_text(
              paragraph_text[i], lower=do_lower_case,
              remove_space=False) == tok_cat_text[j]
              and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1

    max_dist = abs(n - m) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[n - 1, m - 1] > 0.8 * n: break
      max_dist *= 2

    orig_to_chartok_index = [None] * n
    chartok_to_orig_index = [None] * m
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
      if (i, j) not in g: break
      if g[(i, j)] == 2:
        orig_to_chartok_index[i] = j
        chartok_to_orig_index[j] = i
        i, j = i - 1, j - 1
      elif g[(i, j)] == 1:
        j = j - 1
      else:
        i = i - 1

    if (all(v is None for v in orig_to_chartok_index) or
        f[n - 1, m - 1] < 0.8 * n):
      tf.logging.info("MISMATCH DETECTED!")
      continue

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                      n, is_start=True)
      end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                    n, is_start=False)

      tok_start_to_orig_index.append(start_orig_pos)
      tok_end_to_orig_index.append(end_orig_pos)

    if not is_training:
      tok_start_position = tok_end_position = None

    if is_training and example.is_impossible:
      tok_start_position = 0
      tok_end_position = 0

    if is_training and not example.is_impossible:
      start_position = example.start_position
      end_position = start_position + len(example.orig_answer_text) - 1

      start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                         is_start=True)
      tok_start_position = chartok_to_tok_index[start_chartok_pos]

      end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                       is_start=False)
      tok_end_position = chartok_to_tok_index[end_chartok_pos]
      assert tok_start_position <= tok_end_position

    def _piece_to_id(x):
      if six.PY2 and isinstance(x, six.text_type):
        x = six.ensure_binary(x, "utf-8")
      return tokenizer.sp_model.PieceToId(x)

    all_doc_tokens = list(map(_piece_to_id, para_tokens))

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_is_max_context = {}
      segment_ids = []
      p_mask = []

      cur_tok_start_to_orig_index = []
      cur_tok_end_to_orig_index = []

      tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
      segment_ids.append(0)
      p_mask.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
        p_mask.append(1)
      tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
      segment_ids.append(0)
      p_mask.append(1)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i

        cur_tok_start_to_orig_index.append(
            tok_start_to_orig_index[split_token_index])
        cur_tok_end_to_orig_index.append(
            tok_end_to_orig_index[split_token_index])

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
        p_mask.append(0)
      tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
      segment_ids.append(1)
      p_mask.append(1)

      paragraph_len = len(tokens)
      input_ids = tokens

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      span_is_impossible = example.is_impossible
      start_position = None
      end_position = None
      if is_training and not span_is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          # continue
          start_position = 0
          end_position = 0
          span_is_impossible = True
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = 0
        end_position = 0

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tok_start_to_orig_index: %s" % " ".join(
            [str(x) for x in cur_tok_start_to_orig_index]))
        tf.logging.info("tok_end_to_orig_index: %s" % " ".join(
            [str(x) for x in cur_tok_end_to_orig_index]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_pieces: %s" % " ".join(
            [tokenizer.sp_model.IdToPiece(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        if is_training and span_is_impossible:
          tf.logging.info("impossible example span")

        if is_training and not span_is_impossible:
          pieces = [tokenizer.sp_model.IdToPiece(token) for token in
                    tokens[start_position: (end_position + 1)]]
          answer_text = tokenizer.sp_model.DecodePieces(pieces)
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

          # note(zhiliny): With multi processing,
          # the example_index is actually the index within the current process
          # therefore we use example_index=None to avoid being used in the future.
          # The current code does not use example_index of training data.
      if is_training:
        feat_example_index = None
      else:
        feat_example_index = example_index

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=feat_example_index,
          doc_span_index=doc_span_index,
          tok_start_to_orig_index=cur_tok_start_to_orig_index,
          tok_end_to_orig_index=cur_tok_end_to_orig_index,
          token_is_max_context=token_is_max_context,
          tokens=[tokenizer.sp_model.IdToPiece(x) for x in tokens],
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          paragraph_len=paragraph_len,
          start_position=start_position,
          end_position=end_position,
          is_impossible=span_is_impossible,
          p_mask=p_mask)

      # Run callback
      output_fn(feature)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1

  tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
      cnt_pos + cnt_neg, cnt_pos, cnt_neg))


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["p_mask"] = create_int_feature(feature.p_mask)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def input_fn_builder(input_file, seq_length, is_training,
                     drop_remainder, use_tpu, bsz, is_v2):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }
  # p_mask is not required for SQuAD v1.1
  if is_v2:
    name_to_features["p_mask"] = tf.FixedLenFeature([seq_length], tf.int64)

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_v1_model(albert_config, is_training, input_ids, input_mask,
                    segment_ids, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def v1_model_fn_builder(albert_config, init_checkpoint, learning_rate,
                        num_train_steps, num_warmup_steps, use_tpu,
                        use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_v1_model(
        albert_config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_log_prob": start_logits,
          "end_log_prob": end_logits,
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))
    return output_spec

  return model_fn


def accumulate_predictions_v1(result_dict, all_examples, all_features,
                              all_results, n_best_size, max_answer_length):
  """accumulate predictions for each positions in a dictionary."""
  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    if example_index not in result_dict:
      result_dict[example_index] = {}
    features = example_index_to_features[example_index]

    prelim_predictions = []
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      if feature.unique_id not in result_dict[example_index]:
        result_dict[example_index][feature.unique_id] = {}
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_log_prob, n_best_size)
      end_indexes = _get_best_indexes(result.end_log_prob, n_best_size)
      for start_index in start_indexes:
        for end_index in end_indexes:
          doc_offset = feature.tokens.index("[SEP]") + 1
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
            continue
          if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          start_log_prob = result.start_log_prob[start_index]
          end_log_prob = result.end_log_prob[end_index]
          start_idx = start_index - doc_offset
          end_idx = end_index - doc_offset
          if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
            result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
          result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append((start_log_prob, end_log_prob))


def write_predictions_v1(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file, output_nbest_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      for ((start_idx, end_idx), logprobs) in \
        result_dict[example_index][feature.unique_id].items():
        start_log_prob = 0
        end_log_prob = 0
        for logprob in logprobs:
          start_log_prob += logprob[0]
          end_log_prob += logprob[1]
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index=start_idx,
                end_index=end_idx,
                start_log_prob=start_log_prob / len(logprobs),
                end_log_prob=end_log_prob / len(logprobs)))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index >= 0:  # this is a non-null prediction
        tok_start_to_orig_index = feature.tok_start_to_orig_index
        tok_end_to_orig_index = feature.tok_end_to_orig_index
        start_orig_pos = tok_start_to_orig_index[pred.start_index]
        end_orig_pos = tok_end_to_orig_index[pred.end_index]

        paragraph_text = example.paragraph_text
        final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_log_prob=0.0, end_log_prob=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    all_predictions[example.qas_id] = nbest_json[0]["text"]
    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  return all_predictions


####### following are from official SQuAD v1.1 evaluation scripts
def normalize_answer_v1(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer_v1(prediction).split()
  ground_truth_tokens = normalize_answer_v1(ground_truth).split()
  common = (
      collections.Counter(prediction_tokens)
      & collections.Counter(ground_truth_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def exact_match_score(prediction, ground_truth):
  return (normalize_answer_v1(prediction) == normalize_answer_v1(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def evaluate_v1(dataset, predictions):
  f1 = exact_match = total = 0
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph["qas"]:
        total += 1
        if qa["id"] not in predictions:
          message = ("Unanswered question " + six.ensure_str(qa["id"]) +
                     "  will receive score 0.")
          print(message, file=sys.stderr)
          continue
        ground_truths = [x["text"] for x in qa["answers"]]
        # ground_truths = list(map(lambda x: x["text"], qa["answers"]))
        prediction = predictions[qa["id"]]
        exact_match += metric_max_over_ground_truths(exact_match_score,
                                                     prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return {"exact_match": exact_match, "f1": f1}

####### above are from official SQuAD v1.1 evaluation scripts
####### following are from official SQuAD v2.0 evaluation scripts
def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def normalize_answer_v2(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer_v2(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer_v2(a_gold) == normalize_answer_v2(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer_v2(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh


def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

####### above are from official SQuAD v2.0 evaluation scripts

def accumulate_predictions_v2(result_dict, cls_dict, all_examples,
                              all_features, all_results, n_best_size,
                              max_answer_length, start_n_top, end_n_top):
  """accumulate predictions for each positions in a dictionary."""

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    if example_index not in result_dict:
      result_dict[example_index] = {}
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive

    for (feature_index, feature) in enumerate(features):
      if feature.unique_id not in result_dict[example_index]:
        result_dict[example_index][feature.unique_id] = {}
      result = unique_id_to_result[feature.unique_id]
      cur_null_score = result.cls_logits

      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, cur_null_score)

      doc_offset = feature.tokens.index("[SEP]") + 1
      for i in range(start_n_top):
        for j in range(end_n_top):
          start_log_prob = result.start_top_log_probs[i]
          start_index = result.start_top_index[i]

          j_index = i * end_n_top + j

          end_log_prob = result.end_top_log_probs[j_index]
          end_index = result.end_top_index[j_index]
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
            continue
          if start_index - doc_offset < 0:
            continue
          if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          start_idx = start_index - doc_offset
          end_idx = end_index - doc_offset
          if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
            result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
          result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append((start_log_prob, end_log_prob))
    if example_index not in cls_dict:
      cls_dict[example_index] = []
    cls_dict[example_index].append(score_null)


def write_predictions_v2(result_dict, cls_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file,
                         output_nbest_file, output_null_log_odds_file,
                         null_score_diff_threshold):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    # score_null = 1000000  # large and positive

    for (feature_index, feature) in enumerate(features):
      for ((start_idx, end_idx), logprobs) in \
        result_dict[example_index][feature.unique_id].items():
        start_log_prob = 0
        end_log_prob = 0
        for logprob in logprobs:
          start_log_prob += logprob[0]
          end_log_prob += logprob[1]
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index=start_idx,
                end_index=end_idx,
                start_log_prob=start_log_prob / len(logprobs),
                end_log_prob=end_log_prob / len(logprobs)))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      tok_start_to_orig_index = feature.tok_start_to_orig_index
      tok_end_to_orig_index = feature.tok_end_to_orig_index
      start_orig_pos = tok_start_to_orig_index[pred.start_index]
      end_orig_pos = tok_end_to_orig_index[pred.end_index]

      paragraph_text = example.paragraph_text
      final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(
              text="",
              start_log_prob=-1e6,
              end_log_prob=-1e6))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None

    score_diff = sum(cls_dict[example_index]) / len(cls_dict[example_index])
    scores_diff_json[example.qas_id] = score_diff
    # predict null answers when null threshold is provided
    if null_score_diff_threshold is None or score_diff < null_score_diff_threshold:
      all_predictions[example.qas_id] = best_non_null_entry.text
    else:
      all_predictions[example.qas_id] = ""

    all_nbest_json[example.qas_id] = nbest_json
    assert len(nbest_json) >= 1

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
  return all_predictions, scores_diff_json


def create_v2_model(albert_config, is_training, input_ids, input_mask,
                    segment_ids, use_one_hot_embeddings, features,
                    max_seq_length, start_n_top, end_n_top, dropout_prob):
  """Creates a classification model."""
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output = model.get_sequence_output()
  bsz = tf.shape(output)[0]
  return_dict = {}
  output = tf.transpose(output, [1, 0, 2])

  # invalid position mask such as query and special symbols (PAD, SEP, CLS)
  p_mask = tf.cast(features["p_mask"], dtype=tf.float32)

  # logit of the start position
  with tf.variable_scope("start_logits"):
    start_logits = tf.layers.dense(
        output,
        1,
        kernel_initializer=modeling.create_initializer(
            albert_config.initializer_range))
    start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
    start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
    start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

  # logit of the end position
  with tf.variable_scope("end_logits"):
    if is_training:
      # during training, compute the end logits based on the
      # ground truth of the start position
      start_positions = tf.reshape(features["start_positions"], [-1])
      start_index = tf.one_hot(start_positions, depth=max_seq_length, axis=-1,
                               dtype=tf.float32)
      start_features = tf.einsum("lbh,bl->bh", output, start_index)
      start_features = tf.tile(start_features[None], [max_seq_length, 1, 1])
      end_logits = tf.layers.dense(
          tf.concat([output, start_features], axis=-1),
          albert_config.hidden_size,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          activation=tf.tanh,
          name="dense_0")
      end_logits = contrib_layers.layer_norm(end_logits, begin_norm_axis=-1)

      end_logits = tf.layers.dense(
          end_logits,
          1,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          name="dense_1")
      end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
      end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
    else:
      # during inference, compute the end logits based on beam search

      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=start_n_top)
      start_index = tf.one_hot(start_top_index,
                               depth=max_seq_length, axis=-1, dtype=tf.float32)
      start_features = tf.einsum("lbh,bkl->bkh", output, start_index)
      end_input = tf.tile(output[:, :, None],
                          [1, 1, start_n_top, 1])
      start_features = tf.tile(start_features[None],
                               [max_seq_length, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = tf.layers.dense(
          end_input,
          albert_config.hidden_size,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          activation=tf.tanh,
          name="dense_0")
      end_logits = contrib_layers.layer_norm(end_logits, begin_norm_axis=-1)
      end_logits = tf.layers.dense(
          end_logits,
          1,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          name="dense_1")
      end_logits = tf.reshape(end_logits, [max_seq_length, -1, start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (
          1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=end_n_top)
      end_top_log_probs = tf.reshape(
          end_top_log_probs,
          [-1, start_n_top * end_n_top])
      end_top_index = tf.reshape(
          end_top_index,
          [-1, start_n_top * end_n_top])

  if is_training:
    return_dict["start_log_probs"] = start_log_probs
    return_dict["end_log_probs"] = end_log_probs
  else:
    return_dict["start_top_log_probs"] = start_top_log_probs
    return_dict["start_top_index"] = start_top_index
    return_dict["end_top_log_probs"] = end_top_log_probs
    return_dict["end_top_index"] = end_top_index

  # an additional layer to predict answerability
  with tf.variable_scope("answer_class"):
    # get the representation of CLS
    cls_index = tf.one_hot(tf.zeros([bsz], dtype=tf.int32),
                           max_seq_length,
                           axis=-1, dtype=tf.float32)
    cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)

    # get the representation of START
    start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                            name="softmax_start")
    start_feature = tf.einsum("lbh,bl->bh", output, start_p)

    # note(zhiliny): no dependency on end_feature so that we can obtain
    # one single `cls_logits` for each sample
    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = tf.layers.dense(
        ans_feature,
        albert_config.hidden_size,
        activation=tf.tanh,
        kernel_initializer=modeling.create_initializer(
            albert_config.initializer_range),
        name="dense_0")
    ans_feature = tf.layers.dropout(ans_feature, dropout_prob,
                                    training=is_training)
    cls_logits = tf.layers.dense(
        ans_feature,
        1,
        kernel_initializer=modeling.create_initializer(
            albert_config.initializer_range),
        name="dense_1",
        use_bias=False)
    cls_logits = tf.squeeze(cls_logits, -1)

    return_dict["cls_logits"] = cls_logits

  return return_dict


def v2_model_fn_builder(albert_config, init_checkpoint, learning_rate,
                        num_train_steps, num_warmup_steps, use_tpu,
                        use_one_hot_embeddings, max_seq_length, start_n_top,
                        end_n_top, dropout_prob):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    # unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    outputs = create_v2_model(
        albert_config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        features=features,
        max_seq_length=max_seq_length,
        start_n_top=start_n_top,
        end_n_top=end_n_top,
        dropout_prob=dropout_prob)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(log_probs, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)

        loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss

      start_loss = compute_loss(
          outputs["start_log_probs"], features["start_positions"])
      end_loss = compute_loss(
          outputs["end_log_probs"], features["end_positions"])

      total_loss = (start_loss + end_loss) * 0.5

      cls_logits = outputs["cls_logits"]
      is_impossible = tf.reshape(features["is_impossible"], [-1])
      regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(is_impossible, dtype=tf.float32), logits=cls_logits)
      regression_loss = tf.reduce_mean(regression_loss)

      # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
      # comparable to start_loss and end_loss
      total_loss += regression_loss * 0.5
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": features["unique_ids"],
          "start_top_index": outputs["start_top_index"],
          "start_top_log_probs": outputs["start_top_log_probs"],
          "end_top_index": outputs["end_top_index"],
          "end_top_log_probs": outputs["end_top_log_probs"],
          "cls_logits": outputs["cls_logits"]
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def evaluate_v2(result_dict, cls_dict, prediction_json, eval_examples,
                eval_features, all_results, n_best_size, max_answer_length,
                output_prediction_file, output_nbest_file,
                output_null_log_odds_file):
  null_score_diff_threshold = None
  predictions, na_probs = write_predictions_v2(
      result_dict, cls_dict, eval_examples, eval_features,
      all_results, n_best_size, max_answer_length,
      output_prediction_file, output_nbest_file,
      output_null_log_odds_file, null_score_diff_threshold)

  na_prob_thresh = 1.0  # default value taken from the eval script
  qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
  exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                        na_prob_thresh)
  f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                     na_prob_thresh)
  out_eval = make_eval_dict(exact_thresh, f1_thresh)
  find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, na_probs, qid_to_has_ans)
  null_score_diff_threshold = out_eval["best_f1_thresh"]

  predictions, na_probs = write_predictions_v2(
      result_dict, cls_dict,eval_examples, eval_features,
      all_results, n_best_size, max_answer_length,
      output_prediction_file, output_nbest_file,
      output_null_log_odds_file, null_score_diff_threshold)

  qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
  exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                        na_prob_thresh)
  f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                     na_prob_thresh)
  out_eval = make_eval_dict(exact_thresh, f1_thresh)
  out_eval["null_score_diff_threshold"] = null_score_diff_threshold
  return out_eval
