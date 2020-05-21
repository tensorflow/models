# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Run ALBERT on SQuAD 1.1 and SQuAD 2.0 using sentence piece tokenization.

The file is forked from:

https://github.com/google-research/ALBERT/blob/master/run_squad_sp.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import os
from absl import logging
import numpy as np
import tensorflow as tf

from official.nlp.bert import tokenization


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


def read_squad_examples(input_file, is_training, version_2_with_negative):
  """Read a SQuAD json file into a list of SquadExample."""
  del version_2_with_negative
  with tf.io.gfile.GFile(input_file, "r") as reader:
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


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 output_fn,
                                 do_lower_case,
                                 batch_size=None):
  """Loads a data file into a list of `InputBatch`s."""
  cnt_pos, cnt_neg = 0, 0
  base_id = 1000000000
  unique_id = base_id
  max_n, max_m = 1024, 1024
  f = np.zeros((max_n, max_m), dtype=np.float32)

  for (example_index, example) in enumerate(examples):

    if example_index % 100 == 0:
      logging.info("Converting %d/%d pos %d neg %d", example_index,
                   len(examples), cnt_pos, cnt_neg)

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
            example.paragraph_text, lower=do_lower_case))

    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    for i, token in enumerate(para_tokens):
      new_token = token.replace(tokenization.SPIECE_UNDERLINE, " ")
      chartok_to_tok_index.extend([i] * len(new_token))
      tok_start_to_chartok_index.append(char_cnt)
      char_cnt += len(new_token)
      tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = "".join(para_tokens).replace(tokenization.SPIECE_UNDERLINE,
                                                " ")
    n, m = len(paragraph_text), len(tok_cat_text)

    if n > max_n or m > max_m:
      max_n = max(n, max_n)
      max_m = max(m, max_m)
      f = np.zeros((max_n, max_m), dtype=np.float32)

    g = {}
    # pylint: disable=cell-var-from-loop
    def _lcs_match(max_dist, n=n, m=m):
      """Longest-common-substring algorithm."""
      f.fill(0)
      g.clear()

      ### longest common sub sequence
      # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(n):

        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          if j >= m or j < 0:
            continue

          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
          if (tokenization.preprocess_text(
              paragraph_text[i], lower=do_lower_case,
              remove_space=False) == tok_cat_text[j] and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1
    # pylint: enable=cell-var-from-loop

    max_dist = abs(n - m) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[n - 1, m - 1] > 0.8 * n:
        break
      max_dist *= 2

    orig_to_chartok_index = [None] * n
    chartok_to_orig_index = [None] * m
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
      if (i, j) not in g:
        break
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
      logging.info("MISMATCH DETECTED!")
      continue

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(
          chartok_to_orig_index, start_chartok_pos, n, is_start=True)
      end_orig_pos = _convert_index(
          chartok_to_orig_index, end_chartok_pos, n, is_start=False)

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

      start_chartok_pos = _convert_index(
          orig_to_chartok_index, start_position, is_start=True)
      tok_start_position = chartok_to_tok_index[start_chartok_pos]

      end_chartok_pos = _convert_index(
          orig_to_chartok_index, end_position, is_start=False)
      tok_end_position = chartok_to_tok_index[end_chartok_pos]
      assert tok_start_position <= tok_end_position

    def _piece_to_id(x):
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

      cur_tok_start_to_orig_index = []
      cur_tok_end_to_orig_index = []

      tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
      segment_ids.append(0)

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
      tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
      segment_ids.append(1)

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
        logging.info("*** Example ***")
        logging.info("unique_id: %s", (unique_id))
        logging.info("example_index: %s", (example_index))
        logging.info("doc_span_index: %s", (doc_span_index))
        logging.info("tok_start_to_orig_index: %s",
                     " ".join([str(x) for x in cur_tok_start_to_orig_index]))
        logging.info("tok_end_to_orig_index: %s",
                     " ".join([str(x) for x in cur_tok_end_to_orig_index]))
        logging.info(
            "token_is_max_context: %s", " ".join(
                ["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
        logging.info(
            "input_pieces: %s",
            " ".join([tokenizer.sp_model.IdToPiece(x) for x in tokens]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

        if is_training and span_is_impossible:
          logging.info("impossible example span")

        if is_training and not span_is_impossible:
          pieces = [
              tokenizer.sp_model.IdToPiece(token)
              for token in tokens[start_position:(end_position + 1)]
          ]
          answer_text = tokenizer.sp_model.DecodePieces(pieces)
          logging.info("start_position: %d", (start_position))
          logging.info("end_position: %d", (end_position))
          logging.info("answer: %s", (tokenization.printable_text(answer_text)))

          # With multi processing, the example_index is actually the index
          # within the current process therefore we use example_index=None
          # to avoid being used in the future.
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
          is_impossible=span_is_impossible)

      # Run callback
      if is_training:
        output_fn(feature)
      else:
        output_fn(feature, is_padding=False)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1

  if not is_training and feature:
    assert batch_size
    num_padding = 0
    num_examples = unique_id - base_id
    if unique_id % batch_size != 0:
      num_padding = batch_size - (num_examples % batch_size)
    dummy_feature = copy.deepcopy(feature)
    for _ in range(num_padding):
      dummy_feature.unique_id = unique_id

      # Run callback
      output_fn(feature, is_padding=True)
      unique_id += 1

  logging.info("Total number of instances: %d = pos %d neg %d",
               cnt_pos + cnt_neg, cnt_pos, cnt_neg)
  return unique_id - base_id


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


def write_predictions(all_examples,
                      all_features,
                      all_results,
                      n_best_size,
                      max_answer_length,
                      do_lower_case,
                      output_prediction_file,
                      output_nbest_file,
                      output_null_log_odds_file,
                      version_2_with_negative=False,
                      null_score_diff_threshold=0.0,
                      verbose=False):
  """Write final predictions to the json file and log-odds of null if needed."""
  logging.info("Writing predictions to: %s", (output_prediction_file))
  logging.info("Writing nbest to: %s", (output_nbest_file))

  all_predictions, all_nbest_json, scores_diff_json = (
      postprocess_output(all_examples=all_examples,
                         all_features=all_features,
                         all_results=all_results,
                         n_best_size=n_best_size,
                         max_answer_length=max_answer_length,
                         do_lower_case=do_lower_case,
                         version_2_with_negative=version_2_with_negative,
                         null_score_diff_threshold=null_score_diff_threshold,
                         verbose=verbose))

  write_to_json_files(all_predictions, output_prediction_file)
  write_to_json_files(all_nbest_json, output_nbest_file)
  if version_2_with_negative:
    write_to_json_files(scores_diff_json, output_null_log_odds_file)


def postprocess_output(all_examples,
                       all_features,
                       all_results,
                       n_best_size,
                       max_answer_length,
                       do_lower_case,
                       version_2_with_negative=False,
                       null_score_diff_threshold=0.0,
                       verbose=False):
  """Postprocess model output, to form predicton results."""

  del do_lower_case, verbose

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

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
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
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
          # if start_index not in feature.tok_start_to_orig_index:
          #   continue
          # if end_index not in feature.tok_end_to_orig_index:
          #   continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index - doc_offset,
                  end_index=end_index - doc_offset,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=-1,
              end_index=-1,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

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
        final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      assert best_non_null_entry is not None
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  return all_predictions, all_nbest_json, scores_diff_json


def write_to_json_files(json_records, json_file):
  with tf.io.gfile.GFile(json_file, "w") as writer:
    writer.write(json.dumps(json_records, indent=4) + "\n")


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
    tf.io.gfile.makedirs(os.path.dirname(filename))
    self._writer = tf.io.TFRecordWriter(filename)

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


def generate_tf_record_from_json_file(input_file_path,
                                      sp_model_file,
                                      output_path,
                                      max_seq_length=384,
                                      do_lower_case=True,
                                      max_query_length=64,
                                      doc_stride=128,
                                      version_2_with_negative=False):
  """Generates and saves training data into a tf record file."""
  train_examples = read_squad_examples(
      input_file=input_file_path,
      is_training=True,
      version_2_with_negative=version_2_with_negative)
  tokenizer = tokenization.FullSentencePieceTokenizer(
      sp_model_file=sp_model_file)
  train_writer = FeatureWriter(filename=output_path, is_training=True)
  number_of_examples = convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature,
      do_lower_case=do_lower_case)
  train_writer.close()

  meta_data = {
      "task_type": "bert_squad",
      "train_data_size": number_of_examples,
      "max_seq_length": max_seq_length,
      "max_query_length": max_query_length,
      "doc_stride": doc_stride,
      "version_2_with_negative": version_2_with_negative,
  }

  return meta_data
