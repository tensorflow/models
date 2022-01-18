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

# coding=utf-8
"""Utilities used in SQUAD task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gc
import json
import math
import os
import pickle
import re
import string

from absl import logging
import numpy as np
import six
import tensorflow as tf

from official.nlp.xlnet import data_utils
from official.nlp.xlnet import preprocess_utils

SPIECE_UNDERLINE = u"▁"


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               input_ids,
               input_mask,
               p_mask,
               segment_ids,
               paragraph_len,
               cls_index,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index
    self.tok_end_to_orig_index = tok_end_to_orig_index
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.p_mask = p_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.cls_index = cls_index
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        qid_to_has_ans[qa["id"]] = bool(qa["answers"])
  return qid_to_has_ans


def get_raw_scores(dataset, preds):
  """Gets exact scores and f1 scores."""
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article["paragraphs"]:
      for qa in p["qas"]:
        qid = qa["id"]
        gold_answers = [
            a["text"] for a in qa["answers"] if normalize_answer(a["text"])
        ]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = [""]
        if qid not in preds:
          print("Missing prediction for %s" % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def get_tokens(s):
  if not s:
    return []
  return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
  """Computes f1 score."""
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  # pylint: disable=g-explicit-length-test
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  """Finds best threshold."""
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for qid in qid_list:
    if qid not in scores:
      continue
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

  has_ans_score, has_ans_cnt = 0, 0
  for qid in qid_list:
    if not qid_to_has_ans[qid]:
      continue
    has_ans_cnt += 1

    if qid not in scores:
      continue
    has_ans_score += scores[qid]

  return 100.0 * best_score / len(
      scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs,
                         qid_to_has_ans):
  """Finds all best threshold."""
  best_exact, exact_thresh, has_ans_exact = find_best_thresh(
      preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh, has_ans_f1 = find_best_thresh(preds, f1_raw, na_probs,
                                                    qid_to_has_ans)
  main_eval["best_exact"] = best_exact
  main_eval["best_exact_thresh"] = exact_thresh
  main_eval["best_f1"] = best_f1
  main_eval["best_f1_thresh"] = f1_thresh
  main_eval["has_ans_exact"] = has_ans_exact
  main_eval["has_ans_f1"] = has_ans_f1


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction", [
        "feature_index", "start_index", "end_index", "start_log_prob",
        "end_log_prob"
    ])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])
RawResult = collections.namedtuple("RawResult", [
    "unique_id", "start_top_log_probs", "start_top_index", "end_top_log_probs",
    "end_top_index", "cls_logits"
])


def _compute_softmax(scores):
  """Computes softmax probability over raw logits."""
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
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (preprocess_utils.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        preprocess_utils.printable_text(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, orig_data,
                      start_n_top, end_n_top):
  """Writes final predictions to the json file and log-odds of null if needed."""
  logging.info("Writing predictions to: %s", (output_prediction_file))

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

    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]

      cur_null_score = result.cls_logits

      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, cur_null_score)

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
          if start_index >= feature.paragraph_len - 1:
            continue
          if end_index >= feature.paragraph_len - 1:
            continue

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
                  start_index=start_index,
                  end_index=end_index,
                  start_log_prob=start_log_prob,
                  end_log_prob=end_log_prob))

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
      final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()

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
          _NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

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

    score_diff = score_null
    scores_diff_json[example.qas_id] = score_diff

    all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

  qid_to_has_ans = make_qid_to_has_ans(orig_data)
  exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
  out_eval = {}

  find_all_best_thresh(out_eval, all_predictions, exact_raw, f1_raw,
                       scores_diff_json, qid_to_has_ans)

  return out_eval


def read_squad_examples(input_file, is_training):
  """Reads a SQuAD json file into a list of SquadExample."""
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
          is_impossible = qa["is_impossible"]
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


# pylint: disable=invalid-name
def _convert_index(index, pos, M=None, is_start=True):
  """Converts index."""
  if index[pos] is not None:
    return index[pos]
  N = len(index)
  rear = pos
  while rear < N - 1 and index[rear] is None:
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
    if M is not None and index[front] < M - 1:
      if is_start:
        return index[front] + 1
      else:
        return M - 1
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


def convert_examples_to_features(examples, sp_model, max_seq_length, doc_stride,
                                 max_query_length, is_training, output_fn,
                                 uncased):
  """Loads a data file into a list of `InputBatch`s."""

  cnt_pos, cnt_neg = 0, 0
  unique_id = 1000000000
  max_N, max_M = 1024, 1024
  f = np.zeros((max_N, max_M), dtype=np.float32)

  for (example_index, example) in enumerate(examples):
    # pylint: disable=logging-format-interpolation
    if example_index % 100 == 0:
      logging.info("Converting {}/{} pos {} neg {}".format(
          example_index, len(examples), cnt_pos, cnt_neg))

    query_tokens = preprocess_utils.encode_ids(
        sp_model,
        preprocess_utils.preprocess_text(example.question_text, lower=uncased))

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    paragraph_text = example.paragraph_text
    para_tokens = preprocess_utils.encode_pieces(
        sp_model,
        preprocess_utils.preprocess_text(example.paragraph_text, lower=uncased))

    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    for i, token in enumerate(para_tokens):
      chartok_to_tok_index.extend([i] * len(token))
      tok_start_to_chartok_index.append(char_cnt)
      char_cnt += len(token)
      tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = "".join(para_tokens).replace(SPIECE_UNDERLINE, " ")
    N, M = len(paragraph_text), len(tok_cat_text)

    if N > max_N or M > max_M:
      max_N = max(N, max_N)
      max_M = max(M, max_M)
      f = np.zeros((max_N, max_M), dtype=np.float32)
      gc.collect()

    g = {}

    # pylint: disable=cell-var-from-loop
    def _lcs_match(max_dist):
      """LCS match."""
      f.fill(0)
      g.clear()

      ### longest common sub sequence
      # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(N):

        # note(zhiliny):
        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          if j >= M or j < 0:
            continue

          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
          if (preprocess_utils.preprocess_text(
              paragraph_text[i], lower=uncased,
              remove_space=False) == tok_cat_text[j] and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1

    max_dist = abs(N - M) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[N - 1, M - 1] > 0.8 * N:
        break
      max_dist *= 2

    orig_to_chartok_index = [None] * N
    chartok_to_orig_index = [None] * M
    i, j = N - 1, M - 1
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

    if all(
        v is None for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
      print("MISMATCH DETECTED!")
      continue

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(
          chartok_to_orig_index, start_chartok_pos, N, is_start=True)
      end_orig_pos = _convert_index(
          chartok_to_orig_index, end_chartok_pos, N, is_start=False)

      tok_start_to_orig_index.append(start_orig_pos)
      tok_end_to_orig_index.append(end_orig_pos)

    if not is_training:
      tok_start_position = tok_end_position = None

    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1

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
      if six.PY2 and isinstance(x, unicode):  # pylint: disable=undefined-variable
        x = x.encode("utf-8")
      return sp_model.PieceToId(x)

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
        segment_ids.append(data_utils.SEG_ID_P)
        p_mask.append(0)

      paragraph_len = len(tokens)

      tokens.append(data_utils.SEP_ID)
      segment_ids.append(data_utils.SEG_ID_P)
      p_mask.append(1)

      # note(zhiliny): we put P before Q
      # because during pretraining, B is always shorter than A
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(data_utils.SEG_ID_Q)
        p_mask.append(1)
      tokens.append(data_utils.SEP_ID)
      segment_ids.append(data_utils.SEG_ID_Q)
      p_mask.append(1)

      cls_index = len(segment_ids)
      tokens.append(data_utils.CLS_ID)
      segment_ids.append(data_utils.SEG_ID_CLS)
      p_mask.append(0)

      input_ids = tokens

      # The mask has 0 for real tokens and 1 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [0] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(1)
        segment_ids.append(data_utils.SEG_ID_PAD)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(p_mask) == max_seq_length

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
          # note: we put P before Q, so doc_offset should be zero.
          # doc_offset = len(query_tokens) + 2
          doc_offset = 0
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = cls_index
        end_position = cls_index

      if example_index < 20:
        logging.info("*** Example ***")
        logging.info("unique_id: %s", unique_id)
        logging.info("example_index: %s", example_index)
        logging.info("doc_span_index: %s", doc_span_index)
        logging.info("tok_start_to_orig_index: %s",
                     " ".join([str(x) for x in cur_tok_start_to_orig_index]))
        logging.info("tok_end_to_orig_index: %s",
                     " ".join([str(x) for x in cur_tok_end_to_orig_index]))
        logging.info(
            "token_is_max_context: %s", " ".join([
                "%d:%s" % (x, y)
                for (x, y) in six.iteritems(token_is_max_context)
            ]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

        if is_training and span_is_impossible:
          logging.info("impossible example span")

        if is_training and not span_is_impossible:
          pieces = [
              sp_model.IdToPiece(token)
              for token in tokens[start_position:(end_position + 1)]
          ]
          answer_text = sp_model.DecodePieces(pieces)
          logging.info("start_position: %d", start_position)
          logging.info("end_position: %d", end_position)
          logging.info("answer: %s",
                       preprocess_utils.printable_text(answer_text))

          # With multi processing, the example_index is actually the index
          # within the current process therefore we use example_index=None to
          # avoid being used in the future. # The current code does not use
          # example_index of training data.
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
          input_ids=input_ids,
          input_mask=input_mask,
          p_mask=p_mask,
          segment_ids=segment_ids,
          paragraph_len=paragraph_len,
          cls_index=cls_index,
          start_position=start_position,
          end_position=end_position,
          is_impossible=span_is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1

  logging.info("Total number of instances: %d = pos %d + neg %d",
               cnt_pos + cnt_neg, cnt_pos, cnt_neg)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the "max context" doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word "bought" will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for "bought" would be span C since
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


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["p_mask"] = create_float_feature(feature.p_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    features["cls_index"] = create_int_feature([feature.cls_index])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_float_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def create_eval_data(spm_basename,
                     sp_model,
                     eval_examples,
                     max_seq_length,
                     max_query_length,
                     doc_stride,
                     uncased,
                     output_dir=None):
  """Creates evaluation tfrecords."""
  eval_features = []
  eval_writer = None
  if output_dir:
    eval_rec_file = os.path.join(
        output_dir,
        "{}.slen-{}.qlen-{}.eval.tf_record".format(spm_basename, max_seq_length,
                                                   max_query_length))
    eval_feature_file = os.path.join(
        output_dir,
        "{}.slen-{}.qlen-{}.eval.features.pkl".format(spm_basename,
                                                      max_seq_length,
                                                      max_query_length))

    eval_writer = FeatureWriter(filename=eval_rec_file, is_training=False)

  def append_feature(feature):
    eval_features.append(feature)
    if eval_writer:
      eval_writer.process_feature(feature)

  convert_examples_to_features(
      examples=eval_examples,
      sp_model=sp_model,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False,
      output_fn=append_feature,
      uncased=uncased)

  if eval_writer:
    eval_writer.close()
    with tf.io.gfile.GFile(eval_feature_file, "wb") as fout:
      pickle.dump(eval_features, fout)

  return eval_features
