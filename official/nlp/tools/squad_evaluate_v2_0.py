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

"""Evaluation script for SQuAD version 2.0.

The functions are copied and modified from
https://raw.githubusercontent.com/white127/SQUAD-2.0-bidaf/master/evaluate-v2.0.py

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""

import collections
import re
import string

from absl import logging


def _make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans


def _normalize_answer(s):
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


def _get_tokens(s):
  if not s: return []
  return _normalize_answer(s).split()


def _compute_exact(a_gold, a_pred):
  return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))


def _compute_f1(a_gold, a_pred):
  """Compute F1-score."""
  gold_toks = _get_tokens(a_gold)
  pred_toks = _get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if not gold_toks or not pred_toks:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def _get_raw_scores(dataset, predictions):
  """Compute raw scores."""
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if _normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in predictions:
          logging.error('Missing prediction for %s', qid)
          continue
        a_pred = predictions[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(_compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(_compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores


def _apply_no_ans_threshold(
    scores, na_probs, qid_to_has_ans, na_prob_thresh=1.0):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores


def _make_eval_dict(exact_scores, f1_scores, qid_list=None):
  """Make evaluation result dictionary."""
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


def _merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def _make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans):
  """Make evaluation dictionary containing average recision recall."""
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i+1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  return {'ap': 100.0 * avg_prec}


def _run_precision_recall_analysis(
    main_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  """Run precision recall analysis and return result dictionary."""
  num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
  if num_true_pos == 0:
    return
  pr_exact = _make_precision_recall_eval(
      exact_raw, na_probs, num_true_pos, qid_to_has_ans)
  pr_f1 = _make_precision_recall_eval(
      f1_raw, na_probs, num_true_pos, qid_to_has_ans)
  oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
  pr_oracle = _make_precision_recall_eval(
      oracle_scores, na_probs, num_true_pos, qid_to_has_ans)
  _merge_eval(main_eval, pr_exact, 'pr_exact')
  _merge_eval(main_eval, pr_f1, 'pr_f1')
  _merge_eval(main_eval, pr_oracle, 'pr_oracle')


def _find_best_thresh(predictions, scores, na_probs, qid_to_has_ans):
  """Find the best threshold for no answer probability."""
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for qid in qid_list:
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if predictions[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh


def _find_all_best_thresh(
    main_eval, predictions, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = _find_best_thresh(
      predictions, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = _find_best_thresh(
      predictions, f1_raw, na_probs, qid_to_has_ans)
  main_eval['final_exact'] = best_exact
  main_eval['final_exact_thresh'] = exact_thresh
  main_eval['final_f1'] = best_f1
  main_eval['final_f1_thresh'] = f1_thresh


def evaluate(dataset, predictions, na_probs=None):
  """Evaluate prediction results."""
  new_orig_data = []
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        if qa['id'] in predictions:
          new_para = {'qas': [qa]}
          new_article = {'paragraphs': [new_para]}
          new_orig_data.append(new_article)
  dataset = new_orig_data

  if na_probs is None:
    na_probs = {k: 0.0 for k in predictions}
  qid_to_has_ans = _make_qid_to_has_ans(dataset)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  exact_raw, f1_raw = _get_raw_scores(dataset, predictions)
  exact_thresh = _apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans)
  f1_thresh = _apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans)
  out_eval = _make_eval_dict(exact_thresh, f1_thresh)
  if has_ans_qids:
    has_ans_eval = _make_eval_dict(
        exact_thresh, f1_thresh, qid_list=has_ans_qids)
    _merge_eval(out_eval, has_ans_eval, 'HasAns')
  if no_ans_qids:
    no_ans_eval = _make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
    _merge_eval(out_eval, no_ans_eval, 'NoAns')

  _find_all_best_thresh(
      out_eval, predictions, exact_raw, f1_raw, na_probs, qid_to_has_ans)
  _run_precision_recall_analysis(
      out_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans)
  return out_eval
