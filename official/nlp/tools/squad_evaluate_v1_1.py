# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation of SQuAD predictions (version 1.1).

The functions are copied from
https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/.

The SQuAD dataset is described in this paper:
SQuAD: 100,000+ Questions for Machine Comprehension of Text
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang
https://nlp.stanford.edu/pubs/rajpurkar2016squad.pdf
"""

import collections
import re
import string

# pylint: disable=g-bad-import-order

from absl import logging
# pylint: enable=g-bad-import-order


def _normalize_answer(s):
  """Lowers text and remove punctuation, articles and extra whitespace."""

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


def _f1_score(prediction, ground_truth):
  """Computes F1 score by comparing prediction to ground truth."""
  prediction_tokens = _normalize_answer(prediction).split()
  ground_truth_tokens = _normalize_answer(ground_truth).split()
  prediction_counter = collections.Counter(prediction_tokens)
  ground_truth_counter = collections.Counter(ground_truth_tokens)
  common = prediction_counter & ground_truth_counter
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def _exact_match_score(prediction, ground_truth):
  """Checks if predicted answer exactly matches ground truth answer."""
  return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  """Computes the max over all metric scores."""
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
  """Evaluates predictions for a dataset."""
  f1 = exact_match = total = 0
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph["qas"]:
        total += 1
        if qa["id"] not in predictions:
          message = "Unanswered question " + qa["id"] + " will receive score 0."
          logging.error(message)
          continue
        ground_truths = [entry["text"] for entry in qa["answers"]]
        prediction = predictions[qa["id"]]
        exact_match += _metric_max_over_ground_truths(_exact_match_score,
                                                      prediction, ground_truths)
        f1 += _metric_max_over_ground_truths(_f1_score, prediction,
                                             ground_truths)

  exact_match = exact_match / total
  f1 = f1 / total

  return {"exact_match": exact_match, "final_f1": f1}
