# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Sequence tagging evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from task_specific.word_level import tagging_utils
from task_specific.word_level import word_level_scorer


class AccuracyScorer(word_level_scorer.WordLevelScorer):
  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def _get_results(self):
    correct, count = 0, 0
    for example, preds in zip(self._examples, self._preds):
      for y_true, y_pred in zip(example.labels, preds):
        count += 1
        correct += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
    return [
        ("accuracy", 100.0 * correct / count),
        ("loss", self.get_loss())
    ]


class F1Scorer(word_level_scorer.WordLevelScorer):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0

  def _get_results(self):
    if self._n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * self._n_correct / self._n_predicted
      r = 100.0 * self._n_correct / self._n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ("precision", p),
        ("recall", r),
        ("f1", f1),
        ("loss", self.get_loss()),
    ]


class EntityLevelF1Scorer(F1Scorer):
  def __init__(self, label_mapping):
    super(EntityLevelF1Scorer, self).__init__()
    self._inv_label_mapping = {v: k for k, v in label_mapping.iteritems()}

  def _get_results(self):
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for example, preds in zip(self._examples, self._preds):
      sent_spans = set(tagging_utils.get_span_labels(
          example.labels, self._inv_label_mapping))
      span_preds = set(tagging_utils.get_span_labels(
          preds, self._inv_label_mapping))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
    return super(EntityLevelF1Scorer, self)._get_results()
