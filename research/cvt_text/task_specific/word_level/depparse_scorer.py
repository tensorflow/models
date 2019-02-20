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

"""Dependency parsing evaluation (computes UAS/LAS)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from task_specific.word_level import word_level_scorer


class DepparseScorer(word_level_scorer.WordLevelScorer):
  def __init__(self, n_relations, punctuation):
    super(DepparseScorer, self).__init__()
    self._n_relations = n_relations
    self._punctuation = punctuation if punctuation else None

  def _get_results(self):
    correct_unlabeled, correct_labeled, count = 0, 0, 0
    for example, preds in zip(self._examples, self._preds):
      for w, y_true, y_pred in zip(example.words[1:-1], example.labels, preds):
        if w in self._punctuation:
          continue
        count += 1
        correct_labeled += (1 if y_pred == y_true else 0)
        correct_unlabeled += (1 if int(y_pred // self._n_relations) ==
                              int(y_true // self._n_relations) else 0)
    return [
        ("las", 100.0 * correct_labeled / count),
        ("uas", 100.0 * correct_unlabeled / count),
        ("loss", self.get_loss()),
    ]
