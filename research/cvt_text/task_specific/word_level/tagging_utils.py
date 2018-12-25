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

"""Utilities for sequence tagging tasks for entity-level tasks (e.g., NER)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_span_labels(sentence_tags, inv_label_mapping=None):
  """Go from token-level labels to list of entities (start, end, class)."""

  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, last.split('-')[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        sentence_tags[-1].split('-')[-1]))
  return span_labels


def get_tags(span_labels, length, encoding):
  """Converts a list of entities to token-label labels based on the provided
  encoding (e.g., BIOES).
  """

  tags = ['O' for _ in range(length)]
  for s, e, t in span_labels:
    for i in range(s, e + 1):
      tags[i] = 'I-' + t
    if 'E' in encoding:
      tags[e] = 'E-' + t
    if 'B' in encoding:
      tags[s] = 'B-' + t
    if 'S' in encoding and s == e:
      tags[s] = 'S-' + t
  return tags
