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

"""Utilities for constructing minibatches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np

from base import embeddings


def get_bucket(config, l):
  for i, (s, e) in enumerate(config.buckets):
    if s <= l < e:
      return config.buckets[i]


def build_array(nested_lists, dtype='int32'):
  depth_to_sizes = collections.defaultdict(set)
  _get_sizes(nested_lists, depth_to_sizes)
  shape = [max(depth_to_sizes[depth]) for depth in range(len(depth_to_sizes))]

  copy_depth = len(depth_to_sizes) - 1
  while copy_depth > 0 and len(depth_to_sizes[copy_depth]) == 1:
    copy_depth -= 1

  arr = np.zeros(shape, dtype=dtype)
  _fill_array(nested_lists, arr, copy_depth)

  return arr


def _get_sizes(nested_lists, depth_to_sizes, depth=0):
  depth_to_sizes[depth].add(len(nested_lists))
  first_elem = nested_lists[0]
  if (isinstance(first_elem, collections.Sequence) or
      isinstance(first_elem, np.ndarray)):
    for sublist in nested_lists:
      _get_sizes(sublist, depth_to_sizes, depth + 1)


def _fill_array(nested_lists, arr, copy_depth, depth=0):
  if depth == copy_depth:
    for i in range(len(nested_lists)):
      if isinstance(nested_lists[i], np.ndarray):
        arr[i] = nested_lists[i]
      else:
        arr[i] = np.array(nested_lists[i])
  else:
    for i in range(len(nested_lists)):
      _fill_array(nested_lists[i], arr[i], copy_depth, depth + 1)


class Dataset(object):
  def __init__(self, config, examples, task_name='unlabeled', is_training=False):
    self._config = config
    self.examples = examples
    self.size = len(examples)
    self.task_name = task_name
    self.is_training = is_training

  def get_minibatches(self, minibatch_size):
    by_bucket = collections.defaultdict(list)
    for i, e in enumerate(self.examples):
      by_bucket[get_bucket(self._config, len(e.words))].append(i)

    # save memory by weighting examples so longer sentences have
    # smaller minibatches.
    weight = lambda ind: np.sqrt(len(self.examples[ind].words))
    total_weight = float(sum(weight(i) for i in range(len(self.examples))))
    weight_per_batch = minibatch_size * total_weight / len(self.examples)
    cumulative_weight = 0.0
    id_batches = []
    for _, ids in by_bucket.iteritems():
      ids = np.array(ids)
      np.random.shuffle(ids)
      curr_batch, curr_weight = [], 0.0
      for i, curr_id in enumerate(ids):
        curr_batch.append(curr_id)
        curr_weight += weight(curr_id)
        if (i == len(ids) - 1 or cumulative_weight + curr_weight >=
            (len(id_batches) + 1) * weight_per_batch):
          cumulative_weight += curr_weight
          id_batches.append(np.array(curr_batch))
          curr_batch, curr_weight = [], 0.0
    random.shuffle(id_batches)

    for id_batch in id_batches:
      yield self._make_minibatch(id_batch)

  def endless_minibatches(self, minibatch_size):
    while True:
      for mb in self.get_minibatches(minibatch_size):
        yield mb

  def _make_minibatch(self, ids):
    examples = [self.examples[i] for i in ids]
    sentence_lengths = np.array([len(e.words) for e in examples])
    max_word_length = min(max(max(len(word) for word in e.chars)
                              for e in examples),
                          self._config.max_word_length)
    characters = [[[embeddings.PAD] + [embeddings.START] + w[:max_word_length] +
                   [embeddings.END] + [embeddings.PAD] for w in e.chars]
                  for e in examples]
    # the first and last words are masked because they are start/end tokens
    mask = build_array([[0] + [1] * (length - 2) + [0]
                        for length in sentence_lengths])
    words = build_array([e.words for e in examples])
    chars = build_array(characters, dtype='int16')
    return Minibatch(
        task_name=self.task_name,
        size=ids.size,
        examples=examples,
        ids=ids,
        teacher_predictions={},
        words=words,
        chars=chars,
        lengths=sentence_lengths,
        mask=mask,
    )


Minibatch = collections.namedtuple('Minibatch', [
    'task_name', 'size', 'examples', 'ids', 'teacher_predictions',
    'words', 'chars', 'lengths', 'mask'
])
