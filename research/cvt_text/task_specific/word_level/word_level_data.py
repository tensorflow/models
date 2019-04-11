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

"""Utilities for processing word-level datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import tensorflow as tf

from base import embeddings
from base import utils
from corpus_processing import example
from corpus_processing import minibatching
from task_specific.word_level import tagging_utils


class TaggedDataLoader(object):
  def __init__(self, config, name, is_token_level):
    self._config = config
    self._task_name = name
    self._raw_data_path = os.path.join(config.raw_data_topdir, name)
    self._is_token_level = is_token_level
    self.label_mapping_path = os.path.join(
        config.preprocessed_data_topdir,
        (name if is_token_level else
         name + '_' + config.label_encoding) + '_label_mapping.pkl')

    if self.label_mapping:
      self._n_classes = len(set(self.label_mapping.values()))
    else:
      self._n_classes = None

  def get_dataset(self, split):
    if (split == 'train' and not self._config.for_preprocessing and
        tf.gfile.Exists(os.path.join(self._raw_data_path, 'train_subset.txt'))):
      split = 'train_subset'
    return minibatching.Dataset(
        self._config, self._get_examples(split), self._task_name)

  def get_labeled_sentences(self, split):
    sentences = []
    path = os.path.join(self._raw_data_path, split + '.txt')
    if not tf.gfile.Exists(path):
      if self._config.for_preprocessing:
        return []
      else:
        raise ValueError('Unable to load data from', path)

    with tf.gfile.GFile(path, 'r') as f:
      sentence = []
      for line in f:
        line = line.strip().split()
        if not line:
          if sentence:
            words, tags = zip(*sentence)
            sentences.append((words, tags))
            sentence = []
          continue
        if line[0] == '-DOCSTART-':
          continue
        word, tag = line[0], line[-1]
        sentence.append((word, tag))
    return sentences

  @property
  def label_mapping(self):
    if not self._config.for_preprocessing:
      return utils.load_cpickle(self.label_mapping_path)

    tag_counts = collections.Counter()
    train_tags = set()
    for split in ['train', 'dev', 'test']:
      for words, tags in self.get_labeled_sentences(split):
        if not self._is_token_level:
          span_labels = tagging_utils.get_span_labels(tags)
          tags = tagging_utils.get_tags(
              span_labels, len(words), self._config.label_encoding)
        for tag in tags:
          if self._task_name == 'depparse':
            tag = tag.split('-')[1]
          tag_counts[tag] += 1
          if split == 'train':
            train_tags.add(tag)
    if self._task_name == 'ccg':
      # for CCG, there are tags in the test sets that aren't in the train set
      # all tags not in the train set get mapped to a special label
      # the model will never predict this label because it never sees it in the
      # training set
      not_in_train_tags = []
      for tag, count in tag_counts.items():
        if tag not in train_tags:
          not_in_train_tags.append(tag)
      label_mapping = {
          label: i for i, label in enumerate(sorted(filter(
            lambda t: t not in not_in_train_tags, tag_counts.keys())))
      }
      n = len(label_mapping)
      for tag in not_in_train_tags:
        label_mapping[tag] = n
    else:
      labels = sorted(tag_counts.keys())
      if self._task_name == 'depparse':
        labels.remove('root')
        labels.insert(0, 'root')
      label_mapping = {label: i for i, label in enumerate(labels)}
    return label_mapping

  def _get_examples(self, split):
    word_vocab = embeddings.get_word_vocab(self._config)
    char_vocab = embeddings.get_char_vocab()
    examples = [
        TaggingExample(
            self._config, self._is_token_level, words, tags,
            word_vocab, char_vocab, self.label_mapping, self._task_name)
        for words, tags in self.get_labeled_sentences(split)]
    if self._config.train_set_percent < 100:
      utils.log('using reduced train set ({:}%)'.format(
          self._config.train_set_percent))
      random.shuffle(examples)
      examples = examples[:int(len(examples) *
                               self._config.train_set_percent / 100.0)]
    return examples


class TaggingExample(example.Example):
  def __init__(self, config, is_token_level, words, original_tags,
               word_vocab, char_vocab, label_mapping, task_name):
    super(TaggingExample, self).__init__(words, word_vocab, char_vocab)
    if is_token_level:
      labels = original_tags
    else:
      span_labels = tagging_utils.get_span_labels(original_tags)
      labels = tagging_utils.get_tags(
          span_labels, len(words), config.label_encoding)

    if task_name == 'depparse':
      self.labels = []
      for l in labels:
        split = l.split('-')
        self.labels.append(
            len(label_mapping) * (0 if split[0] == '0' else 1 + int(split[0]))
            + label_mapping[split[1]])
    else:
      self.labels = [label_mapping[l] for l in labels]
