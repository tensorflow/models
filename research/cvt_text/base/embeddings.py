# coding=utf-8
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


"""Utilities for handling word embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import numpy as np
import tensorflow as tf

from base import utils


_CHARS = [
    # punctuation
    '!', '\'', '#', '$', '%', '&', '"', '(', ')', '*', '+', ',', '-', '.',
    '/', '\\', '_', '`', '{', '}', '[', ']', '<', '>', ':', ';', '?', '@',
    # digits
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # letters
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # special characters
    '£', '€', '®', '™', '�', '½', '»', '•', '—', '“', '”', '°', '‘', '’'
]

# words not in GloVe that still should have embeddings
_EXTRA_WORDS = [
    # common digit patterns
    '0/0', '0/00', '00/00', '0/000',
    '00/00/00', '0/00/00', '00/00/0000', '0/00/0000',
    '00-00', '00-00-00', '0-00-00', '00-00-0000', '0-00-0000', '0000-00-00',
    '00-0-00-0', '00000000', '0:00.000', '00:00.000',
    '0%', '00%', '00.' '0000.', '0.0bn', '0.0m', '0-', '00-',
    # ontonotes uses **f to represent formulas and -amp- instead of amperstands
    '**f', '-amp-'
]
SPECIAL_TOKENS = ['<pad>', '<unk>', '<start>', '<end>', '<missing>']
NUM_CHARS = len(_CHARS) + len(SPECIAL_TOKENS)
PAD, UNK, START, END, MISSING = 0, 1, 2, 3, 4


class Vocabulary(collections.OrderedDict):
  def __getitem__(self, w):
    return self.get(w, UNK)


@utils.Memoize
def get_char_vocab():
  characters = _CHARS
  for i, special in enumerate(SPECIAL_TOKENS):
    characters.insert(i, special)
  return Vocabulary({c: i for i, c in enumerate(characters)})


@utils.Memoize
def get_inv_char_vocab():
  return {i: c for c, i in get_char_vocab().items()}


def get_word_vocab(config):
  return Vocabulary(utils.load_cpickle(config.word_vocabulary))


def get_word_embeddings(config):
  return utils.load_cpickle(config.word_embeddings)


@utils.Memoize
def _punctuation_ids(vocab_path):
  vocab = Vocabulary(utils.load_cpickle(vocab_path))
  return set(i for w, i in vocab.iteritems() if w in [
      '!', '...', '``', '{', '}', '(', ')', '[', ']', '--', '-', ',', '.',
      "''", '`', ';', ':', '?'])


def get_punctuation_ids(config):
  return _punctuation_ids(config.word_vocabulary)


class PretrainedEmbeddingLoader(object):
  def __init__(self, config):
    self.config = config
    self.vocabulary = {}
    self.vectors = []
    self.vector_size = config.word_embedding_size

  def _add_vector(self, w):
    if w not in self.vocabulary:
      self.vocabulary[w] = len(self.vectors)
      self.vectors.append(np.zeros(self.vector_size, dtype='float32'))

  def build(self):
    utils.log('loading pretrained embeddings from',
              self.config.pretrained_embeddings_file)
    for special in SPECIAL_TOKENS:
      self._add_vector(special)
    for extra in _EXTRA_WORDS:
      self._add_vector(extra)
    with tf.gfile.GFile(
        self.config.pretrained_embeddings_file, 'r') as f:
      for i, line in enumerate(f):
        if i % 10000 == 0:
          utils.log('on line', i)

        split = line.decode('utf8').split()
        w = normalize_word(split[0])

        try:
          vec = np.array(map(float, split[1:]), dtype='float32')
          if vec.size != self.vector_size:
            utils.log('vector for line', i, 'has size', vec.size, 'so skipping')
            utils.log(line[:100] + '...')
            continue
        except:
          utils.log('can\'t parse line', i, 'so skipping')
          utils.log(line[:100] + '...')
          continue
        if w not in self.vocabulary:
          self.vocabulary[w] = len(self.vectors)
          self.vectors.append(vec)
    utils.log('writing vectors!')
    self._write()

  def _write(self):
    utils.write_cpickle(np.vstack(self.vectors), self.config.word_embeddings)
    utils.write_cpickle(self.vocabulary, self.config.word_vocabulary)


def normalize_chars(w):
  if w == '-LRB-':
    return '('
  elif w == '-RRB-':
    return ')'
  elif w == '-LCB-':
    return '{'
  elif w == '-RCB-':
    return '}'
  elif w == '-LSB-':
    return '['
  elif w == '-RSB-':
    return ']'
  return w.replace(r'\/', '/').replace(r'\*', '*')


def normalize_word(w):
  return re.sub(r'\d', '0', normalize_chars(w).lower())
