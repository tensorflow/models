# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Vocabulary(object):
  """Class that holds a vocabulary for the dataset."""

  def __init__(self, filename):

    self._id_to_word = []
    self._word_to_id = {}
    self._unk = -1
    self._bos = -1
    self._eos = -1

    with tf.gfile.Open(filename) as f:
      idx = 0
      for line in f:
        word_name = line.strip()
        if word_name == '<S>':
          self._bos = idx
        elif word_name == '</S>':
          self._eos = idx
        elif word_name == '<UNK>':
          self._unk = idx
        if word_name == '!!!MAXTERMID':
          continue

        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

  @property
  def bos(self):
    return self._bos

  @property
  def eos(self):
    return self._eos

  @property
  def unk(self):
    return self._unk

  @property
  def size(self):
    return len(self._id_to_word)

  def word_to_id(self, word):
    if word in self._word_to_id:
      return self._word_to_id[word]
    else:
      if word.lower() in self._word_to_id:
        return self._word_to_id[word.lower()]
    return self.unk

  def id_to_word(self, cur_id):
    if cur_id < self.size:
      return self._id_to_word[int(cur_id)]
    return '<ERROR_out_of_vocab_id>'

  def decode(self, cur_ids):
    return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

  def encode(self, sentence):
    word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
    return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
  """Vocabulary containing character-level information."""

  def __init__(self, filename, max_word_length):
    super(CharsVocabulary, self).__init__(filename)

    self._max_word_length = max_word_length
    chars_set = set()

    for word in self._id_to_word:
      chars_set |= set(word)

    free_ids = []
    for i in range(256):
      if chr(i) in chars_set:
        continue
      free_ids.append(chr(i))

    if len(free_ids) < 5:
      raise ValueError('Not enough free char ids: %d' % len(free_ids))

    self.bos_char = free_ids[0]  # <begin sentence>
    self.eos_char = free_ids[1]  # <end sentence>
    self.bow_char = free_ids[2]  # <begin word>
    self.eow_char = free_ids[3]  # <end word>
    self.pad_char = free_ids[4]  # <padding>

    chars_set |= {self.bos_char, self.eos_char, self.bow_char, self.eow_char,
                  self.pad_char}

    self._char_set = chars_set
    num_words = len(self._id_to_word)

    self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

    self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
    self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

    for i, word in enumerate(self._id_to_word):
      if i == self.bos:
        self._word_char_ids[i] = self.bos_chars
      elif i == self.eos:
        self._word_char_ids[i] = self.eos_chars
      else:
        self._word_char_ids[i] = self._convert_word_to_char_ids(word)

  @property
  def max_word_length(self):
    return self._max_word_length

  def _convert_word_to_char_ids(self, word):
    code = np.zeros([self.max_word_length], dtype=np.int32)
    code[:] = ord(self.pad_char)

    if len(word) > self.max_word_length - 2:
      word = word[:self.max_word_length-2]
    cur_word = self.bow_char + word + self.eow_char
    for j in range(len(cur_word)):
      code[j] = ord(cur_word[j])
    return code

  def word_to_char_ids(self, word):
    if word in self._word_to_id:
      return self._word_char_ids[self._word_to_id[word]]
    else:
      return self._convert_word_to_char_ids(word)

  def encode_chars(self, sentence):
    chars_ids = [self.word_to_char_ids(cur_word)
                 for cur_word in sentence.split()]
    return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


_SPECIAL_CHAR_MAP = {
    '\xe2\x80\x98': '\'',
    '\xe2\x80\x99': '\'',
    '\xe2\x80\x9c': '"',
    '\xe2\x80\x9d': '"',
    '\xe2\x80\x93': '-',
    '\xe2\x80\x94': '-',
    '\xe2\x88\x92': '-',
    '\xce\x84': '\'',
    '\xc2\xb4': '\'',
    '`': '\''
}

_START_SPECIAL_CHARS = ['.', ',', '?', '!', ';', ':', '[', ']', '\'', '+', '/',
                        '\xc2\xa3', '$', '~', '*', '%', '{', '}', '#', '&', '-',
                        '"', '(', ')', '='] + list(_SPECIAL_CHAR_MAP.keys())
_SPECIAL_CHARS = _START_SPECIAL_CHARS + [
    '\'s', '\'m', '\'t', '\'re', '\'d', '\'ve', '\'ll']


def tokenize(sentence):
  """Tokenize a sentence."""
  sentence = str(sentence)
  words = sentence.strip().split()
  tokenized = []  # return this

  for word in words:
    if word.lower() in ['mr.', 'ms.']:
      tokenized.append(word)
      continue

    # Split special chars at the start of word
    will_split = True
    while will_split:
      will_split = False
      for char in _START_SPECIAL_CHARS:
        if word.startswith(char):
          tokenized.append(char)
          word = word[len(char):]
          will_split = True

    # Split special chars at the end of word
    special_end_tokens = []
    will_split = True
    while will_split:
      will_split = False
      for char in _SPECIAL_CHARS:
        if word.endswith(char):
          special_end_tokens = [char] + special_end_tokens
          word = word[:-len(char)]
          will_split = True

    if word:
      tokenized.append(word)
    tokenized += special_end_tokens

  # Add necessary end of sentence token.
  if tokenized[-1] not in ['.', '!', '?']:
    tokenized += ['.']
  return tokenized


def parse_commonsense_reasoning_test(test_data_name):
  """Read JSON test data."""
  with tf.gfile.Open(os.path.join(
      FLAGS.data_dir, 'commonsense_test',
      '{}.json'.format(test_data_name)), 'r') as f:
    data = json.load(f)

  question_ids = [d['question_id'] for d in data]
  sentences = [tokenize(d['substitution']) for d in data]
  labels = [d['correctness'] for d in data]

  return question_ids, sentences, labels


PAD = '<padding>'


def cut_to_patches(sentences, batch_size, num_timesteps):
  """Cut sentences into patches of shape (batch_size, num_timesteps).

  Args:
    sentences: a list of sentences, each sentence is a list of str token.
    batch_size: batch size
    num_timesteps: number of backprop step

  Returns:
    patches: A 2D matrix,
      each entry is a matrix of shape (batch_size, num_timesteps).
  """
  preprocessed = [['<S>']+sentence+['</S>'] for sentence in sentences]
  max_len = max([len(sent) for sent in preprocessed])

  # Pad to shape [height, width]
  # where height is a multiple of batch_size
  # and width is a multiple of num_timesteps
  nrow = int(np.ceil(len(preprocessed) * 1.0 / batch_size))
  ncol = int(np.ceil(max_len * 1.0 / num_timesteps))
  height, width = nrow * batch_size, ncol * num_timesteps + 1
  preprocessed = [sent + [PAD] * (width - len(sent)) for sent in preprocessed]
  preprocessed += [[PAD] * width] * (height - len(preprocessed))

  # Cut preprocessed into patches of shape [batch_size, num_timesteps]
  patches = []
  for row in range(nrow):
    patches.append([])
    for col in range(ncol):
      patch = [sent[col * num_timesteps:
                    (col+1) * num_timesteps + 1]
               for sent in preprocessed[row * batch_size:
                                        (row+1) * batch_size]]
      if np.all(np.array(patch)[:, 1:] == PAD):
        patch = None  # no need to process this patch.
      patches[-1].append(patch)
  return patches


def _substitution_mask(sent1, sent2):
  """Binary mask identifying substituted part in two sentences.

  Example sentence and their mask:
    First sentence  = "I like the cat        's color"
                       0 0    0   1           0 0
    Second sentence = "I like the yellow dog 's color"
                       0 0    0   1      1    0 0

  Args:
    sent1: first sentence
    sent2: second sentence

  Returns:
    mask1: mask for first sentence
    mask2: mask for second sentence
  """
  mask1_start, mask2_start = [], []
  while sent1[0] == sent2[0]:
    sent1 = sent1[1:]
    sent2 = sent2[1:]
    mask1_start.append(0.)
    mask2_start.append(0.)

  mask1_end, mask2_end = [], []
  while sent1[-1] == sent2[-1]:
    if (len(sent1) == 1) or (len(sent2) == 1):
      break
    sent1 = sent1[:-1]
    sent2 = sent2[:-1]
    mask1_end = [0.] + mask1_end
    mask2_end = [0.] + mask2_end

  assert sent1 or sent2, 'Two sentences are identical.'
  return (mask1_start + [1.] * len(sent1) + mask1_end,
          mask2_start + [1.] * len(sent2) + mask2_end)


def _convert_to_partial(scoring1, scoring2):
  """Convert full scoring into partial scoring."""
  mask1, mask2 = _substitution_mask(
      scoring1['sentence'], scoring2['sentence'])

  def _partial_score(scoring, mask):
    word_probs = [max(_) for _ in zip(scoring['word_probs'], mask)]
    scoring.update(word_probs=word_probs,
                   joint_prob=np.prod(word_probs))

  _partial_score(scoring1, mask1)
  _partial_score(scoring2, mask2)


def compare_substitutions(question_ids, scorings, mode='full'):
  """Return accuracy by comparing two consecutive scorings."""
  prediction_correctness = []
  # Compare two consecutive substitutions
  for i in range(len(scorings) // 2):
    scoring1, scoring2 = scorings[2*i: 2*i+2]
    if mode == 'partial':  # fix joint prob into partial prob
      _convert_to_partial(scoring1, scoring2)

    prediction_correctness.append(
        (scoring2['joint_prob'] > scoring1['joint_prob']) ==
         scoring2['correctness'])

  # Two consecutive substitutions always belong to the same question
  question_ids = [qid for i, qid in enumerate(question_ids) if i % 2 == 0]
  assert len(question_ids) == len(prediction_correctness)
  num_questions = len(set(question_ids))

  # Question is correctly answered only if
  # all predictions of the same question_id is correct
  num_correct_answer = 0
  previous_qid = None
  correctly_answered = False
  for predict, qid in zip(prediction_correctness, question_ids):
    if qid != previous_qid:
      previous_qid = qid
      num_correct_answer += int(correctly_answered)
      correctly_answered = True
    correctly_answered = correctly_answered and predict
  num_correct_answer += int(correctly_answered)

  return num_correct_answer / num_questions
