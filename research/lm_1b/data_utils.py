# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""A library for loading 1B word benchmark dataset."""

import random

import numpy as np
import tensorflow as tf


class Vocabulary(object):
  """Class that holds a vocabulary for the dataset."""

  def __init__(self, filename):
    """Initialize vocabulary.

    Args:
      filename: Vocabulary file name.
    """

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
    return self.unk

  def id_to_word(self, cur_id):
    if cur_id < self.size:
      return self._id_to_word[cur_id]
    return 'ERROR'

  def decode(self, cur_ids):
    """Convert a list of ids to a sentence, with space inserted."""
    return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

  def encode(self, sentence):
    """Convert a sentence to a list of ids, with special tokens added."""
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
      self._word_char_ids[i] = self._convert_word_to_char_ids(word)

  @property
  def word_char_ids(self):
    return self._word_char_ids

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


def get_batch(generator, batch_size, num_steps, max_word_length, pad=False):
  """Read batches of input."""
  cur_stream = [None] * batch_size

  inputs = np.zeros([batch_size, num_steps], np.int32)
  char_inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
  global_word_ids = np.zeros([batch_size, num_steps], np.int32)
  targets = np.zeros([batch_size, num_steps], np.int32)
  weights = np.ones([batch_size, num_steps], np.float32)

  no_more_data = False
  while True:
    inputs[:] = 0
    char_inputs[:] = 0
    global_word_ids[:] = 0
    targets[:] = 0
    weights[:] = 0.0

    for i in range(batch_size):
      cur_pos = 0

      while cur_pos < num_steps:
        if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
          try:
            cur_stream[i] = list(generator.next())
          except StopIteration:
            # No more data, exhaust current streams and quit
            no_more_data = True
            break

        how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
        next_pos = cur_pos + how_many

        inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
        char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
        global_word_ids[i, cur_pos:next_pos] = cur_stream[i][2][:how_many]
        targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]
        weights[i, cur_pos:next_pos] = 1.0

        cur_pos = next_pos
        cur_stream[i][0] = cur_stream[i][0][how_many:]
        cur_stream[i][1] = cur_stream[i][1][how_many:]
        cur_stream[i][2] = cur_stream[i][2][how_many:]

        if pad:
          break

    if no_more_data and np.sum(weights) == 0:
      # There is no more data and this is an empty batch. Done!
      break
    yield inputs, char_inputs, global_word_ids, targets, weights


class LM1BDataset(object):
  """Utility class for 1B word benchmark dataset.

  The current implementation reads the data from the tokenized text files.
  """

  def __init__(self, filepattern, vocab):
    """Initialize LM1BDataset reader.

    Args:
      filepattern: Dataset file pattern.
      vocab: Vocabulary.
    """
    self._vocab = vocab
    self._all_shards = tf.gfile.Glob(filepattern)
    tf.logging.info('Found %d shards at %s', len(self._all_shards), filepattern)

  def _load_random_shard(self):
    """Randomly select a file and read it."""
    return self._load_shard(random.choice(self._all_shards))

  def _load_shard(self, shard_name):
    """Read one file and convert to ids.

    Args:
      shard_name: file path.

    Returns:
      list of (id, char_id, global_word_id) tuples.
    """
    tf.logging.info('Loading data from: %s', shard_name)
    with tf.gfile.Open(shard_name) as f:
      sentences = f.readlines()
    chars_ids = [self.vocab.encode_chars(sentence) for sentence in sentences]
    ids = [self.vocab.encode(sentence) for sentence in sentences]

    global_word_ids = []
    current_idx = 0
    for word_ids in ids:
      current_size = len(word_ids) - 1  # without <BOS> symbol
      cur_ids = np.arange(current_idx, current_idx + current_size)
      global_word_ids.append(cur_ids)
      current_idx += current_size

    tf.logging.info('Loaded %d words.', current_idx)
    tf.logging.info('Finished loading')
    return zip(ids, chars_ids, global_word_ids)

  def _get_sentence(self, forever=True):
    while True:
      ids = self._load_random_shard()
      for current_ids in ids:
        yield current_ids
      if not forever:
        break

  def get_batch(self, batch_size, num_steps, pad=False, forever=True):
    return get_batch(self._get_sentence(forever), batch_size, num_steps,
                     self.vocab.max_word_length, pad=pad)

  @property
  def vocab(self):
    return self._vocab
