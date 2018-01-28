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

"""Data batchers for data described in ..//data_prep/README.md."""

import glob
import random
import struct
import sys

from tensorflow.core.example import example_pb2


# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'


class Vocab(object):
  """Vocabulary class for mapping words and ids."""

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line)
          continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0])
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._count += 1
        if self._count > max_size:
          raise ValueError('Too many words: >%d.' % max_size)

  def CheckVocab(self, word):
    if word not in self._word_to_id:
      return None
    return self._word_to_id[word]
  
  def WordToId(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def IdToWord(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('id not found in vocab: %d.' % word_id)
    return self._id_to_word[word_id]

  def NumIds(self):
    return self._count


def ExampleGen(data_path, num_epochs=None):
  """Generates tf.Examples from path of data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path: path to tf.Example data files.
    num_epochs: Number of times to go through the data. None means infinite.

  Yields:
    Deserialized tf.Example.

  If there are multiple files specified, they accessed in a random order.
  """
  epoch = 0
  while True:
    if num_epochs is not None and epoch >= num_epochs:
      break
    filelist = glob.glob(data_path)
    assert filelist, 'Empty filelist.'
    random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

    epoch += 1


def Pad(ids, pad_id, length):
  """Pad or trim list to len length.

  Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

  Returns:
    ids trimmed or padded with pad_id
  """
  assert pad_id is not None
  assert length is not None

  if len(ids) < length:
    a = [pad_id] * (length - len(ids))
    return ids + a
  else:
    return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
  """Get ids corresponding to words in text.

  Assumes tokens separated by space.

  Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

  Returns:
    A list of ints representing word ids.
  """
  ids = []
  for w in text.split():
    i = vocab.WordToId(w)
    if i >= 0:
      ids.append(i)
    else:
      ids.append(vocab.WordToId(UNKNOWN_TOKEN))
  if pad_len is not None:
    return Pad(ids, pad_id, pad_len)
  return ids


def Ids2Words(ids_list, vocab):
  """Get words from ids.

  Args:
    ids_list: list of int32
    vocab: TextVocabulary object

  Returns:
    List of words corresponding to ids.
  """
  assert isinstance(ids_list, list), '%s  is not a list' % ids_list
  return [vocab.IdToWord(i) for i in ids_list]


def SnippetGen(text, start_tok, end_tok, inclusive=True):
  """Generates consecutive snippets between start and end tokens.

  Args:
    text: a string
    start_tok: a string denoting the start of snippets
    end_tok: a string denoting the end of snippets
    inclusive: Whether include the tokens in the returned snippets.

  Yields:
    String snippets
  """
  cur = 0
  while True:
    try:
      start_p = text.index(start_tok, cur)
      end_p = text.index(end_tok, start_p + 1)
      cur = end_p + len(end_tok)
      if inclusive:
        yield text[start_p:cur]
      else:
        yield text[start_p+len(start_tok):end_p]
    except ValueError as e:
      raise StopIteration('no more snippets in text: %s' % e)


def GetExFeatureText(ex, key):
  return ex.features.feature[key].bytes_list.value[0]


def ToSentences(paragraph, include_token=True):
  """Takes tokens of a paragraph and returns list of sentences.

  Args:
    paragraph: string, text of paragraph
    include_token: Whether include the sentence separation tokens result.

  Returns:
    List of sentence strings.
  """
  s_gen = SnippetGen(paragraph, SENTENCE_START, SENTENCE_END, include_token)
  return [s for s in s_gen]
