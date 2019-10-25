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

"""Base class for training examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base import embeddings


CONTRACTION_WORDS = set(w + 'n' for w in
                        ['do', 'does', 'did', 'is', 'are', 'was', 'were', 'has',
                         'have', 'had', 'could', 'would', 'should', 'ca', 'wo',
                         'ai', 'might'])


class Example(object):
  def __init__(self, words, word_vocab, char_vocab):
    words = words[:]
    # Fix inconsistent tokenization between datasets
    for i in range(len(words)):
      if (words[i].lower() == '\'t' and i > 0 and
          words[i - 1].lower() in CONTRACTION_WORDS):
        words[i] = words[i - 1][-1] + words[i]
        words[i - 1] = words[i - 1][:-1]

    self.words = ([embeddings.START] +
                  [word_vocab[embeddings.normalize_word(w)] for w in words] +
                  [embeddings.END])
    self.chars = ([[embeddings.MISSING]] +
                  [[char_vocab[c] for c in embeddings.normalize_chars(w)]
                   for w in words] +
                  [[embeddings.MISSING]])

  def __repr__(self,):
    inv_char_vocab = embeddings.get_inv_char_vocab()
    return ' '.join([''.join([inv_char_vocab[c] for c in w])
                     for w in self.chars])
