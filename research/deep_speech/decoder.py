
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Deep speech decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from nltk.metrics import distance
import numpy as np


class DeepSpeechDecoder(object):
  """Greedy decoder implementation for Deep Speech model."""

  def __init__(self, labels, blank_index=28):
    """Decoder initialization.

    Args:
      labels: a string specifying the speech labels for the decoder to use.
      blank_index: an integer specifying index for the blank character.
        Defaults to 28.
    """
    # e.g. labels = "[a-z]' _"
    self.labels = labels
    self.blank_index = blank_index
    self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

  def convert_to_string(self, sequence):
    """Convert a sequence of indexes into corresponding string."""
    return ''.join([self.int_to_char[i] for i in sequence])

  def wer(self, decode, target):
    """Computes the Word Error Rate (WER).

    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.

    Args:
      decode: string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number for the WER of the current decode-target pair.
    """
    # Map each word to a new char.
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))

  def cer(self, decode, target):
    """Computes the Character Error Rate (CER).

    CER is defined as the edit distance between the two given strings.

    Args:
      decode: a string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number denoting the CER for the current sentence pair.
    """
    return distance.edit_distance(decode, target)

  def decode(self, logits):
    """Decode the best guess from logits using greedy algorithm."""
    # Choose the class with maximimum probability.
    best = list(np.argmax(logits, axis=1))
    # Merge repeated chars.
    merge = [k for k, _ in itertools.groupby(best)]
    # Remove the blank index in the decoded sequence.
    merge_remove_blank = []
    for k in merge:
      if k != self.blank_index:
        merge_remove_blank.append(k)

    return self.convert_to_string(merge_remove_blank)
