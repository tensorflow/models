"""Utility module for sentiment analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

START_CHAR = 1
END_CHAR = 2
OOV_CHAR = 3


def pad_sentence(sentence, sentence_length):
  """Pad the given sentense at the end.

  If the input is longer than sentence_length,
  the remaining portion is dropped.
  END_CHAR is used for the padding.

  Args:
    sentence: A numpy array of integers.
    sentence_length: The length of the input after the padding.
  Returns:
    A numpy array of integers of the given length.
  """
  sentence = sentence[:sentence_length]
  if len(sentence) < sentence_length:
    sentence = np.pad(sentence, (0, sentence_length - len(sentence)),
                      "constant", constant_values=(START_CHAR, END_CHAR))

  return sentence
