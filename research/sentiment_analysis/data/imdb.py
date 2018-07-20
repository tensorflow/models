"""IMDB Dataset module for sentiment analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from data.util import OOV_CHAR
from data.util import pad_sentence
from data.util import START_CHAR

NUM_CLASS = 2


def load(vocabulary_size, sentence_length):
  """Returns training and evaluation input for imdb dataset.

  Args:
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of length 4, for training and evaluation data,
    each being an numpy array.
  """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
      path="imdb.npz",
      num_words=vocabulary_size,
      skip_top=0,
      maxlen=None,
      seed=113,
      start_char=START_CHAR,
      oov_char=OOV_CHAR,
      index_from=OOV_CHAR+1)

  x_train_processed = []
  for sen in x_train:
    sen = pad_sentence(sen, sentence_length)
    x_train_processed.append(np.array(sen))
  x_train_processed = np.array(x_train_processed)

  x_test_processed = []
  for sen in x_test:
    sen = pad_sentence(sen, sentence_length)
    x_test_processed.append(np.array(sen))
  x_test_processed = np.array(x_test_processed)

  return x_train_processed, np.eye(NUM_CLASS)[y_train], \
         x_test_processed, np.eye(NUM_CLASS)[y_test]
