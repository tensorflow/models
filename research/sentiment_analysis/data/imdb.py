import tensorflow as tf
import numpy as np
from data.util import to_dataset, pad_sentence, START_CHAR, OOV_CHAR

NUM_CLASS = 2


def construct_input_fns(vocabulary_size, sentence_length, batch_size, repeat=1):
  """Returns a tuple of input functions, one for training, the other for evaluation.
  Args:
    vocabulary_size: The number of the most frequent tokens to be used from the corpus.
    sentence_length: The number of words in each sentence. Longer sentences get cut, shorter ones padded.
    batch_size: Number of data in each batch.
    repeat: The number of epoch.
  Raises:
    ValueError: if the dataset value is not valid.
  """
  (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.imdb.load_data(path="imdb.npz",
                                                                              num_words=vocabulary_size,
                                                                              skip_top=0,
                                                                              maxlen=None,
                                                                              seed=113,
                                                                              start_char=START_CHAR,
                                                                              oov_char=OOV_CHAR,
                                                                              index_from=OOV_CHAR + 1)

  def train_input_fn():
    return to_dataset(np.array([pad_sentence(s, sentence_length) for s in _x_train]),
                      np.eye(NUM_CLASS)[_y_train], batch_size, repeat)

  def eval_input_fn():
    return to_dataset(np.array([pad_sentence(s, sentence_length) for s in _x_test]),
                      np.eye(NUM_CLASS)[_y_test], batch_size, repeat)

  return train_input_fn, eval_input_fn
