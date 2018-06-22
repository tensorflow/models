from data.util import pad_sentence, to_dataset, START_CHAR, OOV_CHAR
import tensorflow as tf, numpy as np

NUM_CLASS = 2


def construct_input_fns(vocabulary_size, sentence_length,
                        batch_size, repeat=1):
  """Returns training and evaluation input functions.

  Args:
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
    batch_size: Number of data in each batch.
    repeat: The number of epoch.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of training and evaluation input function.
  """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
      path="imdb.npz",
      num_words=vocabulary_size,
      skip_top=0,
      maxlen=None,
      seed=113,
      start_char=START_CHAR,
      oov_char=OOV_CHAR,
      index_from=OOV_CHAR + 1)

  def train_input_fn():
    dataset = to_dataset(
        np.array([pad_sentence(s, sentence_length) for s in x_train]),
        np.eye(NUM_CLASS)[y_train], batch_size, repeat)
    dataset = dataset.shuffle(len(x_train), reshuffle_each_iteration=True)
    return dataset

  def eval_input_fn():
    dataset = to_dataset(
        np.array([pad_sentence(s, sentence_length) for s in x_test]),
        np.eye(NUM_CLASS)[y_test], batch_size, repeat)
    return dataset

  return train_input_fn, eval_input_fn
