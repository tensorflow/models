"""Main function for the sentiment analysis model.

The model makes use of concatenation of two CNN layers with
different kernel sizes. See `sentiment_model.py`
for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

from data import dataset
import sentiment_model

_DROPOUT_RATE = 0.95


def run_model(dataset_name, emb_dim, voc_size, sen_len,
              hid_dim, batch_size, epochs):
  """Run training loop and an evaluation at the end.

  Args:
    dataset_name: Dataset name to be trained and evaluated.
    emb_dim: The dimension of the Embedding layer.
    voc_size: The number of the most frequent tokens
      to be used from the corpus.
    sen_len: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
    hid_dim: The dimension of the Embedding layer.
    batch_size: The size of each batch during training.
    epochs: The number of the iteration over the training set for training.
  """

  model = sentiment_model.CNN(emb_dim, voc_size, sen_len,
                              hid_dim, dataset.get_num_class(dataset_name),
                              _DROPOUT_RATE)
  model.summary()

  model.compile(loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

  tf.logging.info("Loading the data")
  x_train, y_train, x_test, y_test = dataset.load(
      dataset_name, voc_size, sen_len)

  model.fit(x_train, y_train, batch_size=batch_size,
            validation_split=0.4, epochs=epochs)
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  tf.logging.info("Score: {}".format(score))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--dataset", help="Dataset to be trained "
                                              "and evaluated.",
                      type=str, choices=["imdb"], default="imdb")

  parser.add_argument("-e", "--embedding_dim",
                      help="The dimension of the Embedding layer.",
                      type=int, default=512)

  parser.add_argument("-v", "--vocabulary_size",
                      help="The number of the words to be considered "
                           "in the dataset corpus.",
                      type=int, default=6000)

  parser.add_argument("-s", "--sentence_length",
                      help="The number of words in a data point."
                           "Entries of smaller length are padded.",
                      type=int, default=600)

  parser.add_argument("-c", "--hidden_dim",
                      help="The number of the CNN layer filters.",
                      type=int, default=512)

  parser.add_argument("-b", "--batch_size",
                      help="The size of each batch for training.",
                      type=int, default=500)

  parser.add_argument("-p", "--epochs",
                      help="The number of epochs for training.",
                      type=int, default=55)

  args = parser.parse_args()

  run_model(args.dataset, args.embedding_dim, args.vocabulary_size,
            args.sentence_length, args.hidden_dim,
            args.batch_size, args.epochs)
