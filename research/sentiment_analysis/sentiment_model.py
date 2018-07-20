"""Model for sentiment analysis.

The model makes use of concatenation of two CNN layers with
different kernel sizes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CNN(tf.keras.models.Model):
  """CNN for sentimental analysis."""

  def __init__(self, emb_dim, num_words, sentence_length, hid_dim,
               class_dim, dropout_rate):
    """Initialize CNN model.

    Args:
      emb_dim: The dimension of the Embedding layer.
      num_words: The number of the most frequent tokens
        to be used from the corpus.
      sentence_length: The number of words in each sentence.
        Longer sentences get cut, shorter ones padded.
      hid_dim: The dimension of the Embedding layer.
      class_dim: The number of the CNN layer filters.
      dropout_rate: The portion of kept value in the Dropout layer.
    Returns:
      tf.keras.models.Model: A Keras model.
    """

    input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)

    layer = tf.keras.layers.Embedding(num_words, output_dim=emb_dim)(input_layer)

    layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer)
    layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)

    layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer)
    layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)

    layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)

    super(CNN, self).__init__(inputs=[input_layer], outputs=output)
