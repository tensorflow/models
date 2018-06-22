from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _dynamic_pooling(w_embs):
  """Dynamic Pooling layer.

  Given the variable-sized output of the convolution layer,
  the pooling with a fixed pooling kernel size and stride would
  produce variable-sized output, whereas the following fully-connected
  layer expects fixes input layer size.
  Thus we fix the number of pooling units (to 2) and dynamically
  determine the pooling region size on each data point.

  Args:
    w_embs: a input tensor with dimensionality of 1.
  Returns:
    A tensor of size 2.
  """
  # a Lambda layer maintain separate context, so that tf should be imported
  # here.
  import tensorflow as tf
  t = tf.expand_dims(w_embs, 2)
  pool_size = w_embs.shape[1].value / 2
  pooled = tf.keras.backend.pool2d(t, (pool_size, 1), strides=(
      pool_size, 1), data_format="channels_last")
  return tf.squeeze(pooled, 2)


def _dynamic_pooling_output_shape(input_shape):
  """Output shape for the dynamic pooling layer.

  This function is used for keras Lambda layer to indicate
  the output shape of the dynamic poolic layer.

  Args:
    input_shape: A tuple for the input shape.
  Returns:
    output shape for the dynamic pooling layer.
  """
  shape = list(input_shape)
  assert len(shape) == 2  # only valid for 2D tensors
  shape[1] = 2
  return tuple(shape)


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
      tf.keras.models.Model: A model.
    """

    input = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)

    layer = tf.keras.layers.Embedding(num_words, output_dim=emb_dim)(input)

    layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer)
    layer_conv3 = tf.keras.layers.Lambda(_dynamic_pooling,
        output_shape=_dynamic_pooling_output_shape)(layer_conv3)
    layer_conv3 = tf.keras.layers.Flatten()(layer_conv3)

    layer_conv2 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer)
    layer_conv2 = tf.keras.layers.Lambda(_dynamic_pooling,
        output_shape=_dynamic_pooling_output_shape)(layer_conv2)
    layer_conv2 = tf.keras.layers.Flatten()(layer_conv2)

    layer = tf.keras.layers.concatenate([layer_conv2, layer_conv3], axis=1)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)

    output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)

    super(CNN, self).__init__(inputs=[input], outputs=output)
