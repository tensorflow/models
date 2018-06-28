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
"""Network structure for DeepSpeech model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Supported rnn cells
SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTM,
    "rnn": tf.keras.layers.SimpleRNN,
    "gru": tf.keras.layers.GRU,
}

# Parameters for batch normalization
_MOMENTUM = 0.1
_EPSILON = 1e-05


def _conv_bn_layer(cnn_input, filters, kernel_size, strides, layer_id):
  """2D convolution + batch normalization layer.

  Args:
    cnn_input: input data for convolution layer.
    filters: an integer, number of output filters in the convolution.
    kernel_size: a tuple specifying the height and width of the 2D convolution
      window.
    strides: a tuple specifying the stride length of the convolution.
    layer_id: an integer specifying the layer index.

  Returns:
    tensor output from the current layer.
  """
  output = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
      activation="linear", name="cnn_{}".format(layer_id))(cnn_input)
  output = tf.keras.layers.BatchNormalization(
      momentum=_MOMENTUM, epsilon=_EPSILON)(output)
  return output


def _rnn_layer(input_data, rnn_cell, rnn_hidden_size, layer_id, rnn_activation,
               is_batch_norm, is_bidirectional):
  """Defines a batch normalization + rnn layer.

  Args:
    input_data: input tensors for the current layer.
    rnn_cell: RNN cell instance to use.
    rnn_hidden_size: an integer for the dimensionality of the rnn output space.
    layer_id: an integer for the index of current layer.
    rnn_activation: activation function to use.
    is_batch_norm: a boolean specifying whether to perform batch normalization
      on input states.
    is_bidirectional: a boolean specifying whether the rnn layer is
      bi-directional.

  Returns:
    tensor output for the current layer.
  """
  if is_batch_norm:
    input_data = tf.keras.layers.BatchNormalization(
        momentum=_MOMENTUM, epsilon=_EPSILON)(input_data)
  rnn_layer = rnn_cell(
      rnn_hidden_size, activation=rnn_activation, return_sequences=True,
      name="rnn_{}".format(layer_id))
  if is_bidirectional:
    rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")

  return rnn_layer(input_data)


def _ctc_lambda_func(args):
  """Compute ctc loss."""
  # py2 needs explicit tf import for keras Lambda layer
  import tensorflow as tf

  y_pred, labels, input_length, label_length = args
  return tf.keras.backend.ctc_batch_cost(
      labels, y_pred, input_length, label_length)


def _calc_ctc_input_length(args):
  """Compute the actual input length after convolution for ctc_loss function.

  Basically, we need to know the scaled input_length after conv layers.
  new_input_length = old_input_length * ctc_time_steps / max_time_steps

  Args:
    args: the input args to compute ctc input length.

  Returns:
    ctc_input_length, which is required for ctc loss calculation.
  """
  # py2 needs explicit tf import for keras Lambda layer
  import tensorflow as tf

  input_length, input_data, y_pred = args
  max_time_steps = tf.shape(input_data)[1]
  ctc_time_steps = tf.shape(y_pred)[1]
  ctc_input_length = tf.multiply(
      tf.to_float(input_length), tf.to_float(ctc_time_steps))
  ctc_input_length = tf.to_int32(tf.floordiv(
      ctc_input_length, tf.to_float(max_time_steps)))
  return ctc_input_length


class DeepSpeech(tf.keras.models.Model):
  """DeepSpeech model."""

  def __init__(self, input_shape, num_rnn_layers, rnn_type, is_bidirectional,
               rnn_hidden_size, rnn_activation, num_classes, use_bias):
    """Initialize DeepSpeech model.

    Args:
      input_shape: an tuple to indicate the dimension of input dataset. It has
        the format of [time_steps(T), feature_bins(F), channel(1)]
      num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
      rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
      is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
      rnn_hidden_size: an integer for the number of hidden states in each unit.
      rnn_activation: a string to indicate rnn activation function. It can be
        one of tanh and relu.
      num_classes: an integer, the number of output classes/labels.
      use_bias: a boolean specifying whether to use bias in the last fc layer.
    """
    # Input variables
    input_data = tf.keras.layers.Input(
        shape=input_shape, name="features")

    # Two cnn layers
    conv_layer_1 = _conv_bn_layer(
        input_data, filters=32, kernel_size=(41, 11), strides=(2, 2),
        layer_id=1)

    conv_layer_2 = _conv_bn_layer(
        conv_layer_1, filters=32, kernel_size=(21, 11), strides=(2, 1),
        layer_id=2)
    # output of conv_layer2 with the shape of
    # [batch_size (N), times (T), features (F), channels (C)]

    # RNN layers.
    # Convert the conv output to rnn input
    rnn_input = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(
        conv_layer_2)

    rnn_cell = SUPPORTED_RNNS[rnn_type]
    for layer_counter in xrange(num_rnn_layers):
      # No batch normalization on the first layer
      is_batch_norm = (layer_counter != 0)
      rnn_input = _rnn_layer(
          rnn_input, rnn_cell, rnn_hidden_size, layer_counter + 1,
          rnn_activation, is_batch_norm, is_bidirectional)

    # FC layer with batch norm
    fc_input = tf.keras.layers.BatchNormalization(
        momentum=_MOMENTUM, epsilon=_EPSILON)(rnn_input)

    y_pred = tf.keras.layers.Dense(num_classes, activation="softmax",
                                   use_bias=use_bias, name="y_pred")(fc_input)

    # For ctc loss
    labels = tf.keras.layers.Input(name="labels", shape=[None,], dtype="int32")
    label_length = tf.keras.layers.Input(
        name="label_length", shape=[1], dtype="int32")
    input_length = tf.keras.layers.Input(
        name="input_length", shape=[1], dtype="int32")
    ctc_input_length = tf.keras.layers.Lambda(
        _calc_ctc_input_length, output_shape=(1,), name="ctc_input_length")(
            [input_length, input_data, y_pred])

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    ctc_loss = tf.keras.layers.Lambda(
        _ctc_lambda_func, output_shape=(1,), name="ctc_loss")(
            [y_pred, labels, ctc_input_length, label_length])

    super(DeepSpeech, self).__init__(
        inputs=[input_data, labels, input_length, label_length],
        outputs=[ctc_input_length, ctc_loss, y_pred])
