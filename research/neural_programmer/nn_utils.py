# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Author: aneelakantan (Arvind Neelakantan)
"""

import tensorflow as tf

def get_embedding(word, utility, params):
  return tf.nn.embedding_lookup(params["word"], word)


def apply_dropout(x, dropout_rate, mode):
  if (dropout_rate > 0.0):
    if (mode == "train"):
      x = tf.nn.dropout(x, dropout_rate)
    else:
      x = x
  return x


def LSTMCell(x, mprev, cprev, key, params):
  """Create an LSTM cell.

  Implements the equations in pg.2 from
  "Long Short-Term Memory Based Recurrent Neural Network Architectures
  For Large Vocabulary Speech Recognition",
  Hasim Sak, Andrew Senior, Francoise Beaufays.

  Args:
    w: A dictionary of the weights and optional biases as returned
      by LSTMParametersSplit().
    x: Inputs to this cell.
    mprev: m_{t-1}, the recurrent activations (same as the output)
      from the previous cell.
    cprev: c_{t-1}, the cell activations from the previous cell.
    keep_prob: Keep probability on the input and the outputs of a cell.

  Returns:
    m: Outputs of this cell.
    c: Cell Activations.
    """

  i = tf.matmul(x, params[key + "_ix"]) + tf.matmul(mprev, params[key + "_im"])
  i = tf.nn.bias_add(i, params[key + "_i"])
  f = tf.matmul(x, params[key + "_fx"]) + tf.matmul(mprev, params[key + "_fm"])
  f = tf.nn.bias_add(f, params[key + "_f"])
  c = tf.matmul(x, params[key + "_cx"]) + tf.matmul(mprev, params[key + "_cm"])
  c = tf.nn.bias_add(c, params[key + "_c"])
  o = tf.matmul(x, params[key + "_ox"]) + tf.matmul(mprev, params[key + "_om"])
  o = tf.nn.bias_add(o, params[key + "_o"])
  i = tf.sigmoid(i, name="i_gate")
  f = tf.sigmoid(f, name="f_gate")
  o = tf.sigmoid(o, name="o_gate")
  c = f * cprev + i * tf.tanh(c)
  m = o * c
  return m, c
