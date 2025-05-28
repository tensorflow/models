# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for nn_layers."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.yt8m.modeling import nn_layers


class NNLayersTest(parameterized.TestCase, tf.test.TestCase):
  """Class for testing nn_layers."""

  @parameterized.product(
      hidden_layer_size=(0, 8, 16),
      additive_residual=(True, False),
      pooling_method=["average", "max", "swap", "none", None],
  )
  def test_context_gate(
      self, hidden_layer_size, additive_residual, pooling_method
  ):
    """Test for creation of a context gate layer."""

    context_gate = nn_layers.ContextGate(
        normalizer_fn=tf_keras.layers.BatchNormalization,
        hidden_layer_size=hidden_layer_size,
        additive_residual=additive_residual,
        pooling_method=pooling_method,
    )

    if pooling_method is None:
      inputs = tf.ones([2, 32], dtype=tf.float32)
    elif pooling_method == "none":
      inputs = tf.ones([2, 1, 32], dtype=tf.float32)
    else:
      inputs = tf.ones([2, 24, 32], dtype=tf.float32)

    outputs = context_gate(inputs)
    self.assertShapeEqual(inputs, outputs)

    context_vars_len = 12 if hidden_layer_size else 6
    context_trainable_vars_len = 8 if hidden_layer_size else 4
    self.assertLen(context_gate.variables, context_vars_len)
    self.assertLen(context_gate.trainable_variables, context_trainable_vars_len)


if __name__ == "__main__":
  tf.test.main()
