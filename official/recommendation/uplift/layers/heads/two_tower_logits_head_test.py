# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for two_tower_logits_head."""

import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.layers.heads import two_tower_logits_head


class TwoTowerLogitsHeadTest(keras_test_case.KerasTestCase):

  def _get_layer(
      self,
      control_head=tf_keras.layers.Dense(1),
      treatment_head=tf_keras.layers.Dense(1),
      **kwargs
  ):
    logits_head = two_tower_logits_head.TwoTowerLogitsHead(
        control_head=control_head, treatment_head=treatment_head, **kwargs
    )
    return tf_keras.models.clone_model(logits_head)

  def test_layer_correctness(self):
    control_embedding = tf.ones((5, 10))
    treatment_embedding = tf.ones((5, 6))
    inputs = (control_embedding, treatment_embedding)
    layer = self._get_layer(
        control_head=tf_keras.layers.Dense(1, kernel_initializer="ones"),
        treatment_head=tf_keras.layers.Dense(1, kernel_initializer="ones"),
    )
    logits = layer(inputs)
    expected_logits = (10 * tf.ones((5, 1)), 6 * tf.ones((5, 1)))
    self.assertAllEqual(expected_logits, logits)

  def test_layer_stable(self):
    layer = self._get_layer()
    inputs = (tf.random.normal((10, 12)), tf.random.normal((10, 6)))
    self.assertLayerStable(inputs=inputs, layer=layer)

  def test_layer_savable(self):
    layer = self._get_layer(name="logits_head")
    inputs = (tf.random.normal((5, 10)), tf.random.normal((5, 6)))
    self.assertLayerSavable(inputs=inputs, layer=layer)

  def test_layer_configurable(self):
    layer = self._get_layer()
    self.assertLayerConfigurable(layer=layer)

  def test_different_logit_shapes_raises_error(self):
    layer = two_tower_logits_head.TwoTowerLogitsHead(
        control_head=tf_keras.layers.Dense(1),
        treatment_head=tf_keras.layers.Dense(2),
    )
    inputs = (tf.zeros(3, 1), tf.ones(3, 1))
    with self.assertRaises(ValueError):
      layer(inputs=inputs)


if __name__ == "__main__":
  tf.test.main()
