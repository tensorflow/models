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

"""Tests for two_tower_logits_head."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.layers.heads import two_tower_logits_head


class TwoTowerLogitsHeadTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  def _get_layer(
      self,
      control_head=tf_keras.layers.Dense(1),
      treatment_head=tf_keras.layers.Dense(1),
      layering_method=two_tower_logits_head.LayeringMethod.NONE,
      linear_layering_config=two_tower_logits_head.LinearLayeringConfig(),
      **kwargs
  ):
    logits_head = two_tower_logits_head.TwoTowerLogitsHead(
        control_head=control_head,
        treatment_head=treatment_head,
        layering_config=two_tower_logits_head.LayeringConfig(
            layering_method=layering_method,
            linear_layering_config=linear_layering_config,
        ),
        **kwargs
    )
    return tf_keras.models.clone_model(logits_head)

  @parameterized.named_parameters(
      {
          "testcase_name": "layering_method_none",
          "layering_method": two_tower_logits_head.LayeringMethod.NONE,
          "expected_logits": (10 * tf.ones((5, 1)), 6 * tf.ones((5, 1))),
      },
      {
          "testcase_name": "layering_method_logit_sum",
          "layering_method": two_tower_logits_head.LayeringMethod.LOGIT_SUM,
          "expected_logits": (10 * tf.ones((5, 1)), (6 + 10) * tf.ones((5, 1))),
      },
      {
          "testcase_name": "layering_method_linear_layering",
          "layering_method": (
              two_tower_logits_head.LayeringMethod.LINEAR_LAYERING
          ),
          "expected_logits": (
              10 * tf.ones((5, 1)),
              ((10 + 1) * 6) * tf.ones((5, 1)),
          ),
          "linear_layering_config": two_tower_logits_head.LinearLayeringConfig(
              kernel_initializer="ones"
          ),
      },
  )
  def test_layer_correctness(
      self, layering_method, expected_logits, linear_layering_config=None
  ):
    control_embedding = tf.ones((5, 10))
    treatment_embedding = tf.ones((5, 6))
    inputs = (control_embedding, treatment_embedding)
    layer = self._get_layer(
        control_head=tf_keras.layers.Dense(1, kernel_initializer="ones"),
        treatment_head=tf_keras.layers.Dense(1, kernel_initializer="ones"),
        layering_method=layering_method,
        linear_layering_config=linear_layering_config,
    )
    logits = layer(inputs)
    self.assertAllEqual(expected_logits, logits)

  @parameterized.parameters(
      two_tower_logits_head.LayeringMethod.NONE,
      two_tower_logits_head.LayeringMethod.LOGIT_SUM,
      two_tower_logits_head.LayeringMethod.LINEAR_LAYERING,
  )
  def test_layer_stable(self, layering_method):
    layer = self._get_layer(layering_method=layering_method)
    inputs = (tf.random.normal((10, 12)), tf.random.normal((10, 6)))
    self.assertLayerStable(inputs=inputs, layer=layer)

  @parameterized.parameters(
      two_tower_logits_head.LayeringMethod.NONE,
      two_tower_logits_head.LayeringMethod.LOGIT_SUM,
      two_tower_logits_head.LayeringMethod.LINEAR_LAYERING,
  )
  def test_layer_savable(self, layering_method):
    layer = self._get_layer(layering_method=layering_method, name="logits_head")
    inputs = (tf.random.normal((5, 10)), tf.random.normal((5, 6)))
    self.assertLayerSavable(inputs=inputs, layer=layer)

  @parameterized.parameters(
      two_tower_logits_head.LayeringMethod.NONE,
      two_tower_logits_head.LayeringMethod.LOGIT_SUM,
      two_tower_logits_head.LayeringMethod.LINEAR_LAYERING,
  )
  def test_stop_gradient_does_not_update_control_tower(self, layering_method):
    linear_layering_config = None
    if layering_method == two_tower_logits_head.LayeringMethod.LINEAR_LAYERING:
      linear_layering_config = two_tower_logits_head.LinearLayeringConfig()

    # Build dummy model.
    inputs = tf_keras.Input(shape=(1,))
    control_embedding = tf_keras.layers.Dense(
        10, use_bias=False, kernel_initializer="ones", name="control_tower"
    )(inputs)
    treatment_embedding = tf_keras.layers.Dense(
        6, use_bias=False, kernel_initializer="ones", name="treatment_tower"
    )(inputs)
    logits = two_tower_logits_head.TwoTowerLogitsHead(
        control_head=tf_keras.layers.Dense(
            1, use_bias=False, kernel_initializer="ones", name="control_head"
        ),
        treatment_head=tf_keras.layers.Dense(
            1, use_bias=False, kernel_initializer="ones", name="treatment_head"
        ),
        layering_config=two_tower_logits_head.LayeringConfig(
            layering_method=layering_method,
            linear_layering_config=linear_layering_config,
        ),
        name="logits_head",
    )((control_embedding, treatment_embedding))
    model = tf_keras.Model(inputs=inputs, outputs=logits)

    def _get_kernel(name: str):
      if name == "control_head":
        return model.get_layer("logits_head")._control_head.weights[0]
      elif name == "treatment_head":
        return model.get_layer("logits_head")._treatment_head.weights[0]
      return model.get_layer(name=name).weights[0]

    self.assertAllClose(_get_kernel("control_tower"), tf.ones((1, 10)))
    self.assertAllClose(_get_kernel("treatment_tower"), tf.ones((1, 6)))
    self.assertAllClose(_get_kernel("control_head"), tf.ones((10, 1)))
    self.assertAllClose(_get_kernel("treatment_head"), tf.ones((6, 1)))

    # Train for one step on treatment examples only. The control weights should
    # remain unchanged due to the stop_gradient op.
    with tf.GradientTape() as tape:
      _, treatment_logits = model(tf.ones((3, 1)), training=True)
      loss = tf.reduce_mean(
          tf_keras.losses.mse(tf.ones((3, 1)), treatment_logits)
      )
    optimizer = tf_keras.optimizers.SGD(learning_rate=0.1)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    self.assertAllClose(_get_kernel("control_tower"), tf.ones((1, 10)))
    self.assertNotAllClose(_get_kernel("treatment_tower"), tf.ones((1, 6)))
    self.assertAllClose(_get_kernel("control_head"), tf.ones((10, 1)))
    self.assertNotAllClose(_get_kernel("treatment_head"), tf.ones((6, 1)))

  @parameterized.parameters(
      two_tower_logits_head.LayeringMethod.NONE,
      two_tower_logits_head.LayeringMethod.LOGIT_SUM,
      two_tower_logits_head.LayeringMethod.LINEAR_LAYERING,
  )
  def test_layer_configurable(self, layering_method):
    layer = self._get_layer(layering_method=layering_method)
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
