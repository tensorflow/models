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

"""Tests for tf_utils."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.modeling import tf_utils


def all_strategy_combinations():
  return combinations.combine(
      strategy=[
          strategy_combinations.cloud_tpu_strategy,
          # TODO(b/285797201):disable multi-gpu tests due to hanging.
          # strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode='eager',
  )


class TFUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_cross_replica_concat(self, strategy):
    num_cores = strategy.num_replicas_in_sync

    shape = (2, 3, 4)

    def concat(axis):

      @tf.function
      def function():
        replica_value = tf.fill(shape, tf_utils.get_replica_id())
        return tf_utils.cross_replica_concat(replica_value, axis=axis)

      return function

    def expected(axis):
      values = [np.full(shape, i) for i in range(num_cores)]
      return np.concatenate(values, axis=axis)

    per_replica_results = strategy.run(concat(axis=0))
    replica_0_result = per_replica_results.values[0].numpy()
    for value in per_replica_results.values[1:]:
      self.assertAllClose(value.numpy(), replica_0_result)
    self.assertAllClose(replica_0_result, expected(axis=0))

    replica_0_result = strategy.run(concat(axis=1)).values[0].numpy()
    self.assertAllClose(replica_0_result, expected(axis=1))

    replica_0_result = strategy.run(concat(axis=2)).values[0].numpy()
    self.assertAllClose(replica_0_result, expected(axis=2))

  @combinations.generate(all_strategy_combinations())
  def test_cross_replica_concat_gradient(self, strategy):
    num_cores = strategy.num_replicas_in_sync

    shape = (10, 5)

    @tf.function
    def function():
      replica_value = tf.random.normal(shape)
      with tf.GradientTape() as tape:
        tape.watch(replica_value)
        concat_value = tf_utils.cross_replica_concat(replica_value, axis=0)
        output = tf.reduce_sum(concat_value)
      return tape.gradient(output, replica_value)

    per_replica_gradients = strategy.run(function)
    for gradient in per_replica_gradients.values:
      self.assertAllClose(gradient, num_cores * tf.ones(shape))

  @parameterized.parameters(('relu', True), ('relu', False),
                            ('leaky_relu', False), ('leaky_relu', True),
                            ('mish', True), ('mish', False), ('gelu', True))
  def test_get_activations(self, name, use_keras_layer):
    fn = tf_utils.get_activation(name, use_keras_layer)
    self.assertIsNotNone(fn)

  @combinations.generate(all_strategy_combinations())
  def test_get_leaky_relu_layer(self, strategy):
    @tf.function
    def forward(x):
      fn = tf_utils.get_activation(
          'leaky_relu', use_keras_layer=True, alpha=0.1)
      return strategy.run(fn, args=(x,)).values[0]

    got = forward(tf.constant([-1]))
    self.assertAllClose(got, tf.constant([-0.1]))


if __name__ == '__main__':
  tf.test.main()
