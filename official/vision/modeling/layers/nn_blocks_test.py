# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for nn_blocks."""

from typing import Any, Iterable, Tuple
# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.modeling.layers import nn_blocks


def distribution_strategy_combinations() -> Iterable[Tuple[Any, ...]]:
  """Returns the combinations of end-to-end tests to run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],)


class NNBlocksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (nn_blocks.ResidualBlock, 1, False, 0.0, None),
      (nn_blocks.ResidualBlock, 2, True, 0.2, 0.25),
  )
  def test_residual_block_creation(self, block_fn, strides, use_projection,
                                   stochastic_depth_drop_rate, se_ratio):
    input_size = 128
    filter_size = 256
    inputs = tf.keras.Input(
        shape=(input_size, input_size, filter_size), batch_size=1)
    block = block_fn(
        filter_size,
        strides,
        use_projection=use_projection,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
    )

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, filter_size],
        features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.BottleneckBlock, 1, False, 0.0, None),
      (nn_blocks.BottleneckBlock, 2, True, 0.2, 0.25),
  )
  def test_bottleneck_block_creation(self, block_fn, strides, use_projection,
                                     stochastic_depth_drop_rate, se_ratio):
    input_size = 128
    filter_size = 256
    inputs = tf.keras.Input(
        shape=(input_size, input_size, filter_size * 4), batch_size=1)
    block = block_fn(
        filter_size,
        strides,
        use_projection=use_projection,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate)

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, filter_size * 4],
        features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.InvertedBottleneckBlock, 1, 1, None, None),
      (nn_blocks.InvertedBottleneckBlock, 6, 1, None, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 2, None, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 1, 0.2, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 1, None, 0.2),
  )
  def test_invertedbottleneck_block_creation(self, block_fn, expand_ratio,
                                             strides, se_ratio,
                                             stochastic_depth_drop_rate):
    input_size = 128
    in_filters = 24
    out_filters = 40
    inputs = tf.keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1)
    block = block_fn(
        in_filters=in_filters,
        out_filters=out_filters,
        expand_ratio=expand_ratio,
        strides=strides,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate)

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, out_filters],
        features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.TuckerConvBlock, 1, 0.25, 0.25),
      (nn_blocks.TuckerConvBlock, 2, 0.25, 0.25),
  )
  def test_tucker_conv_block(
      self, block_fn, strides,
      input_compression_ratio, output_compression_ratio):
    input_size = 128
    in_filters = 24
    out_filters = 24
    inputs = tf.keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1)
    block = block_fn(
        in_filters=in_filters,
        out_filters=out_filters,
        input_compression_ratio=input_compression_ratio,
        output_compression_ratio=output_compression_ratio,
        strides=strides)

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, out_filters],
        features.shape.as_list())


class ResidualInnerTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(distribution_strategy_combinations())
  def test_shape(self, distribution):
    bsz, h, w, c = 8, 32, 32, 32
    filters = 64
    strides = 2

    input_tensor = tf.random.uniform(shape=[bsz, h, w, c])
    with distribution.scope():
      test_layer = nn_blocks.ResidualInner(filters, strides)

    output = test_layer(input_tensor)
    expected_output_shape = [bsz, h // strides, w // strides, filters]
    self.assertEqual(expected_output_shape, output.shape.as_list())


class BottleneckResidualInnerTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(distribution_strategy_combinations())
  def test_shape(self, distribution):
    bsz, h, w, c = 8, 32, 32, 32
    filters = 64
    strides = 2

    input_tensor = tf.random.uniform(shape=[bsz, h, w, c])
    with distribution.scope():
      test_layer = nn_blocks.BottleneckResidualInner(filters, strides)

    output = test_layer(input_tensor)
    expected_output_shape = [bsz, h // strides, w // strides, filters * 4]
    self.assertEqual(expected_output_shape, output.shape.as_list())


class DepthwiseSeparableConvBlockTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(distribution_strategy_combinations())
  def test_shape(self, distribution):
    batch_size, height, width, num_channels = 8, 32, 32, 32
    num_filters = 64
    strides = 2

    input_tensor = tf.random.normal(
        shape=[batch_size, height, width, num_channels])
    with distribution.scope():
      block = nn_blocks.DepthwiseSeparableConvBlock(
          num_filters, strides=strides)
      config_dict = block.get_config()
      recreate_block = nn_blocks.DepthwiseSeparableConvBlock(**config_dict)

    output_tensor = block(input_tensor)
    expected_output_shape = [
        batch_size, height // strides, width // strides, num_filters
    ]
    self.assertEqual(output_tensor.shape.as_list(), expected_output_shape)

    output_tensor = recreate_block(input_tensor)
    self.assertEqual(output_tensor.shape.as_list(), expected_output_shape)


class ReversibleLayerTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(distribution_strategy_combinations())
  def test_downsampling_non_reversible_step(self, distribution):
    bsz, h, w, c = 8, 32, 32, 32
    filters = 64
    strides = 2

    input_tensor = tf.random.uniform(shape=[bsz, h, w, c])
    with distribution.scope():
      f = nn_blocks.ResidualInner(
          filters=filters // 2, strides=strides, batch_norm_first=True)
      g = nn_blocks.ResidualInner(
          filters=filters // 2, strides=1, batch_norm_first=True)
      test_layer = nn_blocks.ReversibleLayer(f, g)
      test_layer.build(input_tensor.shape)
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    @tf.function
    def step_fn():
      with tf.GradientTape() as tape:
        output = test_layer(input_tensor, training=True)
      grads = tape.gradient(output, test_layer.trainable_variables)
      # Test applying gradients with optimizer works
      optimizer.apply_gradients(zip(grads, test_layer.trainable_variables))

      return output

    replica_output = distribution.run(step_fn)
    outputs = distribution.experimental_local_results(replica_output)

    # Assert forward pass shape
    expected_output_shape = [bsz, h // strides, w // strides, filters]
    for output in outputs:
      self.assertEqual(expected_output_shape, output.shape.as_list())

  @combinations.generate(distribution_strategy_combinations())
  def test_reversible_step(self, distribution):
    # Reversible layers satisfy: (a) strides = 1 (b) in_filter = out_filter
    bsz, h, w, c = 8, 32, 32, 32
    filters = c
    strides = 1

    input_tensor = tf.random.uniform(shape=[bsz, h, w, c])
    with distribution.scope():
      f = nn_blocks.ResidualInner(
          filters=filters // 2, strides=strides, batch_norm_first=False)
      g = nn_blocks.ResidualInner(
          filters=filters // 2, strides=1, batch_norm_first=False)
      test_layer = nn_blocks.ReversibleLayer(f, g)
      test_layer(input_tensor, training=False)  # init weights
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    @tf.function
    def step_fn():
      with tf.GradientTape() as tape:
        output = test_layer(input_tensor, training=True)
      grads = tape.gradient(output, test_layer.trainable_variables)
      # Test applying gradients with optimizer works
      optimizer.apply_gradients(zip(grads, test_layer.trainable_variables))

      return output

    @tf.function
    def fwd():
      test_layer(input_tensor)

    distribution.run(fwd)  # Initialize variables
    prev_variables = tf.identity_n(test_layer.trainable_variables)
    replica_output = distribution.run(step_fn)
    outputs = distribution.experimental_local_results(replica_output)

    # Assert variables values have changed values
    for v0, v1 in zip(prev_variables, test_layer.trainable_variables):
      self.assertNotAllEqual(v0, v1)

    # Assert forward pass shape
    expected_output_shape = [bsz, h // strides, w // strides, filters]
    for output in outputs:
      self.assertEqual(expected_output_shape, output.shape.as_list())

  @combinations.generate(distribution_strategy_combinations())
  def test_manual_gradients_correctness(self, distribution):
    bsz, h, w, c = 8, 32, 32, 32
    filters = c
    strides = 1

    input_tensor = tf.random.uniform(shape=[bsz, h, w, c * 4])  # bottleneck
    with distribution.scope():
      f_manual = nn_blocks.BottleneckResidualInner(
          filters=filters // 2, strides=strides, batch_norm_first=False)
      g_manual = nn_blocks.BottleneckResidualInner(
          filters=filters // 2, strides=1, batch_norm_first=False)
      manual_grad_layer = nn_blocks.ReversibleLayer(f_manual, g_manual)
      manual_grad_layer(input_tensor, training=False)  # init weights

      f_auto = nn_blocks.BottleneckResidualInner(
          filters=filters // 2, strides=strides, batch_norm_first=False)
      g_auto = nn_blocks.BottleneckResidualInner(
          filters=filters // 2, strides=1, batch_norm_first=False)
      auto_grad_layer = nn_blocks.ReversibleLayer(
          f_auto, g_auto, manual_grads=False)
      auto_grad_layer(input_tensor)  # init weights
      # Clone all weights (tf.keras.layers.Layer has no .clone())
      auto_grad_layer._f.set_weights(manual_grad_layer._f.get_weights())
      auto_grad_layer._g.set_weights(manual_grad_layer._g.get_weights())

    @tf.function
    def manual_fn():
      with tf.GradientTape() as tape:
        output = manual_grad_layer(input_tensor, training=True)
      grads = tape.gradient(output, manual_grad_layer.trainable_variables)
      return grads

    @tf.function
    def auto_fn():
      with tf.GradientTape() as tape:
        output = auto_grad_layer(input_tensor, training=True)
      grads = tape.gradient(output, auto_grad_layer.trainable_variables)
      return grads

    manual_grads = distribution.run(manual_fn)
    auto_grads = distribution.run(auto_fn)

    # Assert gradients calculated manually are close to that from autograd
    for manual_grad, auto_grad in zip(manual_grads, auto_grads):
      self.assertAllClose(
          distribution.experimental_local_results(manual_grad),
          distribution.experimental_local_results(auto_grad),
          atol=5e-3,
          rtol=5e-3)

    # Verify that BN moving mean and variance is correct.
    for manual_var, auto_var in zip(manual_grad_layer.non_trainable_variables,
                                    auto_grad_layer.non_trainable_variables):
      self.assertAllClose(manual_var, auto_var)


if __name__ == '__main__':
  tf.test.main()
