# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers


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
    inputs = tf_keras.Input(
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

  def test_layerscale_call(self):
    # Set up test inputs
    input_shape = (2, 3, 4)
    init_values = 1e-4
    inputs = tf.ones(input_shape, dtype=tf.float32)

    # Instantiate LayerScale object
    layer_scale = nn_blocks.LayerScale(init_values)

    # Call LayerScale object on test inputs
    output = layer_scale(inputs)

    # Check output shape
    expected_output_shape = input_shape
    self.assertAllEqual(output.shape, expected_output_shape)

    # Check that output values are correct
    expected_output_values = init_values * np.ones(input_shape)
    self.assertAllClose(
        output.numpy(), expected_output_values, rtol=1e-5, atol=1e-5)

  def test_layerscale_training(self):
    # Verify that gamma values have changed from their initial values in one
    # step forward pass.
    # Set up test inputs
    input_shape = (1, 3, 4)
    init_values = 1e-4
    inputs = tf.ones(input_shape, dtype=tf.float32)
    targets = tf.ones(input_shape, dtype=tf.float32)

    # Instantiate LayerScale object
    layer_scale = nn_blocks.LayerScale(init_values)

    # Define optimizer and loss function
    optimizer = tf_keras.optimizers.Adam()
    loss_fn = tf_keras.losses.MeanSquaredError()

    # Train the model for one step
    with tf.GradientTape() as tape:
      predictions = layer_scale(inputs)
      loss = loss_fn(targets, predictions)
    grads = tape.gradient(loss, layer_scale.trainable_variables)
    optimizer.apply_gradients(zip(grads, layer_scale.trainable_variables))

    # Check that gamma values have changed
    updated_gamma = layer_scale.gamma.numpy()[0, 0, 0]
    self.assertNotEqual(updated_gamma, init_values)

  @parameterized.parameters(
      (nn_blocks.BottleneckBlock, 1, False, 0.0, None),
      (nn_blocks.BottleneckBlock, 2, True, 0.2, 0.25),
  )
  def test_bottleneck_block_creation(self, block_fn, strides, use_projection,
                                     stochastic_depth_drop_rate, se_ratio):
    input_size = 128
    filter_size = 256
    inputs = tf_keras.Input(
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
    inputs = tf_keras.Input(
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
      (2, True, 0, 5, 0, 12, 12, 2),
      (2, False, 5, 0, 0, 12, 18, 4),
      (1, True, 0, 0, 0, 12, 12, 6),
      (1, True, 3, 0, 0, 12, 18, 2),
      (1, True, 3, 3, 0, 12, 12, 4),
      (1, True, 3, 3, 3, 12, 18, 6),
      (1, True, 0, 3, 3, 12, 12, 2),
      (1, True, 0, 0, 3, 12, 18, 4),
      (1, True, 3, 0, 3, 12, 12, 6),
  )
  def test_universalinvertedbottleneck_block_creation(
      self,
      strides,
      middle_dw_downsample,
      start_dw_kernel_size,
      middle_dw_kernel_size,
      end_dw_kernel_size,
      in_filters,
      out_filters,
      expand_ratio,
  ):
    input_size = 128
    inputs = tf_keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1
    )
    block = nn_blocks.UniversalInvertedBottleneckBlock(
        in_filters=in_filters,
        out_filters=out_filters,
        expand_ratio=expand_ratio,
        strides=strides,
        middle_dw_downsample=middle_dw_downsample,
        start_dw_kernel_size=start_dw_kernel_size,
        middle_dw_kernel_size=middle_dw_kernel_size,
        end_dw_kernel_size=end_dw_kernel_size,
    )

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, out_filters],
        features.shape.as_list(),
    )

  @parameterized.parameters(
      (2, True, 0, 5, 0, 12, 12, 2, True, 0.1),
  )
  def test_universalinvertedbottleneck_block_layer_scale_creation(
      self,
      strides,
      middle_dw_downsample,
      start_dw_kernel_size,
      middle_dw_kernel_size,
      end_dw_kernel_size,
      in_filters,
      out_filters,
      expand_ratio,
      use_layer_scale,
      layer_scale_init_value,
  ):
    input_size = 128
    inputs = tf_keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1
    )
    block = nn_blocks.UniversalInvertedBottleneckBlock(
        in_filters=in_filters,
        out_filters=out_filters,
        expand_ratio=expand_ratio,
        strides=strides,
        middle_dw_downsample=middle_dw_downsample,
        start_dw_kernel_size=start_dw_kernel_size,
        middle_dw_kernel_size=middle_dw_kernel_size,
        end_dw_kernel_size=end_dw_kernel_size,
        use_layer_scale=use_layer_scale,
        layer_scale_init_value=layer_scale_init_value,
    )

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, out_filters],
        features.shape.as_list(),
    )

  @parameterized.parameters((False, 0, 3), (True, 3, 0))
  def test_universalinvertedbottleneck_inconsistent_downsampling(
      self, middle_dw_downsample, start_dw_kernel_size, middle_dw_kernel_size
  ):
    with self.assertRaises(ValueError):
      _ = nn_blocks.UniversalInvertedBottleneckBlock(
          in_filters=24,
          out_filters=24,
          expand_ratio=4,
          strides=2,
          middle_dw_downsample=middle_dw_downsample,
          start_dw_kernel_size=start_dw_kernel_size,
          middle_dw_kernel_size=middle_dw_kernel_size,
      )

  @parameterized.parameters(
      (nn_blocks.TuckerConvBlock, 1, 0.25, 0.25),
      (nn_blocks.TuckerConvBlock, 2, 0.25, 0.25),
  )
  def test_tucker_conv_block(self, block_fn, strides, input_compression_ratio,
                             output_compression_ratio):
    input_size = 128
    in_filters = 24
    out_filters = 24
    inputs = tf_keras.Input(
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
      optimizer = tf_keras.optimizers.SGD(learning_rate=0.01)

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
      optimizer = tf_keras.optimizers.SGD(learning_rate=0.01)

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
      # Clone all weights (tf_keras.layers.Layer has no .clone())
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


# Test class that wraps a standard attention layer. If this layer is called
# at any point, the list passed to the config object will be filled with a
# boolean 'True'. We register this class as a Keras serializable so we can
# test serialization below.
@tf_keras.utils.register_keras_serializable(package='TestOnlyAttention')
class ValidatedAttentionLayer(nn_layers.MultiHeadAttention):

  def __init__(self, call_list, **kwargs):
    super(ValidatedAttentionLayer, self).__init__(**kwargs)
    self.list = call_list

  def call(
      self,
      query,
      value,
      attention_mask=None,
      return_attention_scores=False,
  ):
    self.list.append(True)
    return super(ValidatedAttentionLayer, self).call(
        query,
        value,
        attention_mask=attention_mask,
        return_attention_scores=return_attention_scores)

  def get_config(self):
    config = super(ValidatedAttentionLayer, self).get_config()
    config['call_list'] = self.list
    return config


# Test class implements a simple feedforward layer. If this layer is called
# at any point, the list passed to the config object will be filled with a
# boolean 'True'. We register this class as a Keras serializable so we can
# test serialization below.
@tf_keras.utils.register_keras_serializable(package='TestOnlyFeedforward')
class ValidatedFeedforwardLayer(tf_keras.layers.Layer):

  def __init__(self, call_list, activation, **kwargs):
    super(ValidatedFeedforwardLayer, self).__init__(**kwargs)
    self.list = call_list
    self.activation = activation

  def build(self, input_shape):
    hidden_size = input_shape[-1]
    self._feedforward_dense = tf_keras.layers.EinsumDense(
        '...x,xy->...y',
        output_shape=hidden_size,
        bias_axes='y',
        activation=self.activation,
        name='feedforward')

  def call(self, inputs):
    self.list.append(True)
    return self._feedforward_dense(inputs)

  def get_config(self):
    config = super(ValidatedFeedforwardLayer, self).get_config()
    config['call_list'] = []
    config['activation'] = self.activation
    return config


class TransformerLayerTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(TransformerLayerTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy('float32')

  @parameterized.parameters(None, 2)
  def test_layer_creation(self, max_attention_inference_parallelism):
    sequence_length = 21
    width = 80

    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': []
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        max_attention_inference_parallelism=max_attention_inference_parallelism,
    )

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    call_list = test_layer._attention_layer.get_config()['call_list']
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_creation_with_feedforward_cls(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    feedforward_call_list = []
    feedforward_layer_cfg = {
        'activation': 'relu',
        'call_list': feedforward_call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        feedforward_cls=ValidatedFeedforwardLayer,
        feedforward_cfg=feedforward_layer_cfg,
        num_attention_heads=10,
        inner_dim=None,
        inner_activation=None)

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")
    self.assertNotEmpty(feedforward_call_list)
    self.assertTrue(feedforward_call_list[0],
                    "The passed layer class wasn't instantiated.")

  def test_layer_creation_with_mask(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  @parameterized.parameters(None, 2)
  def test_layer_invocation(self, max_attention_inference_parallelism):
    sequence_length = 21
    width = 80

    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': [],
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        max_attention_inference_parallelism=max_attention_inference_parallelism)

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)

    # Create a model from the test layer.
    model = tf_keras.Model(data_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    _ = model.predict(input_data)

    call_list = test_layer._attention_layer.get_config()['call_list']
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_invocation_with_feedforward_cls(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    feedforward_call_list = []
    feedforward_layer_cfg = {
        'activation': 'relu',
        'call_list': feedforward_call_list,
    }
    feedforward_layer = ValidatedFeedforwardLayer(**feedforward_layer_cfg)
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        feedforward_cls=feedforward_layer,
        num_attention_heads=10,
        inner_dim=None,
        inner_activation=None)

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")
    self.assertNotEmpty(feedforward_call_list)
    self.assertTrue(feedforward_call_list[0],
                    "The passed layer class wasn't instantiated.")

  def test_layer_invocation_with_mask(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_invocation_with_float16_dtype(self):
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = (10 * np.random.random_sample(
        (batch_size, sequence_length, width)))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_transform_with_initializer(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02))

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output.shape.as_list())
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0])

  def test_layer_restoration_from_config(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
        'name': 'test_layer',
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    pre_serialization_output = model.predict([input_data, mask_data])

    # Serialize the model config. Pass the serialized data through json to
    # ensure that we can serialize this layer to disk.
    serialized_data = model.get_config()

    # Create a new model from the old config, and copy the weights. These models
    # should have identical outputs.
    new_model = tf_keras.Model.from_config(serialized_data)
    new_model.set_weights(model.get_weights())
    output = new_model.predict([input_data, mask_data])

    self.assertAllClose(pre_serialization_output, output)

    # If the layer was configured correctly, it should have a list attribute
    # (since it should have the custom class and config passed to it).
    new_model.summary()
    new_call_list = new_model.get_layer(
        name='transformer_scaffold')._attention_layer.list
    self.assertNotEmpty(new_call_list)
    self.assertTrue(new_call_list[0],
                    "The passed layer class wasn't instantiated.")

  def test_layer_with_feedforward_cls_restoration_from_config(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_dim': 8,
        'call_list': call_list,
        'name': 'test_layer',
    }
    feedforward_call_list = []
    feedforward_layer_cfg = {
        'activation': 'relu',
        'call_list': feedforward_call_list,
    }
    test_layer = nn_blocks.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        feedforward_cls=ValidatedFeedforwardLayer,
        feedforward_cfg=feedforward_layer_cfg,
        num_attention_heads=10,
        inner_dim=None,
        inner_activation=None)

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    pre_serialization_output = model.predict([input_data, mask_data])

    serialized_data = model.get_config()
    # Create a new model from the old config, and copy the weights. These models
    # should have identical outputs.
    new_model = tf_keras.Model.from_config(serialized_data)
    new_model.set_weights(model.get_weights())
    output = new_model.predict([input_data, mask_data])

    self.assertAllClose(pre_serialization_output, output)

    # If the layer was configured correctly, it should have a list attribute
    # (since it should have the custom class and config passed to it).
    new_model.summary()
    new_call_list = new_model.get_layer(
        name='transformer_scaffold')._attention_layer.list
    self.assertNotEmpty(new_call_list)
    self.assertTrue(new_call_list[0],
                    "The passed layer class wasn't instantiated.")
    new_feedforward_call_list = new_model.get_layer(
        name='transformer_scaffold')._feedforward_block.list
    self.assertNotEmpty(new_feedforward_call_list)
    self.assertTrue(new_feedforward_call_list[0],
                    "The passed layer class wasn't instantiated.")

  @parameterized.parameters(
      (4, 64, 7, 8, 64, 64, 1, 1, 1),
      (4, 64, 7, None, 64, 64, 1, 1, 1),
      (4, 64, 7, 8, 64, 64, 2, 1, 1),
      (4, 64, 7, 8, 64, 32, 1, 2, 1),
      (4, 64, 7, 8, 64, 32, 1, 1, 2),
  )
  def test_multi_head_self_attention(
      self,
      batch_size,
      in_filters,
      cpe_dw_kernel_size,
      num_heads,
      key_dim,
      value_dim,
      query_h_strides,
      query_w_strides,
      kv_strides,
  ):
    input_size = 128
    inputs = tf_keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=batch_size
    )
    features = nn_blocks.MultiHeadSelfAttentionBlock(
        input_dim=in_filters,
        cpe_dw_kernel_size=cpe_dw_kernel_size,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        query_h_strides=query_h_strides,
        query_w_strides=query_w_strides,
        kv_strides=kv_strides,
    )(inputs)

    self.assertAllEqual(
        [batch_size, input_size, input_size, in_filters],
        features.shape.as_list(),
    )

  @parameterized.parameters(
      (10, 64, 48, 256, 8, 32, 16),
      (5, 32, 48, 8, 16, 16, 64),
  )
  def test_multi_query_attention_v1(
      self,
      batch_size,
      x_size,
      m_size,
      channel_dim,
      num_heads,
      key_dim,
      value_dim,
  ):
    layer = nn_blocks.MultiQueryAttentionLayerV1(num_heads, key_dim, value_dim)

    x_inputs = tf.random.uniform([batch_size, x_size, channel_dim])
    m_inputs = tf.random.uniform([batch_size, m_size, channel_dim])

    outputs = layer((x_inputs, m_inputs))
    self.assertAllEqual(
        [batch_size, x_size, channel_dim], outputs.shape.as_list()
    )

    opt_outputs = layer((x_inputs, m_inputs), optimize_einsum=True)
    self.assertAllEqual(
        [batch_size, x_size, channel_dim], opt_outputs.shape.as_list()
    )
    self.assertAllClose(outputs, opt_outputs)

  @parameterized.parameters(
      (10, 64, 48, 256, 8, 32, 16),
      (5, 32, 48, 8, 16, 16, 64),
  )
  def test_multi_query_attention_v2(
      self,
      batch_size,
      x_size,
      m_size,
      channel_dim,
      num_heads,
      key_dim,
      value_dim,
  ):
    layer = nn_blocks.MultiQueryAttentionLayerV2(num_heads, key_dim, value_dim)

    x_inputs = tf.random.uniform([batch_size, x_size, channel_dim])
    m_inputs = tf.random.uniform([batch_size, m_size, channel_dim])

    outputs = layer((x_inputs, m_inputs))
    self.assertAllEqual(
        [batch_size, x_size, channel_dim], outputs.shape.as_list()
    )

  @parameterized.parameters(
      (10, 32, 48, 8, 32, 16, 1, 1, 1),
      (10, 32, 48, 8, 32, 16, 1, 1, 2),
      (10, 32, 24, 8, 32, 16, 2, 1, 2),
      (10, 32, 24, 8, 32, 16, 2, 1, 1),
      (10, 32, 24, 8, 32, 16, 2, 2, 2),
  )
  def test_multi_query_attention_with_downsampling(
      self,
      batch_size,
      input_size,
      channel_dim,
      num_heads,
      key_dim,
      value_dim,
      query_h_strides,
      query_w_strides,
      kv_strides,
  ):
    inputs = tf.random.uniform(
        [batch_size, input_size, input_size, channel_dim]
    )

    layer = nn_blocks.OptimizedMultiQueryAttentionLayerWithDownSampling(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        query_h_strides=query_h_strides,
        query_w_strides=query_w_strides,
        kv_strides=kv_strides,
    )
    outputs = layer(inputs)
    self.assertAllEqual(
        [batch_size, input_size, input_size, channel_dim],
        outputs.shape.as_list(),
    )


if __name__ == '__main__':
  tf.test.main()
