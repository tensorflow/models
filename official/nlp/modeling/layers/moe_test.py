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

"""Tests for moe.py."""

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import moe


def small_config():
  """Creates a small model config that can be used by all tests."""
  config = {}
  config['d_ff'] = 32
  config['output_dropout'] = 0.1

  config['num_experts'] = 2
  config['expert_d_ff'] = 33
  config['expert_dropout_rate'] = 0.1
  config['jitter_noise'] = 0.1
  config['train_capacity_factor'] = 1.0
  config['eval_capacity_factor'] = 1.0
  config['examples_per_group'] = 2.0

  config['backbone_d_ff'] = 13
  return config


def make_input_ones(batch_size: int = 4,
                    seq_length: int = 10,
                    hidden_dim: int = 7) -> tf.Tensor:
  return tf.ones((batch_size, seq_length, hidden_dim))


def make_experts_input_ones(num_groups: int = 1,
                            num_experts: int = 2,
                            expert_capacity: int = 5,
                            hidden_dim: int = 7) -> tf.Tensor:
  return tf.ones((num_groups, num_experts, expert_capacity, hidden_dim))


class MoeTest(tf.test.TestCase):

  def tearDown(self):
    super().tearDown()
    tf.keras.mixed_precision.set_global_policy('float32')

  def test_router_z_loss_dtype(self):
    x = tf.constant([[[10.0, 5.0]]], dtype=tf.float32)
    y = moe._router_z_loss(x)
    expected = (5 + np.log(np.exp(5) + 1))**2
    self.assertAllClose(expected, y, atol=1e-7)
    self.assertDTypeEqual(y, tf.float32)

  def test_router_z_loss_shape(self):
    x = make_input_ones(2, 5, 7)
    y = moe._router_z_loss(x)
    expected = (np.log(7) + 1)**2
    self.assertAllClose(expected, y, atol=1e-7)

  def test_experts_choose_masked_router_dtype_shape(self):
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    num_groups = 2
    tokens_per_group = 3
    hidden_dim = tokens_per_group
    num_experts = tokens_per_group
    expert_capacity = 2
    x = np.zeros([num_groups, tokens_per_group, hidden_dim])
    x[0, 0, 0] += 1
    x[0, :2, :2] += 1
    x[1, 1:, 1:] += 1
    x[1, -1, -1] += 1

    router = moe.ExpertsChooseMaskedRouter(
        num_experts=num_experts,
        jitter_noise=0.1,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.get('identity'),
        bias_initializer=tf.keras.initializers.get('ones'))
    router_mask = router(x, expert_capacity=expert_capacity, training=False)

    self.assertDTypeEqual(router_mask.dispatch_mask, tf.bfloat16)
    self.assertDTypeEqual(router_mask.combine_array, tf.bfloat16)

    expect_shape = [num_groups, tokens_per_group, num_experts, expert_capacity]
    self.assertEqual(expect_shape, router_mask.dispatch_mask.shape)
    self.assertEqual(expect_shape, router_mask.combine_array.shape)

    # top_k call may not be sorted, so can't compare the output directly
    # Check that the output contains only 0s and 1s
    out_dm = router_mask.dispatch_mask.numpy()
    self.assertSetEqual({0, 1}, set(out_dm.flatten().astype(np.int32)))
    # Check that the right tokens for selected
    out_dm_indices = np.dot(
        out_dm.transpose((0, 2, 3, 1)), np.arange(tokens_per_group))
    # Shape [num_groups, num_experts, expert_capacity]
    self.assertSetEqual({0, 1}, set(out_dm_indices[0, 0, :].astype(np.int32)))
    self.assertSetEqual({1, 2}, set(out_dm_indices[0, 1, :].astype(np.int32)))
    self.assertSetEqual({1, 2}, set(out_dm_indices[0, 2, :].astype(np.int32)))
    self.assertSetEqual({0, 1}, set(out_dm_indices[1, 0, :].astype(np.int32)))
    self.assertSetEqual({0, 1}, set(out_dm_indices[1, 1, :].astype(np.int32)))
    self.assertSetEqual({1, 2}, set(out_dm_indices[1, 2, :].astype(np.int32)))

    out_ca = router_mask.combine_array.numpy()
    out_ca = np.dot(out_ca, np.ones((expert_capacity,)))

    expected_combine_array = np.array([[[0.66, 0.0, 0.0], [0.42, 0.42, 0.16],
                                        [0.0, 0.33, 0.33]],
                                       [[0.33, 0.33, 0.0], [0.16, 0.42, 0.42],
                                        [0.0, 0.0, 0.66]]])
    self.assertAllClose(expected_combine_array, out_ca, atol=1e-2)

  def test_feed_forward_shape_and_vars(self):
    config = small_config()
    layer = moe.FeedForward(
        d_ff=config['d_ff'], output_dropout=config['output_dropout'])
    inputs = make_input_ones()
    outputs = layer(inputs)
    self.assertAllEqual(tf.shape(inputs), tf.shape(outputs))
    var_names = sorted([v.name for v in layer.trainable_variables])
    self.assertAllEqual([
        'feed_forward/intermediate/bias:0',
        'feed_forward/intermediate/kernel:0', 'feed_forward/output/bias:0',
        'feed_forward/output/kernel:0'
    ], var_names)

  def test_feed_forward_manual(self):
    config = small_config()
    layer = moe.FeedForward(
        d_ff=config['d_ff'],
        output_dropout=config['output_dropout'],
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.get('ones'),
        bias_initializer=tf.keras.initializers.get('ones'))
    inputs = make_input_ones(1, 2, 3)
    outputs = layer(inputs, training=False)
    manual_outputs = tf.constant([[[129.0, 129.0, 129.0], [129.0, 129.0,
                                                           129.0]]])
    self.assertAllClose(manual_outputs, outputs, atol=1e-7)

  def test_feed_forward_experts_shape_and_vars(self):
    config = small_config()
    layer = moe.FeedForwardExperts(
        num_experts=config['num_experts'],
        d_ff=config['expert_d_ff'],
        output_dropout=config['expert_dropout_rate'])
    inputs = make_experts_input_ones()
    outputs = layer(inputs)
    self.assertAllEqual(tf.shape(inputs), tf.shape(outputs))
    var_names = sorted([v.name for v in layer.trainable_variables])
    self.assertAllEqual([
        'experts/intermediate/bias:0', 'experts/intermediate/kernel:0',
        'experts/output/bias:0', 'experts/output/kernel:0'
    ], var_names)

  def test_feed_forward_experts_manual(self):
    config = small_config()
    layer = moe.FeedForwardExperts(
        num_experts=1,
        d_ff=config['expert_d_ff'],
        output_dropout=config['expert_dropout_rate'],
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.get('ones'),
        bias_initializer=tf.keras.initializers.get('ones'))
    inputs = make_experts_input_ones(1, 1, 2, 3)
    outputs = layer(inputs, training=False)
    manual_outputs = tf.constant([[[[133.0, 133.0, 133.0],
                                    [133.0, 133.0, 133.0]]]])
    self.assertAllClose(manual_outputs, outputs, atol=1e-7)

  def test_moe_layer(self):
    config = small_config()
    experts = moe.FeedForwardExperts(
        num_experts=config['num_experts'],
        d_ff=config['expert_d_ff'],
        output_dropout=config['expert_dropout_rate'])
    router = moe.ExpertsChooseMaskedRouter(
        config['num_experts'], jitter_noise=config['jitter_noise'])
    moe_layer = moe.MoeLayer(
        experts,
        router,
        train_capacity_factor=config['train_capacity_factor'],
        eval_capacity_factor=config['eval_capacity_factor'],
        examples_per_group=config['examples_per_group'])

    inputs = make_input_ones()
    outputs = moe_layer(inputs, training=True)
    self.assertAllEqual(tf.shape(inputs), tf.shape(outputs))

    var_names = sorted([v.name for v in moe_layer.trainable_variables])
    self.assertAllEqual([
        'moe/experts/intermediate/bias:0', 'moe/experts/intermediate/kernel:0',
        'moe/experts/output/bias:0', 'moe/experts/output/kernel:0',
        'moe/router/router_weights/bias:0', 'moe/router/router_weights/kernel:0'
    ], var_names)
    self.assertLen(moe_layer.losses, 1)
    metrics = [metric.name for metric in moe_layer.metrics]
    self.assertSetEqual(
        {
            'router_z_loss', 'unscaled_router_z_loss', 'load_balancing_loss',
            'fraction_tokens_left_behind', 'router_confidence', 'expert_usage'
        }, set(metrics))

  def test_moe_layer_with_backbone(self):
    config = small_config()
    experts = moe.FeedForwardExperts(
        num_experts=config['num_experts'],
        d_ff=config['expert_d_ff'],
        output_dropout=config['expert_dropout_rate'])
    router = moe.ExpertsChooseMaskedRouter(
        config['num_experts'], jitter_noise=config['jitter_noise'])
    moe_layer = moe.MoeLayer(
        experts,
        router,
        train_capacity_factor=config['train_capacity_factor'],
        eval_capacity_factor=config['eval_capacity_factor'],
        examples_per_group=config['examples_per_group'])
    layer = moe.MoeLayerWithBackbone(moe_layer, config['backbone_d_ff'])

    inputs = make_input_ones()
    outputs = layer(inputs)
    self.assertAllEqual(tf.shape(inputs), tf.shape(outputs))


if __name__ == '__main__':
  tf.test.main()
