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

"""Tests for official.nlp.projects.kernel.attention."""
import itertools

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.modeling.layers import kernel_attention as attention


_FEATURE_TRANSFORM = ["relu", "elu", "exp", "expplus"]
_REDRAW = [True, False]
_TRAINING = [True, False]
_IS_SHORT_SEQ = [True, False]
_BEGIN_KERNEL = [0, 512]


class KernelAttentionTest(tf.test.TestCase, parameterized.TestCase):

  # expplus is only designed for bi-directional use case.
  # exp can be numeric unstable.
  @parameterized.parameters(itertools.product(
      ["relu", "elu"], [1, 4], [0.9]))
  def test_causal_windowed_attention_projection_streaming(
      self, feature_transform, causal_chunk_length, causal_weight_decay):
    num_heads = 12
    key_dim = 64
    seq_length = 16
    num_chunks = seq_length // causal_chunk_length
    causal_window_length = num_chunks
    batch_size = 2
    training = False
    num_random_features = 0
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        feature_transform=feature_transform,
        num_random_features=num_random_features,
        redraw=False,
        is_short_seq=False,
        begin_kernel=False,
        use_causal_windowed=True,
        causal_chunk_length=causal_chunk_length,
        causal_window_length=causal_window_length,
        causal_window_decay=causal_weight_decay,
        causal_padding=None,
        )
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim), seed=2)
    value = query
    encoder_inputs_mask = tf.ones((batch_size, seq_length), dtype=tf.int32)
    masks = tf.cast(encoder_inputs_mask, dtype=tf.float32)
    output = test_layer(
        query=query,
        value=value,
        attention_mask=masks,
        training=training)
    dim = num_random_features if num_random_features > 0 else key_dim
    kv_cache = tf.zeros(
        (batch_size, num_heads, dim, dim))
    k_sum_cache = tf.zeros((batch_size, num_heads, dim))
    stream_output = []
    cache = {"kv": kv_cache, "k_sum": k_sum_cache}
    for i in range(num_chunks):
      stream_output.append(
          test_layer(
              query=query[:, i * causal_chunk_length:(i + 1) *
                          causal_chunk_length, :],
              value=value[:, i * causal_chunk_length:(i + 1) *
                          causal_chunk_length, :],
              attention_mask=masks[:, i * causal_chunk_length:(i + 1) *
                                   causal_chunk_length],
              cache=cache,
              training=training))
    stream_output = tf.concat(stream_output, axis=1)
    self.assertAllClose(output, stream_output)

  @parameterized.parameters(
      itertools.product(_FEATURE_TRANSFORM, [127], _TRAINING, [True, False],
                        _IS_SHORT_SEQ, _BEGIN_KERNEL))
  def test_attention_projection(
      self, feature_transform, num_random_features, training, redraw, is_short,
      begin_kernel):
    num_heads = 12
    key_dim = 64
    seq_length = 1024
    batch_size = 2
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        feature_transform=feature_transform,
        num_random_features=num_random_features,
        redraw=redraw,
        is_short_seq=is_short,
        begin_kernel=begin_kernel)
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim))
    value = query
    encoder_inputs_mask = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    masks = tf.cast(encoder_inputs_mask, dtype=tf.float32)
    output = test_layer(
        query=query,
        value=value,
        attention_mask=masks,
        training=training)
    self.assertEqual(output.shape, [batch_size, seq_length, key_dim])

  @parameterized.parameters(
      itertools.product(["relu", "exp"], [127], _TRAINING, [True, False],
                        [0], [None, 0.97], [None, "left", "right"]))
  def test_causal_windowed_attention_projection(
      self, feature_transform, num_random_features, training, redraw,
      begin_kernel, causal_window_decay, causal_padding):
    num_heads = 12
    key_dim = 64
    seq_length = 1024
    batch_size = 2
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        feature_transform=feature_transform,
        num_random_features=num_random_features,
        redraw=redraw,
        is_short_seq=False,
        begin_kernel=begin_kernel,
        use_causal_windowed=True,
        causal_chunk_length=8,
        causal_window_length=3,
        causal_window_decay=causal_window_decay,
        causal_padding=causal_padding)
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim))
    value = query
    encoder_inputs_mask = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    masks = tf.cast(encoder_inputs_mask, dtype=tf.float32)
    output = test_layer(
        query=query,
        value=value,
        attention_mask=masks,
        training=training)
    self.assertEqual(output.shape, [batch_size, seq_length, key_dim])

  @parameterized.parameters(itertools.product(
      _FEATURE_TRANSFORM, [0], _TRAINING, [False],
      _IS_SHORT_SEQ, _BEGIN_KERNEL))
  def test_attention_no_projection(
      self, feature_transform, num_random_features, training, redraw, is_short,
      begin_kernel):
    num_heads = 12
    key_dim = 64
    seq_length = 1024
    batch_size = 2
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        feature_transform=feature_transform,
        num_random_features=num_random_features,
        redraw=redraw,
        is_short_seq=is_short,
        begin_kernel=begin_kernel)
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim))
    value = query
    encoder_inputs_mask = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    masks = tf.cast(encoder_inputs_mask, dtype=tf.float32)
    output = test_layer(
        query=query,
        value=value,
        attention_mask=masks,
        training=training)
    self.assertEqual(output.shape, [batch_size, seq_length, key_dim])

  @parameterized.parameters([128, 512])
  def test_attention_scale_by_length(self, seq_length):
    num_heads = 12
    key_dim = 64
    batch_size = 2
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        num_random_features=0,
        scale_by_length=True)
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim))
    value = query
    encoder_inputs_mask = tf.ones((batch_size, seq_length), dtype=tf.int32)
    masks = tf.cast(encoder_inputs_mask, dtype=tf.float32)
    output_scale_by_length = test_layer(
        query=query, value=value, attention_mask=masks)

    test_layer._scale_by_length = False
    output_no_scale_by_length = test_layer(
        query=query, value=value, attention_mask=masks)
    if seq_length == 512:  # Equals because log(seq_length, base=512) = 1.0
      self.assertAllClose(output_scale_by_length, output_no_scale_by_length)
    else:
      self.assertNotAllClose(output_scale_by_length, output_no_scale_by_length)

  def test_unsupported_feature_transform(self):
    with self.assertRaisesRegex(ValueError, "Unsupported feature_transform.*"):
      _ = attention.KernelAttention(feature_transform="test")

  def test_redraw_true_no_projection(self):
    with self.assertRaisesRegex(
        ValueError, "There is nothing to redraw when num_random_features.*"):
      _ = attention.KernelAttention(
          num_heads=2, key_dim=64, feature_transform="elu",
          num_random_features=0, redraw=True)

  def test_config(self):
    num_heads = 12
    key_dim = 64
    test_layer = attention.KernelAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        feature_transform="exp",
        num_random_features=128,
        is_short_seq=True)
    new_layer = attention.KernelAttention.from_config(
        test_layer.get_config())
    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())

  def test_rectangular_window_sum(self):
    x = tf.ones([2, 5, 2, 2, 2])
    winsum = attention.rectangular_window_sum(x, 3)
    self.assertEqual(winsum.shape, x.shape)
    self.assertAllClose(
        tf.tile(
            tf.reshape([1., 2., 3., 3., 3.], [1, -1, 1, 1, 1]),
            [2, 1, 2, 2, 2]),
        winsum)

  def test_weighted_window_sum(self):
    x = tf.ones([2, 5, 2, 2, 2])
    winsum = attention.weighted_window_sum(x, 3, [0.01, 0.1, 1.])
    self.assertEqual(winsum.shape, x.shape)
    self.assertAllClose(
        tf.tile(
            tf.reshape([1., 1.1, 1.11, 1.11, 1.11], [1, -1, 1, 1, 1]),
            [2, 1, 2, 2, 2]),
        winsum)

if __name__ == "__main__":
  tf.test.main()
