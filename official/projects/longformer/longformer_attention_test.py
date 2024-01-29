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

"""Tests for official.nlp.projects.longformer.longformer_attention."""

import numpy as np
import tensorflow as tf

from official.modeling.tf_utils import get_shape_list
from official.projects.longformer import longformer_attention


def _create_mock_attention_data(num_heads,
                                key_dim,
                                value_dim,
                                q_seq_length,
                                kv_seq_length,
                                batch_size,
                                include_mask=False):
  """Creates mock testing data.

  Args:
    num_heads: `int`, Number of attention heads.
    key_dim: `int`, Size of query head.
    value_dim: `int`, Size of key, value dim.
    q_seq_length: `int`, query sequence length of the input.
    kv_seq_length: `int`, key, value sequence length of the input.
    batch_size: `int`, the batch size.
    include_mask: optional `bool`, whether or not to include mask data.

  Returns:
    A dictionary with `str` as keys and `Tensor` as values.
  """
  query_shape = (batch_size, q_seq_length, key_dim)
  value_shape = (batch_size, kv_seq_length, value_dim)

  data = dict(
      query=tf.random.normal(shape=query_shape),
      value=tf.random.normal(shape=value_shape),
      key=tf.random.normal(shape=value_shape))

  total_seq_length = kv_seq_length

  if include_mask:
    mask_shape = (batch_size, num_heads, q_seq_length, total_seq_length)
    mask_data = np.random.randint(2, size=mask_shape).astype('float32')
    mask_data = dict(attention_mask=mask_data)
    data.update(mask_data)

  return data


class LongformerAttentionTest(tf.test.TestCase):

  def setUp(self):
    super(LongformerAttentionTest, self).setUp()
    np.random.seed(0)
    tf.random.set_seed(0)

  def _get_hidden_states(self):
    return tf.convert_to_tensor(
        [[
            [
                4.98332758e-01,
                2.69175139e00,
                -7.08081422e-03,
                1.04915401e00,
                -1.83476661e00,
                7.67220476e-01,
                2.98580543e-01,
                2.84803992e-02,
            ],
            [
                -7.58357372e-01,
                4.20635998e-01,
                -4.04739919e-02,
                1.59924145e-01,
                2.05135748e00,
                -1.15997978e00,
                5.37166397e-01,
                2.62873606e-01,
            ],
            [
                -1.69438001e00,
                4.17574660e-01,
                -1.49196962e00,
                -1.76483717e00,
                -1.94566312e-01,
                -1.71183858e00,
                7.72903565e-01,
                -1.11557056e00,
            ],
            [
                5.44028163e-01,
                2.05466114e-01,
                -3.63045868e-01,
                2.41865062e-01,
                3.20348382e-01,
                -9.05611176e-01,
                -1.92690727e-01,
                -1.19917547e00,
            ],
        ]],
        dtype=tf.float32,
    )

  def test_diagonalize(self):
    hidden_states = self._get_hidden_states()
    hidden_states = tf.reshape(hidden_states,
                               (1, 8, 4))  # set seq length = 8, hidden dim = 4
    chunked_hidden_states = longformer_attention.LongformerAttention._chunk(
        hidden_states, window_overlap=2)
    window_overlap_size = get_shape_list(chunked_hidden_states)[2]
    self.assertEqual(window_overlap_size, 4)

    padded_hidden_states = longformer_attention.LongformerAttention._pad_and_diagonalize(
        chunked_hidden_states)

    self.assertEqual(
        get_shape_list(padded_hidden_states)[-1],
        get_shape_list(chunked_hidden_states)[-1] + window_overlap_size - 1)

    # first row => [0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000]
    tf.debugging.assert_near(
        padded_hidden_states[0, 0, 0, :4],
        chunked_hidden_states[0, 0, 0],
        rtol=1e-3)
    tf.debugging.assert_near(
        padded_hidden_states[0, 0, 0, 4:],
        tf.zeros((3,), dtype=tf.dtypes.float32),
        rtol=1e-3)

    # last row => [0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629]
    tf.debugging.assert_near(
        padded_hidden_states[0, 0, -1, 3:],
        chunked_hidden_states[0, 0, -1],
        rtol=1e-3)
    tf.debugging.assert_near(
        padded_hidden_states[0, 0, -1, :3],
        tf.zeros((3,), dtype=tf.dtypes.float32),
        rtol=1e-3)

  def test_pad_and_transpose_last_two_dims(self):
    hidden_states = self._get_hidden_states()
    self.assertTrue(get_shape_list(hidden_states), [1, 8, 4])

    # pad along seq length dim
    paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]],
                           dtype=tf.dtypes.int32)

    hidden_states = longformer_attention.LongformerAttention._chunk(
        hidden_states, window_overlap=2)
    padded_hidden_states = longformer_attention.LongformerAttention._pad_and_transpose_last_two_dims(
        hidden_states, paddings)
    self.assertEqual(get_shape_list(padded_hidden_states), [1, 1, 8, 5])

    expected_added_dim = tf.zeros((5,), dtype=tf.dtypes.float32)
    tf.debugging.assert_near(
        expected_added_dim, padded_hidden_states[0, 0, -1, :], rtol=1e-6)
    tf.debugging.assert_near(
        hidden_states[0, 0, -1, :],
        tf.reshape(padded_hidden_states, (1, -1))[0, 24:32],
        rtol=1e-6)

  def test_mask_invalid_locations(self):
    hidden_states = self._get_hidden_states()
    batch_size = 1
    seq_length = 8
    hidden_size = 4
    hidden_states = tf.reshape(hidden_states,
                               (batch_size, seq_length, hidden_size))
    hidden_states = longformer_attention.LongformerAttention._chunk(
        hidden_states, window_overlap=2)

    hid_states_1 = longformer_attention.LongformerAttention._mask_invalid_locations(
        hidden_states, 1)
    hid_states_2 = longformer_attention.LongformerAttention._mask_invalid_locations(
        hidden_states, 2)
    hid_states_3 = longformer_attention.LongformerAttention._mask_invalid_locations(
        hidden_states[:, :, :, :3], 2)
    hid_states_4 = longformer_attention.LongformerAttention._mask_invalid_locations(
        hidden_states[:, :, 2:, :], 2)

    self.assertEqual(
        tf.math.reduce_sum(
            tf.cast(tf.math.is_inf(hid_states_1), tf.dtypes.int32)), 8)
    self.assertEqual(
        tf.math.reduce_sum(
            tf.cast(tf.math.is_inf(hid_states_2), tf.dtypes.int32)), 24)
    self.assertEqual(
        tf.math.reduce_sum(
            tf.cast(tf.math.is_inf(hid_states_3), tf.dtypes.int32)), 24)
    self.assertEqual(
        tf.math.reduce_sum(
            tf.cast(tf.math.is_inf(hid_states_4), tf.dtypes.int32)), 12)

  def test_chunk(self):
    hidden_states = self._get_hidden_states()
    batch_size = 1
    seq_length = 8
    hidden_size = 4
    hidden_states = tf.reshape(hidden_states,
                               (batch_size, seq_length, hidden_size))

    chunked_hidden_states = longformer_attention.LongformerAttention._chunk(
        hidden_states, window_overlap=2)

    # expected slices across chunk and seq length dim
    expected_slice_along_seq_length = tf.convert_to_tensor(
        [0.4983, -0.7584, -1.6944], dtype=tf.dtypes.float32)
    expected_slice_along_chunk = tf.convert_to_tensor(
        [0.4983, -1.8348, -0.7584, 2.0514], dtype=tf.dtypes.float32)

    self.assertEqual(get_shape_list(chunked_hidden_states), [1, 3, 4, 4])
    tf.debugging.assert_near(
        chunked_hidden_states[0, :, 0, 0],
        expected_slice_along_seq_length,
        rtol=1e-3)
    tf.debugging.assert_near(
        chunked_hidden_states[0, 0, :, 0],
        expected_slice_along_chunk,
        rtol=1e-3)

  def test_layer_local_attn(self):
    hidden_states = self._get_hidden_states()
    batch_size, seq_length, _ = hidden_states.shape
    layer = longformer_attention.LongformerAttention(
        num_heads=2,
        key_dim=4,
        value_dim=4,
        layer_id=0,
        attention_window=4,
        global_attention_size=0,
    )

    attention_mask = tf.zeros((batch_size, seq_length), dtype=tf.dtypes.float32)
    is_index_global_attn = tf.math.greater(attention_mask, 1)

    attention_mask = tf.where(
        tf.range(4)[None, :, None, None] > 1, -10000.0,
        attention_mask[:, :, None, None])
    is_index_masked = tf.math.less(attention_mask[:, :, 0, 0], 0)

    output_hidden_states = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        is_index_masked=is_index_masked,
        is_index_global_attn=is_index_global_attn,
    )[0]

    self.assertTrue(output_hidden_states.shape, (1, 4, 8))

  def test_layer_global_attn(self):
    layer = longformer_attention.LongformerAttention(
        num_heads=2,
        key_dim=4,
        value_dim=4,
        layer_id=0,
        attention_window=4,
        global_attention_size=1,
    )
    hidden_states = self._get_hidden_states()

    hidden_states = tf.concat(
        [self._get_hidden_states(),
         self._get_hidden_states() - 0.5], axis=0)
    _, seq_length, _ = hidden_states.shape

    # create attn mask
    attention_mask_1 = tf.zeros((1, 1, 1, seq_length), dtype=tf.dtypes.float32)
    attention_mask_2 = tf.zeros((1, 1, 1, seq_length), dtype=tf.dtypes.float32)

    attention_mask_1 = tf.where(
        tf.range(4)[None, :, None, None] == 0, 10000.0, attention_mask_1)
    attention_mask_1 = tf.where(
        tf.range(4)[None, :, None, None] > 2, -10000.0, attention_mask_1)
    attention_mask_2 = tf.where(
        tf.range(4)[None, :, None, None] == 0, 10000.0, attention_mask_2)
    attention_mask = tf.concat([attention_mask_1, attention_mask_2], axis=0)

    is_index_masked = tf.math.less(attention_mask[:, :, 0, 0], 0)
    is_index_global_attn = tf.math.greater(attention_mask[:, :, 0, 0], 0)

    output_hidden_states = layer(
        hidden_states=hidden_states,
        attention_mask=-tf.math.abs(attention_mask),
        is_index_masked=is_index_masked,
        is_index_global_attn=is_index_global_attn,
    )[0]

    self.assertTrue(output_hidden_states.shape, (2, 4, 8))


if __name__ == '__main__':
  tf.test.main()
