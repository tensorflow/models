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

"""Tests for the attention layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.nlp.modeling.layers import relative_attention


def _create_mock_attention_data(
    num_heads,
    key_dim,
    value_dim,
    seq_length,
    batch_size,
    memory_length=0,
    num_predictions=2,
    two_stream=False,
    include_state=False,
    include_mask=False,
    include_segment=False):
  """Creates mock testing data.

  Args:
    num_heads: `int`, Number of attention heads.
    key_dim: `int`, Size of query head.
    value_dim: `int`, Size of key, value dim.
    seq_length: `int`, Sequence length of the input.
    batch_size: `int`, the batch size.
    memory_length: optional `int`, the length of the state. Defaults to 0.
    num_predictions: `int`, the number of predictions used in two stream
      attention.
    two_stream: `bool`, whether or not to generate two stream data.
    include_state: optional `bool`, whether or not to include state data.
    include_mask: optional `bool`, whether or not to include mask data.
    include_segment: optional `bool`, whether or not to include segment data.

  Returns:
    A dictionary with `str` as keys and `Tensor` as values.
  """
  query_shape = (batch_size, seq_length, key_dim)
  value_shape = (batch_size, seq_length, value_dim)
  encoding_shape = (batch_size, seq_length * 2, key_dim)
  attention_bias_shape = (num_heads, key_dim)

  data = dict(
      relative_position_encoding=tf.random.normal(shape=encoding_shape),
      content_attention_bias=tf.random.normal(shape=attention_bias_shape),
      positional_attention_bias=tf.random.normal(shape=attention_bias_shape))

  if two_stream:
    query_stream_shape = (batch_size, num_predictions, key_dim)
    target_mapping_shape = (batch_size, num_predictions, seq_length)
    stream_data = dict(
        content_stream=tf.random.normal(shape=query_shape),
        query_stream=tf.random.normal(shape=query_stream_shape),
        target_mapping=tf.random.normal(shape=target_mapping_shape))
  else:
    stream_data = dict(
        query=tf.random.normal(shape=query_shape),
        value=tf.random.normal(shape=value_shape),
        key=tf.random.normal(shape=value_shape))

  data.update(stream_data)

  if include_state:
    total_seq_length = seq_length + memory_length
    state_data = dict(
        state=tf.random.normal(shape=(batch_size, memory_length, value_dim)))
    data.update(state_data)
  else:
    total_seq_length = seq_length

  if include_mask:
    mask_shape = (batch_size, num_heads, seq_length, total_seq_length)
    mask_data = np.random.randint(2, size=mask_shape).astype("float32")
    if two_stream:
      mask_data = dict(
          content_attention_mask=mask_data,
          query_attention_mask=mask_data)
    else:
      mask_data = dict(attention_mask=mask_data)
    data.update(mask_data)

  if include_segment:
    segment_encoding_shape = (2, num_heads, key_dim)
    segment_matrix = np.random.randint(
        2, size=(batch_size, seq_length, total_seq_length))
    segment_matrix = tf.math.equal(segment_matrix, 1)
    segment_data = dict(
        segment_attention_bias=tf.random.normal(shape=attention_bias_shape),
        segment_encoding=tf.random.normal(shape=segment_encoding_shape),
        segment_matrix=segment_matrix)
    data.update(segment_data)

  return data


class MultiHeadRelativeAttentionTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      value_dim=[32, 64],
      memory_length=[0, 4],
      state=[True, False],
      mask=[True, False],
      segment=[True, False]))
  def test_attention_scores(self,
                            value_dim,
                            memory_length,
                            state,
                            mask,
                            segment):
    """Tests combinations of attention score calculations."""
    batch_size, num_heads, key_dim, seq_length = 2, 12, 64, 8
    test_layer = relative_attention.MultiHeadRelativeAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim)
    data = _create_mock_attention_data(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seq_length=seq_length,
        memory_length=memory_length,
        two_stream=False,
        batch_size=batch_size,
        include_state=state,
        include_mask=mask,
        include_segment=segment)
    output = test_layer(**data)
    self.assertEqual(output.shape, [batch_size, seq_length, key_dim])


class TwoStreamRelativeAttentionTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      num_predictions=[2, 10],
      memory_length=[0, 4],
      state=[True, False],
      mask=[True, False],
      segment=[True, False]))
  def test_attention_scores(self,
                            num_predictions,
                            memory_length,
                            state,
                            mask,
                            segment):
    """Tests combinations of attention score calculations."""
    batch_size, num_heads, key_dim, seq_length = 2, 12, 64, 8
    test_layer = relative_attention.TwoStreamRelativeAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=key_dim)
    data = _create_mock_attention_data(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=key_dim,
        seq_length=seq_length,
        memory_length=memory_length,
        num_predictions=num_predictions,
        two_stream=True,
        batch_size=batch_size,
        include_state=state,
        include_mask=mask,
        include_segment=segment)
    content_output, query_output, = test_layer(**data)
    self.assertEqual(content_output.shape, [batch_size, seq_length, key_dim])
    self.assertEqual(query_output.shape, [batch_size, num_predictions, key_dim])


if __name__ == "__main__":
  np.random.seed(0)
  tf.random.set_seed(0)
  tf.test.main()
