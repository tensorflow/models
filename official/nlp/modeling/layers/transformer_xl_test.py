# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for Transformer XL."""

import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import

from official.nlp.modeling.layers import transformer_xl


def create_mock_transformer_xl_data(
    batch_size,
    num_heads,
    head_size,
    hidden_size,
    seq_length,
    memory_length=0,
    num_predictions=2,
    two_stream=False,
    include_state=False,
    include_mask=False,
    include_segment=False):
  """Creates mock testing data.

  Args:
    batch_size: `int`, the batch size.
    num_heads: `int`, number of attention heads.
    head_size: `int`, the size of each attention head.
    hidden_size: `int`, the layer's hidden size.
    seq_length: `int`, Sequence length of the input.
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
  encoding_shape = (batch_size, seq_length * 2, hidden_size)
  attention_bias_shape = (num_heads, head_size)

  data = dict(
      content_stream=tf.random.normal(
          shape=(batch_size, seq_length, hidden_size)),
      relative_position_encoding=tf.random.normal(shape=encoding_shape),
      content_attention_bias=tf.random.normal(shape=attention_bias_shape),
      positional_attention_bias=tf.random.normal(shape=attention_bias_shape))

  if two_stream:
    two_stream_data = dict(
        query_stream=tf.random.normal(
            shape=(batch_size, num_predictions, hidden_size)),
        target_mapping=tf.random.normal(
            shape=(batch_size, num_predictions, seq_length)))
    data.update(two_stream_data)

  if include_state:
    total_seq_length = seq_length + memory_length
    data["state"] = tf.random.normal(
        shape=(batch_size, memory_length, hidden_size))
  else:
    total_seq_length = seq_length

  if include_mask:
    mask_shape = (batch_size, num_heads, seq_length, total_seq_length)
    mask_data = np.random.randint(2, size=mask_shape).astype("float32")
    data["content_attention_mask"] = mask_data
    if two_stream:
      data["query_attention_mask"] = mask_data

  if include_segment:
    segment_encoding_shape = (2, num_heads, head_size)
    segment_matrix = np.random.randint(
        2, size=(batch_size, seq_length, total_seq_length))
    segment_matrix = tf.math.equal(segment_matrix, 1)
    segment_data = dict(
        segment_attention_bias=tf.random.normal(shape=attention_bias_shape),
        segment_encoding=tf.random.normal(shape=segment_encoding_shape),
        segment_matrix=segment_matrix)
    data.update(segment_data)

  return data


@keras_parameterized.run_all_keras_modes
class TransformerXLBlockTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(
      memory_length=[0, 4],
      two_stream=[True, False],
      state=[True, False],
      mask=[True, False],
      segment=[True, False]))
  def test_transformer_xl(self,
                          two_stream,
                          memory_length,
                          state,
                          mask,
                          segment):
    """Tests combinations of Transformer XL calculations."""
    batch_size, num_heads, head_size, seq_length = 2, 12, 64, 8
    hidden_size, num_predictions, inner_size = 24, 8, 12

    data = create_mock_transformer_xl_data(
        num_heads=num_heads,
        head_size=head_size,
        hidden_size=hidden_size,
        seq_length=seq_length,
        batch_size=batch_size,
        memory_length=memory_length,
        num_predictions=num_predictions,
        two_stream=two_stream,
        include_state=state,
        include_mask=mask,
        include_segment=segment)

    test_layer = transformer_xl.TransformerXLBlock(
        vocab_size=32000,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_size=head_size,
        inner_size=inner_size,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        two_stream=two_stream)
    output = test_layer(**data)
    content_attention = output["content_attention"]
    self.assertEqual(content_attention.shape,
                     [batch_size, seq_length, hidden_size])

    if two_stream:
      self.assertIn("query_attention", output)
      self.assertEqual(output["query_attention"].shape,
                       [batch_size, num_predictions, hidden_size])
    else:
      self.assertNotIn("query_attention", output)

  def test_get_config(self):
    transformer_xl_block = transformer_xl.TransformerXLBlock(
        vocab_size=32000,
        head_size=64,
        num_attention_heads=2,
        hidden_size=10,
        inner_size=50,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        two_stream=False)
    transformer_xl_block_config = transformer_xl_block.get_config()
    new_block = transformer_xl.TransformerXLBlock.from_config(
        transformer_xl_block_config)
    self.assertEqual(transformer_xl_block_config, new_block.get_config())


if __name__ == "__main__":
  np.random.seed(0)
  tf.random.set_seed(0)
  tf.test.main()
