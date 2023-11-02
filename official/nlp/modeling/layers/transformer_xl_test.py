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

"""Tests for Transformer XL."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations

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
    num_layers=1,
    include_biases=True,
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
    num_layers: `int`, the number of Transformer XL blocks.
    include_biases: optional `bool`, whether or not to include attention biases.
    include_state: optional `bool`, whether or not to include state data.
    include_mask: optional `bool`, whether or not to include mask data.
    include_segment: optional `bool`, whether or not to include segment data.

  Returns:
    A dictionary with `str` as keys and `Tensor` as values.
  """
  encoding_shape = (batch_size, seq_length * 2, hidden_size)

  data = dict(
      relative_position_encoding=tf.random.normal(shape=encoding_shape),
      content_stream=tf.random.normal(
          shape=(batch_size, seq_length, hidden_size)))

  if include_biases:
    attention_bias_shape = (num_heads, head_size)
    data.update(dict(
        content_attention_bias=tf.random.normal(shape=attention_bias_shape),
        segment_attention_bias=tf.random.normal(shape=attention_bias_shape),
        positional_attention_bias=tf.random.normal(shape=attention_bias_shape)))

  if two_stream:
    data.update(dict(
        query_stream=tf.random.normal(
            shape=(batch_size, num_predictions, hidden_size)),
        target_mapping=tf.random.normal(
            shape=(batch_size, num_predictions, seq_length))))

  if include_state:
    total_seq_length = seq_length + memory_length
    if num_layers > 1:
      state_shape = (num_layers, batch_size, memory_length, hidden_size)
    else:
      state_shape = (batch_size, memory_length, hidden_size)
    data.update(dict(
        state=tf.random.normal(shape=state_shape)))
  else:
    total_seq_length = seq_length

  if include_mask:
    mask_shape = (batch_size, num_heads, seq_length, total_seq_length)
    mask_data = np.random.randint(2, size=mask_shape).astype("float32")
    data["content_attention_mask"] = mask_data
    if two_stream:
      data["query_attention_mask"] = mask_data

  if include_segment:
    # A transformer XL block takes an individual segment "encoding" from the
    # entirety of the Transformer XL segment "embedding".
    if num_layers > 1:
      segment_encoding_shape = (num_layers, 2, num_heads, head_size)
      segment_encoding_name = "segment_embedding"
    else:
      segment_encoding_shape = (2, num_heads, head_size)
      segment_encoding_name = "segment_encoding"

    segment_matrix = np.random.randint(
        2, size=(batch_size, seq_length, total_seq_length))
    data["segment_matrix"] = tf.math.equal(segment_matrix, 1)
    data[segment_encoding_name] = tf.random.normal(shape=segment_encoding_shape)

  return data


class TransformerXLBlockTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      memory_length=[0, 4],
      two_stream=[True, False],
      state=[True, False],
      mask=[True, False],
      segment=[True, False]))
  def test_transformer_xl_block(
      self,
      two_stream,
      memory_length,
      state,
      mask,
      segment):
    """Tests combinations of Transformer XL block calculations."""
    batch_size, num_heads, head_size, seq_length = 2, 12, 64, 8
    hidden_size, num_predictions, inner_size = 24, 8, 12

    data = create_mock_transformer_xl_data(
        include_biases=True,
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


class TransformerXLTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      two_stream=[True, False],
      memory_length=[0, 4],
      reuse_length=[0, 4],
      tie_attention_biases=[True, False],
      state=[True, False],
      mask=[True, False],
      segment=[True, False]))
  def test_transformer_xl(
      self,
      two_stream,
      memory_length,
      reuse_length,
      tie_attention_biases,
      state,
      mask,
      segment):
    batch_size, num_heads, head_size, seq_length = 2, 12, 64, 8
    hidden_size, num_predictions, inner_size = 24, 8, 12
    num_layers = 3

    data = create_mock_transformer_xl_data(
        include_biases=False,
        num_heads=num_heads,
        head_size=head_size,
        hidden_size=hidden_size,
        seq_length=seq_length,
        batch_size=batch_size,
        memory_length=memory_length,
        num_predictions=num_predictions,
        two_stream=two_stream,
        num_layers=num_layers,
        include_state=state,
        include_mask=mask,
        include_segment=segment)
    transformer_xl_layer = transformer_xl.TransformerXL(
        vocab_size=32000,
        num_layers=num_layers,
        head_size=head_size,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        inner_size=inner_size,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        two_stream=two_stream,
        tie_attention_biases=tie_attention_biases,
        memory_length=memory_length,
        reuse_length=reuse_length,
        inner_activation="relu")
    attention_output, cached_memory_states = transformer_xl_layer(**data)
    if two_stream:
      self.assertEqual(attention_output.shape,
                       [batch_size, num_predictions, hidden_size])
    else:
      self.assertEqual(attention_output.shape,
                       [batch_size, seq_length, hidden_size])
    self.assertLen(cached_memory_states, num_layers)

  def test_get_config(self):
    transformer_xl_layer = transformer_xl.TransformerXL(
        vocab_size=32000,
        num_layers=12,
        hidden_size=36,
        head_size=12,
        num_attention_heads=12,
        inner_size=12,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        two_stream=False,
        tie_attention_biases=True,
        memory_length=0,
        reuse_length=0,
        inner_activation="relu")
    transformer_xl_config = transformer_xl_layer.get_config()
    new_transformer_xl = transformer_xl.TransformerXL.from_config(
        transformer_xl_config)
    self.assertEqual(transformer_xl_config, new_transformer_xl.get_config())


if __name__ == "__main__":
  np.random.seed(0)
  tf.random.set_seed(0)
  tf.test.main()
