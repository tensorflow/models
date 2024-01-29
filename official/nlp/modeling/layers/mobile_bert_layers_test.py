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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import mobile_bert_layers
from official.nlp.modeling.networks import mobile_bert_encoder


def generate_fake_input(batch_size=1, seq_len=5, vocab_size=10000, seed=0):
  """Generate consistent fake integer input sequences."""
  np.random.seed(seed)
  fake_input = []
  for _ in range(batch_size):
    fake_input.append([])
    for _ in range(seq_len):
      fake_input[-1].append(np.random.randint(0, vocab_size))
  fake_input = np.asarray(fake_input)
  return fake_input


class MobileBertEncoderTest(parameterized.TestCase, tf.test.TestCase):

  def test_embedding_layer_with_token_type(self):
    layer = mobile_bert_layers.MobileBertEmbedding(10, 8, 2, 16)
    input_seq = tf.Variable([[2, 3, 4, 5]])
    token_type = tf.Variable([[0, 1, 1, 1]])
    output = layer(input_seq, token_type)
    output_shape = output.shape.as_list()
    expected_shape = [1, 4, 16]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_embedding_layer_without_token_type(self):
    layer = mobile_bert_layers.MobileBertEmbedding(10, 8, 2, 16)
    input_seq = tf.Variable([[2, 3, 4, 5]])
    output = layer(input_seq)
    output_shape = output.shape.as_list()
    expected_shape = [1, 4, 16]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_embedding_layer_get_config(self):
    layer = mobile_bert_layers.MobileBertEmbedding(
        word_vocab_size=16,
        word_embed_size=32,
        type_vocab_size=4,
        output_embed_size=32,
        max_sequence_length=32,
        normalization_type='layer_norm',
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        dropout_rate=0.5)
    layer_config = layer.get_config()
    new_layer = mobile_bert_layers.MobileBertEmbedding.from_config(layer_config)
    self.assertEqual(layer_config, new_layer.get_config())

  def test_no_norm(self):
    layer = mobile_bert_layers.NoNorm()
    feature = tf.random.normal([2, 3, 4])
    output = layer(feature)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 4]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  @parameterized.named_parameters(('with_kq_shared_bottleneck', False),
                                  ('without_kq_shared_bottleneck', True))
  def test_transfomer_kq_shared_bottleneck(self, is_kq_shared):
    feature = tf.random.uniform([2, 3, 512])
    layer = mobile_bert_layers.MobileBertTransformer(
        key_query_shared_bottleneck=is_kq_shared)
    output = layer(feature)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 512]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_transfomer_with_mask(self):
    feature = tf.random.uniform([2, 3, 512])
    input_mask = [[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]],
                  [[0., 1., 1.], [0., 1., 1.], [0., 1., 1.]]]
    input_mask = np.asarray(input_mask)
    layer = mobile_bert_layers.MobileBertTransformer()
    output = layer(feature, input_mask)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 512]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_transfomer_return_attention_score(self):
    sequence_length = 5
    num_attention_heads = 8
    feature = tf.random.uniform([2, sequence_length, 512])
    layer = mobile_bert_layers.MobileBertTransformer(
        num_attention_heads=num_attention_heads)
    _, attention_score = layer(feature, return_attention_scores=True)
    expected_shape = [2, num_attention_heads, sequence_length, sequence_length]
    self.assertListEqual(
        attention_score.shape.as_list(), expected_shape, msg=None)

  def test_transformer_get_config(self):
    layer = mobile_bert_layers.MobileBertTransformer(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=48,
        intermediate_act_fn='gelu',
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.4,
        intra_bottleneck_size=64,
        use_bottleneck_attention=True,
        key_query_shared_bottleneck=False,
        num_feedforward_networks=2,
        normalization_type='layer_norm',
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        name='block')
    layer_config = layer.get_config()
    new_layer = mobile_bert_layers.MobileBertTransformer.from_config(
        layer_config)
    self.assertEqual(layer_config, new_layer.get_config())


class MobileBertMaskedLMTest(tf.test.TestCase):

  def create_layer(self,
                   vocab_size,
                   hidden_size,
                   embedding_width,
                   output='predictions',
                   xformer_stack=None):
    # First, create a transformer stack that we can use to get the LM's
    # vocabulary weight.
    if xformer_stack is None:
      xformer_stack = mobile_bert_encoder.MobileBERTEncoder(
          word_vocab_size=vocab_size,
          num_blocks=1,
          hidden_size=hidden_size,
          num_attention_heads=4,
          word_embed_size=embedding_width)

    # Create a maskedLM from the transformer stack.
    test_layer = mobile_bert_layers.MobileBertMaskedLM(
        embedding_table=xformer_stack.get_embedding_table(), output=output)
    return test_layer

  def test_layer_creation(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    embedding_width = 32
    num_predictions = 21
    test_layer = self.create_layer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embedding_width=embedding_width)

    # Make sure that the output tensor of the masked LM is the right shape.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
    output = test_layer(lm_input_tensor, masked_positions=masked_positions)

    expected_output_shape = [None, num_predictions, vocab_size]
    self.assertEqual(expected_output_shape, output.shape.as_list())

  def test_layer_invocation_with_external_logits(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    embedding_width = 32
    num_predictions = 21
    xformer_stack = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=vocab_size,
        num_blocks=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        word_embed_size=embedding_width)
    test_layer = self.create_layer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embedding_width=embedding_width,
        xformer_stack=xformer_stack,
        output='predictions')
    logit_layer = self.create_layer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embedding_width=embedding_width,
        xformer_stack=xformer_stack,
        output='logits')

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
    output = test_layer(lm_input_tensor, masked_positions)
    logit_output = logit_layer(lm_input_tensor, masked_positions)
    logit_output = tf.keras.layers.Activation(tf.nn.log_softmax)(logit_output)
    logit_layer.set_weights(test_layer.get_weights())
    model = tf.keras.Model([lm_input_tensor, masked_positions], output)
    logits_model = tf.keras.Model(([lm_input_tensor, masked_positions]),
                                  logit_output)

    # Invoke the masked LM on some fake data to make sure there are no runtime
    # errors in the code.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        sequence_length, size=(batch_size, num_predictions))
    # ref_outputs = model.predict([lm_input_data, masked_position_data])
    # outputs = logits_model.predict([lm_input_data, masked_position_data])
    ref_outputs = model([lm_input_data, masked_position_data])
    outputs = logits_model([lm_input_data, masked_position_data])

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_predictions, vocab_size)
    self.assertEqual(expected_output_shape, ref_outputs.shape)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertAllClose(ref_outputs, outputs)

  def test_layer_invocation(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    embedding_width = 32
    num_predictions = 21
    test_layer = self.create_layer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embedding_width=embedding_width)

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
    output = test_layer(lm_input_tensor, masked_positions)
    model = tf.keras.Model([lm_input_tensor, masked_positions], output)

    # Invoke the masked LM on some fake data to make sure there are no runtime
    # errors in the code.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    _ = model.predict([lm_input_data, masked_position_data])

  def test_unknown_output_type_fails(self):
    with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
      _ = self.create_layer(
          vocab_size=8, hidden_size=8, embedding_width=4, output='bad')

  def test_hidden_size_smaller_than_embedding_width(self):
    hidden_size = 8
    sequence_length = 32
    num_predictions = 20
    with self.assertRaisesRegex(
        ValueError, 'hidden size 8 cannot be smaller than embedding width 16.'):
      test_layer = self.create_layer(
          vocab_size=8, hidden_size=8, embedding_width=16)
      lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
      masked_positions = tf.keras.Input(
          shape=(num_predictions,), dtype=tf.int32)
      _ = test_layer(lm_input_tensor, masked_positions)


if __name__ == '__main__':
  tf.test.main()
