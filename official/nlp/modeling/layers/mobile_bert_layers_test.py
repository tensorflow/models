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
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import mobile_bert_layers


def generate_fake_input(batch_size=1, seq_len=5, vocab_size=10000, seed=0):
  """Generate consisitant fake integer input sequences."""
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


if __name__ == '__main__':
  tf.test.main()
