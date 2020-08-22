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
from official.nlp.modeling import models
from official.nlp.modeling.networks import mobile_bert_encoder
from official.nlp.projects.mobilebert import utils


class ModelingTest(parameterized.TestCase, tf.test.TestCase):

  def test_embedding_layer_with_token_type(self):
    layer = mobile_bert_encoder.MobileBertEmbedding(10, 8, 2, 16)
    input_seq = tf.Variable([[2, 3, 4, 5]])
    token_type = tf.Variable([[0, 1, 1, 1]])
    output = layer(input_seq, token_type)
    output_shape = output.shape.as_list()
    expected_shape = [1, 4, 16]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_embedding_layer_without_token_type(self):
    layer = mobile_bert_encoder.MobileBertEmbedding(10, 8, 2, 16)
    input_seq = tf.Variable([[2, 3, 4, 5]])
    output = layer(input_seq)
    output_shape = output.shape.as_list()
    expected_shape = [1, 4, 16]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_no_norm(self):
    layer = mobile_bert_encoder.NoNorm()
    feature = tf.random.normal([2, 3, 4])
    output = layer(feature)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 4]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  @parameterized.named_parameters(
      ('with_kq_shared_bottleneck', False),
      ('without_kq_shared_bottleneck', True))
  def test_transfomer_kq_shared_bottleneck(self, is_kq_shared):
    feature = tf.random.uniform([2, 3, 512])
    layer = mobile_bert_encoder.TransformerLayer(
        key_query_shared_bottleneck=is_kq_shared)
    output = layer(feature)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 512]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_transfomer_with_mask(self):
    feature = tf.random.uniform([2, 3, 512])
    input_mask = [[[0., 0., 1.],
                   [0., 0., 1.],
                   [0., 0., 1.]],
                  [[0., 1., 1.],
                   [0., 1., 1.],
                   [0., 1., 1.]]]
    input_mask = np.asarray(input_mask)
    layer = mobile_bert_encoder.TransformerLayer()
    output = layer(feature, input_mask)
    output_shape = output.shape.as_list()
    expected_shape = [2, 3, 512]
    self.assertListEqual(output_shape, expected_shape, msg=None)

  def test_transfomer_return_attention_score(self):
    sequence_length = 5
    num_attention_heads = 8
    feature = tf.random.uniform([2, sequence_length, 512])
    layer = mobile_bert_encoder.TransformerLayer(
        num_attention_heads=num_attention_heads)
    _, attention_score = layer(feature, return_attention_scores=True)
    expected_shape = [2, num_attention_heads, sequence_length, sequence_length]
    self.assertListEqual(attention_score.shape.as_list(), expected_shape,
                         msg=None)

  @parameterized.named_parameters(
      ('default_setting', 'relu', True, 'no_norm', False),
      ('gelu', 'gelu', True, 'no_norm', False),
      ('kq_not_shared', 'relu', False, 'no_norm', False),
      ('layer_norm', 'relu', True, 'layer_norm', False),
      ('use_pooler', 'relu', True, 'no_norm', True),
      ('with_pooler_layer', 'relu', True, 'layer_norm', False))
  def test_mobilebert_encoder(self, act_fn, kq_shared_bottleneck,
                              normalization_type, use_pooler):
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=100,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        intermediate_act_fn=act_fn,
        key_query_shared_bottleneck=kq_shared_bottleneck,
        normalization_type=normalization_type,
        classifier_activation=use_pooler)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    layer_output, pooler_output = test_network([word_ids, mask, type_ids])

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, num_blocks)

    layer_output_shape = [None, sequence_length, hidden_size]
    self.assertAllEqual(layer_output.shape.as_list(), layer_output_shape)
    pooler_output_shape = [None, hidden_size]
    self.assertAllEqual(pooler_output.shape.as_list(), pooler_output_shape)
    self.assertAllEqual(tf.float32, layer_output.dtype)

  def test_mobilebert_encoder_return_all_layer_output(self):
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=100,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        return_all_layers=True)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    all_layer_output, _ = test_network([word_ids, mask, type_ids])

    self.assertIsInstance(all_layer_output, list)
    self.assertLen(all_layer_output, num_blocks + 1)

  def test_mobilebert_encoder_invocation(self):
    vocab_size = 100
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        return_all_layers=False)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    layer_out_tensor, pooler_out_tensor = test_network([word_ids,
                                                        mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids],
                           [layer_out_tensor, pooler_out_tensor])

    input_seq = utils.generate_fake_input(batch_size=1,
                                          seq_len=sequence_length,
                                          vocab_size=vocab_size)
    input_mask = utils.generate_fake_input(batch_size=1,
                                           seq_len=sequence_length,
                                           vocab_size=2)
    token_type = utils.generate_fake_input(batch_size=1,
                                           seq_len=sequence_length,
                                           vocab_size=2)
    layer_output, pooler_output = model.predict([input_seq, input_mask,
                                                 token_type])

    layer_output_shape = [1, sequence_length, hidden_size]
    self.assertAllEqual(layer_output.shape, layer_output_shape)
    pooler_output_shape = [1, hidden_size]
    self.assertAllEqual(pooler_output.shape, pooler_output_shape)

  def test_mobilebert_encoder_invocation_with_attention_score(self):
    vocab_size = 100
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        return_all_layers=False,
        return_attention_score=True)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    layer_out_tensor, pooler_out_tensor, attention_out_tensor = test_network(
        [word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids],
                           [layer_out_tensor, pooler_out_tensor,
                            attention_out_tensor])

    input_seq = utils.generate_fake_input(batch_size=1,
                                          seq_len=sequence_length,
                                          vocab_size=vocab_size)
    input_mask = utils.generate_fake_input(batch_size=1,
                                           seq_len=sequence_length,
                                           vocab_size=2)
    token_type = utils.generate_fake_input(batch_size=1,
                                           seq_len=sequence_length,
                                           vocab_size=2)
    _, _, attention_score_output = model.predict([input_seq, input_mask,
                                                  token_type])
    self.assertLen(attention_score_output, num_blocks)

  @parameterized.named_parameters(
      ('sequence_classification', models.BertClassifier, [None, 5]),
      ('token_classification', models.BertTokenClassifier, [None, 16, 5]))
  def test_mobilebert_encoder_for_downstream_task(self, task, prediction_shape):
    hidden_size = 32
    sequence_length = 16
    mobilebert_encoder = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=100, hidden_size=hidden_size)
    num_classes = 5
    classifier = task(network=mobilebert_encoder,
                      num_classes=num_classes)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    prediction = classifier([word_ids, mask, type_ids])
    self.assertAllEqual(prediction.shape.as_list(), prediction_shape)

if __name__ == '__main__':
  tf.test.main()
