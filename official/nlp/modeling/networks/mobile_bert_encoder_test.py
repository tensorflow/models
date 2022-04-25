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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from official.nlp.modeling import models
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
    outputs = test_network([word_ids, mask, type_ids])
    layer_output, pooler_output = outputs['sequence_output'], outputs[
        'pooled_output']

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
        num_blocks=num_blocks)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    outputs = test_network([word_ids, mask, type_ids])
    all_layer_output = outputs['encoder_outputs']

    self.assertIsInstance(all_layer_output, list)
    self.assertLen(all_layer_output, num_blocks + 1)

  @parameterized.parameters('int32', 'float32')
  def test_mobilebert_encoder_invocation(self, input_mask_dtype):
    vocab_size = 100
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        input_mask_dtype=input_mask_dtype)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=input_mask_dtype)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    outputs = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], outputs)

    input_seq = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=vocab_size)
    input_mask = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=2)
    token_type = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=2)
    outputs = model.predict([input_seq, input_mask, token_type])

    sequence_output_shape = [1, sequence_length, hidden_size]
    self.assertAllEqual(outputs['sequence_output'].shape, sequence_output_shape)
    pooled_output_shape = [1, hidden_size]
    self.assertAllEqual(outputs['pooled_output'].shape, pooled_output_shape)

  def test_mobilebert_encoder_invocation_with_attention_score(self):
    vocab_size = 100
    hidden_size = 32
    sequence_length = 16
    num_blocks = 3
    test_network = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=num_blocks)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    outputs = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], outputs)

    input_seq = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=vocab_size)
    input_mask = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=2)
    token_type = generate_fake_input(
        batch_size=1, seq_len=sequence_length, vocab_size=2)
    outputs = model.predict([input_seq, input_mask, token_type])
    self.assertLen(outputs['attention_scores'], num_blocks)

  @parameterized.named_parameters(
      ('sequence_classification', models.BertClassifier, [None, 5]),
      ('token_classification', models.BertTokenClassifier, [None, 16, 5]))
  def test_mobilebert_encoder_for_downstream_task(self, task, prediction_shape):
    hidden_size = 32
    sequence_length = 16
    mobilebert_encoder = mobile_bert_encoder.MobileBERTEncoder(
        word_vocab_size=100, hidden_size=hidden_size)
    num_classes = 5
    classifier = task(network=mobilebert_encoder, num_classes=num_classes)

    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    prediction = classifier([word_ids, mask, type_ids])
    if task == models.BertTokenClassifier:
      prediction = prediction['logits']
    self.assertAllEqual(prediction.shape.as_list(), prediction_shape)


if __name__ == '__main__':
  tf.test.main()
