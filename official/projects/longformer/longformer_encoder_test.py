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

"""Tests for official.nlp.projects.longformer.longformer_encoder."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.projects.longformer.longformer_encoder import LongformerEncoder


class LongformerEncoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(LongformerEncoderTest, self).setUp()
    np.random.seed(0)
    tf.random.set_seed(0)

  @combinations.generate(
      combinations.combine(
          attention_window=[32, 128], global_attention_size=[0, 1, 2]))
  def test_encoder(self, attention_window, global_attention_size):
    sequence_length = 128
    batch_size = 2
    vocab_size = 1024
    hidden_size = 256
    network = LongformerEncoder(
        global_attention_size=global_attention_size,
        vocab_size=vocab_size,
        attention_window=[attention_window],
        hidden_size=hidden_size,
        num_layers=1,
        num_attention_heads=4,
        max_sequence_length=512)
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length), dtype=np.int32)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    type_id_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    inputs = {
        'input_word_ids': word_id_data,
        'input_mask': mask_data,
        'input_type_ids': type_id_data,
    }
    outputs = network(inputs)
    self.assertEqual(outputs['sequence_output'].shape,
                     (batch_size, sequence_length, hidden_size))

  @combinations.generate(
      combinations.combine(
          norm_first=[True, False], global_attention_size=[0, 1, 2]))
  def test_norm_first(self, norm_first, global_attention_size):
    sequence_length = 128
    batch_size = 2
    vocab_size = 1024
    hidden_size = 256
    network = LongformerEncoder(
        global_attention_size=global_attention_size,
        vocab_size=vocab_size,
        attention_window=[32],
        hidden_size=hidden_size,
        num_layers=1,
        num_attention_heads=4,
        max_sequence_length=512,
        norm_first=norm_first)
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length), dtype=np.int32)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    type_id_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    inputs = {
        'input_word_ids': word_id_data,
        'input_mask': mask_data,
        'input_type_ids': type_id_data,
    }
    outputs = network(inputs)
    self.assertEqual(outputs['sequence_output'].shape,
                     (batch_size, sequence_length, hidden_size))


if __name__ == '__main__':
  tf.test.main()
