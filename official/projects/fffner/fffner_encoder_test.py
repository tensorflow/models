# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.projects.fffner.fffner_encoder."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.fffner import fffner_encoder


class FFFNerEncoderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    tf.random.set_seed(0)

  def test_encoder(self):
    sequence_length = 128
    batch_size = 2
    vocab_size = 1024
    hidden_size = 256
    network = fffner_encoder.FFFNerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=1,
        num_attention_heads=4,
        max_sequence_length=512,
        dict_outputs=True)
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length), dtype=np.int32)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    type_id_data = np.random.randint(
        2, size=(batch_size, sequence_length), dtype=np.int32)
    is_entity_token_pos = np.random.randint(
        sequence_length, size=(batch_size,), dtype=np.int32)
    entity_type_token_pos = np.random.randint(
        sequence_length, size=(batch_size,), dtype=np.int32)
    inputs = {
        'input_word_ids': word_id_data,
        'input_mask': mask_data,
        'input_type_ids': type_id_data,
        'is_entity_token_pos': is_entity_token_pos,
        'entity_type_token_pos': entity_type_token_pos
    }
    outputs = network(inputs)
    self.assertEqual(outputs['sequence_output'].shape,
                     (batch_size, sequence_length, hidden_size))

    self.assertEqual(outputs['pooled_output'].shape,
                     (batch_size, 2 * hidden_size))


if __name__ == '__main__':
  tf.test.main()
