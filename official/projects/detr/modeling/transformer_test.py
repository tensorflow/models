# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for transformer."""

import tensorflow as tf, tf_keras

from official.projects.detr.modeling import transformer


class TransformerTest(tf.test.TestCase):

  def test_transformer_encoder_block(self):
    batch_size = 2
    sequence_length = 100
    feature_size = 256
    num_attention_heads = 2
    inner_dim = 256
    inner_activation = 'relu'
    model = transformer.TransformerEncoderBlock(num_attention_heads, inner_dim,
                                                inner_activation)
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    attention_mask = tf.ones((batch_size, sequence_length, sequence_length),
                             dtype=tf.int64)
    pos_embed = tf.ones((batch_size, sequence_length, feature_size))

    out = model([input_tensor, attention_mask, pos_embed])
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size))

  def test_transformer_encoder_block_get_config(self):
    num_attention_heads = 2
    inner_dim = 256
    inner_activation = 'relu'
    model = transformer.TransformerEncoderBlock(num_attention_heads, inner_dim,
                                                inner_activation)
    config = model.get_config()
    expected_config = {
        'name': 'transformer_encoder_block',
        'trainable': True,
        'dtype': 'float32',
        'num_attention_heads': 2,
        'inner_dim': 256,
        'inner_activation': 'relu',
        'output_dropout': 0.0,
        'attention_dropout': 0.0,
        'output_range': None,
        'kernel_initializer': {
            'class_name': 'GlorotUniform',
            'config': {
                'seed': None}
        },
        'bias_initializer': {
            'class_name': 'Zeros',
            'config': {}
        },
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
        'use_bias': True,
        'norm_first': False,
        'norm_epsilon': 1e-12,
        'inner_dropout': 0.0,
        'attention_initializer': {
            'class_name': 'GlorotUniform',
            'config': {'seed': None}
        },
        'attention_axes': None}
    self.assertAllEqual(expected_config, config)

  def test_transformer_encoder(self):
    batch_size = 2
    sequence_length = 100
    feature_size = 256
    num_layers = 2
    num_attention_heads = 2
    intermediate_size = 256
    model = transformer.TransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size)
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    attention_mask = tf.ones((batch_size, sequence_length, sequence_length),
                             dtype=tf.int64)
    pos_embed = tf.ones((batch_size, sequence_length, feature_size))
    out = model(input_tensor, attention_mask, pos_embed)
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size))

  def test_transformer_encoder_get_config(self):
    num_layers = 2
    num_attention_heads = 2
    intermediate_size = 256
    model = transformer.TransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size)
    config = model.get_config()
    expected_config = {
        'name': 'transformer_encoder',
        'trainable': True,
        'dtype': 'float32',
        'num_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'activation': 'relu',
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'use_bias': False,
        'norm_first': True,
        'norm_epsilon': 1e-06,
        'intermediate_dropout': 0.0
    }
    self.assertAllEqual(expected_config, config)

  def test_transformer_decoder_block(self):
    batch_size = 2
    sequence_length = 100
    memory_length = 200
    feature_size = 256
    num_attention_heads = 2
    intermediate_size = 256
    intermediate_activation = 'relu'
    model = transformer.TransformerDecoderBlock(num_attention_heads,
                                                intermediate_size,
                                                intermediate_activation)
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    memory = tf.ones((batch_size, memory_length, feature_size))
    attention_mask = tf.ones((batch_size, sequence_length, memory_length),
                             dtype=tf.int64)
    self_attention_mask = tf.ones(
        (batch_size, sequence_length, sequence_length), dtype=tf.int64)
    input_pos_embed = tf.ones((batch_size, sequence_length, feature_size))
    memory_pos_embed = tf.ones((batch_size, memory_length, feature_size))

    out, _ = model([
        input_tensor, memory, attention_mask, self_attention_mask,
        input_pos_embed, memory_pos_embed
    ])
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size))

  def test_transformer_decoder_block_get_config(self):
    num_attention_heads = 2
    intermediate_size = 256
    intermediate_activation = 'relu'
    model = transformer.TransformerDecoderBlock(num_attention_heads,
                                                intermediate_size,
                                                intermediate_activation)
    config = model.get_config()
    expected_config = {
        'name': 'transformer_decoder_block',
        'trainable': True,
        'dtype': 'float32',
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'intermediate_activation': 'relu',
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'kernel_initializer': {
            'class_name': 'GlorotUniform',
            'config': {
                'seed': None
            }
        },
        'bias_initializer': {
            'class_name': 'Zeros',
            'config': {}
        },
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
        'use_bias': True,
        'norm_first': False,
        'norm_epsilon': 1e-12,
        'intermediate_dropout': 0.0,
        'attention_initializer': {
            'class_name': 'GlorotUniform',
            'config': {
                'seed': None
            }
        }
    }
    self.assertAllEqual(expected_config, config)

  def test_transformer_decoder(self):
    batch_size = 2
    sequence_length = 100
    memory_length = 200
    feature_size = 256
    num_layers = 2
    num_attention_heads = 2
    intermediate_size = 256
    model = transformer.TransformerDecoder(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size)
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    memory = tf.ones((batch_size, memory_length, feature_size))
    attention_mask = tf.ones((batch_size, sequence_length, memory_length),
                             dtype=tf.int64)
    self_attention_mask = tf.ones(
        (batch_size, sequence_length, sequence_length), dtype=tf.int64)
    input_pos_embed = tf.ones((batch_size, sequence_length, feature_size))
    memory_pos_embed = tf.ones((batch_size, memory_length, feature_size))

    outs = model(
        input_tensor,
        memory,
        self_attention_mask,
        attention_mask,
        return_all_decoder_outputs=True,
        input_pos_embed=input_pos_embed,
        memory_pos_embed=memory_pos_embed)
    self.assertLen(outs, 2)  # intermeidate decoded outputs.
    for out in outs:
      self.assertAllEqual(
          tf.shape(out), (batch_size, sequence_length, feature_size))

  def test_transformer_decoder_get_config(self):
    num_layers = 2
    num_attention_heads = 2
    intermediate_size = 256
    model = transformer.TransformerDecoder(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size)
    config = model.get_config()
    expected_config = {
        'name': 'transformer_decoder',
        'trainable': True,
        'dtype': 'float32',
        'num_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'activation': 'relu',
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'use_bias': False,
        'norm_first': True,
        'norm_epsilon': 1e-06,
        'intermediate_dropout': 0.0
    }
    self.assertAllEqual(expected_config, config)


if __name__ == '__main__':
  tf.test.main()
