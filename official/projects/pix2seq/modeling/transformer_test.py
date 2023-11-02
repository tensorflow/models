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

"""Tests for transformer."""

import tensorflow as tf, tf_keras

from official.projects.pix2seq.modeling import transformer


class TransformerTest(tf.test.TestCase):

  def test_transformer_encoder(self):
    batch_size = 2
    sequence_length = 100
    feature_size = 256
    model = transformer.TransformerEncoder(
        num_layers=3,
        dim=feature_size,
        mlp_ratio=4.0,
        num_heads=2,
    )
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    out = model(input_tensor, mask=None, training=False)
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size)
    )

  def test_transformer_encoder_get_config(self):
    model = transformer.TransformerEncoder(
        num_layers=2,
        dim=256,
        mlp_ratio=4.0,
        num_heads=2,
    )
    config = model.get_config()
    expected_config = {
        'name': 'transformer_encoder',
        'trainable': True,
        'dtype': 'float32',
        'num_layers': 2,
        'dim': 256,
        'mlp_ratio': 4.0,
        'num_heads': 2,
        'drop_path': 0.1,
        'drop_units': 0.1,
        'drop_att': 0.0,
        'self_attention': True,
        'use_ffn_ln': False,
        'ln_scale_shift': True,
    }
    self.assertAllEqual(expected_config, config)

  def test_transformer_decoder_layer(self):
    batch_size = 2
    sequence_length = 100
    memory_length = 200
    feature_size = 256
    model = transformer.TransformerDecoderLayer(
        dim=feature_size,
        mlp_ratio=4.0,
        num_heads=2
    )
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    memory = tf.ones((batch_size, memory_length, feature_size))
    self_attention_mask = tf.ones(
        (batch_size, sequence_length, sequence_length), dtype=tf.int64
    )

    out, _ = model(
        input_tensor,
        memory,
        None,
        self_attention_mask,
        None,
        training=False,
    )
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size)
    )

  def test_transformer_decoder_layer_get_config(self):
    model = transformer.TransformerDecoderLayer(
        dim=256,
        mlp_ratio=4.0,
        num_heads=2
    )
    config = model.get_config()
    expected_config = {
        'name': 'transformer_decoder_layer',
        'trainable': True,
        'dtype': 'float32',
        'dim': 256,
        'mlp_ratio': 4.0,
        'num_heads': 2,
        'drop_path': 0.1,
        'drop_units': 0.1,
        'drop_att': 0.0,
        'dim_x_att': None,
        'self_attention': True,
        'cross_attention': True,
        'use_mlp': True,
        'use_enc_ln': False,
        'use_ffn_ln': False,
        'ln_scale_shift': True,
    }
    self.assertAllEqual(expected_config, config)

  def test_transformer_decoder(self):
    batch_size = 2
    sequence_length = 100
    memory_length = 200
    feature_size = 256
    num_layers = 3
    model = transformer.TransformerDecoder(
        num_layers=num_layers,
        dim=feature_size,
        mlp_ratio=4.0,
        num_heads=2,
    )
    input_tensor = tf.ones((batch_size, sequence_length, feature_size))
    memory = tf.ones((batch_size, memory_length, feature_size))
    self_attention_mask = tf.ones(
        (batch_size, sequence_length, sequence_length), dtype=tf.int64
    )

    out, cache = model(
        input_tensor, memory, None, self_attention_mask, None, training=False
    )
    self.assertAllEqual(
        tf.shape(out), (batch_size, sequence_length, feature_size)
    )
    self.assertAllEqual(
        tf.shape(cache), (num_layers, batch_size, sequence_length, feature_size)
    )

  def test_transformer_decoder_get_config(self):
    num_layers = 2
    num_attention_heads = 2
    intermediate_size = 256
    model = transformer.TransformerDecoder(
        num_layers=num_layers,
        dim=intermediate_size,
        mlp_ratio=4.0,
        num_heads=num_attention_heads,
    )
    config = model.get_config()
    expected_config = {
        'name': 'transformer_decoder',
        'trainable': True,
        'dtype': 'float32',
        'num_layers': 2,
        'dim': 256,
        'mlp_ratio': 4.0,
        'num_heads': 2,
        'drop_path': 0.1,
        'drop_units': 0.1,
        'drop_att': 0.0,
        'dim_x_att': None,
        'self_attention': True,
        'cross_attention': True,
        'use_mlp': True,
        'use_enc_ln': False,
        'use_ffn_ln': False,
        'ln_scale_shift': True,
    }
    self.assertAllEqual(expected_config, config)


if __name__ == '__main__':
  tf.test.main()
