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

"""Tests for TransformerDecoder."""

import tensorflow as tf

from official.projects.const_cl.modeling.heads import transformer_decoder


class TransformerTest(tf.test.TestCase):

  def test_decoder_unit_return_shape(self):
    decoder_unit = transformer_decoder.DecoderUnit(
        num_channels=128,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    batch_size = 16
    num_inputs = 128
    num_channels = 256
    input_tensor = tf.zeros([batch_size, num_inputs, num_channels])
    memory_tensor = tf.ones([batch_size, num_inputs * 4, num_channels])
    outputs = decoder_unit(input_tensor, memory_tensor, training=False)
    self.assertAllEqual(outputs['hidden_states'].shape,
                        [batch_size, num_inputs, num_inputs])

    self.assertAllEqual(outputs['attention_weights'].shape,
                        [batch_size, num_inputs, 4 * num_inputs])

  def test_decoder_unit_serialize_deserialize(self):
    decoder_unit = transformer_decoder.DecoderUnit(
        num_channels=128,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    config = decoder_unit.get_config()
    new_decoder_unit = (
        transformer_decoder.DecoderUnit.from_config(config))
    self.assertAllEqual(
        decoder_unit.get_config(), new_decoder_unit.get_config())

  def test_decoder_layer_return_shape(self):
    decoder_layer = transformer_decoder.TransformerDecoderLayer(
        num_channels=128,
        num_heads=3,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    batch_size = 16
    num_inputs = 128
    num_channels = 256
    input_tensor = tf.zeros([batch_size, num_inputs, num_channels])
    memory_tensor = tf.ones([batch_size, num_inputs * 4, num_channels])
    outputs = decoder_layer(input_tensor, memory_tensor, training=False)
    self.assertAllEqual(outputs['hidden_states'].shape,
                        [batch_size, num_inputs, num_inputs * 3])

    self.assertAllEqual(outputs['attention_weights'][-1].shape,
                        [batch_size, num_inputs, 4 * num_inputs])

  def test_decoder_layer_serialize_deserialize(self):
    decoder_layer = transformer_decoder.TransformerDecoderLayer(
        num_channels=128,
        num_heads=3,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    config = decoder_layer.get_config()
    new_decoder_layer = (
        transformer_decoder.TransformerDecoderLayer.from_config(config))
    self.assertAllEqual(
        decoder_layer.get_config(), new_decoder_layer.get_config())

  def test_decoder_return_shape(self):
    decoder = transformer_decoder.TransformerDecoder(
        num_channels=128,
        num_layers=5,
        num_heads=3,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    batch_size = 16
    num_inputs = 128
    num_channels = 256
    input_tensor = tf.zeros([batch_size, num_inputs, num_channels])
    memory_tensor = tf.ones([batch_size, num_inputs * 4, num_channels])
    outputs = decoder(input_tensor, memory_tensor, training=False)
    self.assertLen(outputs['attention_weights'], 5)
    self.assertAllEqual(outputs['hidden_states'][-1].shape,
                        [batch_size, num_inputs, num_inputs * 3])
    self.assertAllEqual(outputs['attention_weights'][-1][-1].shape,
                        [batch_size, num_inputs, 4 * num_inputs])

  def test_decoder_serialize_deserialize(self):
    decoder = transformer_decoder.TransformerDecoder(
        num_channels=128,
        num_layers=5,
        num_heads=3,
        use_bias=True,
        dropout_rate=0.5,
        activation='relu',
        layer_norm_epsilon=1e-7)
    config = decoder.get_config()
    new_decoder = transformer_decoder.TransformerDecoder.from_config(config)
    self.assertAllEqual(
        decoder.get_config(), new_decoder.get_config())

if __name__ == '__main__':
  tf.test.main()
