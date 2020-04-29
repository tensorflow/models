# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras-based transformer block layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import attention
from official.nlp.modeling.layers import transformer_scaffold


# Test class that wraps a standard attention layer. If this layer is called
# at any point, the list passed to the config object will be filled with a
# boolean 'True'. We register this class as a Keras serializable so we can
# test serialization below.
@tf.keras.utils.register_keras_serializable(package='TestOnly')
class ValidatedAttentionLayer(attention.MultiHeadAttention):

  def __init__(self, call_list, **kwargs):
    super(ValidatedAttentionLayer, self).__init__(**kwargs)
    self.list = call_list

  def call(self, inputs, attention_mask=None):
    self.list.append(True)
    return super(ValidatedAttentionLayer, self).call(
        inputs, attention_mask=attention_mask)

  def get_config(self):
    config = super(ValidatedAttentionLayer, self).get_config()
    config['call_list'] = []
    return config


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class TransformerLayerTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(TransformerLayerTest, self).tearDown()
    tf.keras.mixed_precision.experimental.set_policy('float32')

  def test_layer_creation(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_creation_with_mask(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_creation_with_incorrect_mask_fails(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length - 3))
    with self.assertRaisesRegex(ValueError, 'When passing a mask tensor.*'):
      _ = test_layer([data_tensor, mask_tensor])

  def test_layer_invocation(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(data_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    _ = model.predict(input_data)
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_invocation_with_mask(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_layer_invocation_with_float16_dtype(self):
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = (10 * np.random.random_sample(
        (batch_size, sequence_length, width)))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_transform_with_initializer(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output.shape.as_list())
    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0])

  def test_layer_restoration_from_config(self):
    sequence_length = 21
    width = 80

    call_list = []
    attention_layer_cfg = {
        'num_heads': 10,
        'key_size': 8,
        'call_list': call_list,
        'name': 'test_layer',
    }
    test_layer = transformer_scaffold.TransformerScaffold(
        attention_cls=ValidatedAttentionLayer,
        attention_cfg=attention_layer_cfg,
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')

    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    pre_serialization_output = model.predict([input_data, mask_data])

    # Serialize the model config. Pass the serialized data through json to
    # ensure that we can serialize this layer to disk.
    serialized_data = json.dumps(model.get_config())
    post_string_serialized_data = json.loads(serialized_data)

    # Create a new model from the old config, and copy the weights. These models
    # should have identical outputs.
    new_model = tf.keras.Model.from_config(post_string_serialized_data)
    new_model.set_weights(model.get_weights())
    output = new_model.predict([input_data, mask_data])

    self.assertAllClose(pre_serialization_output, output)

    # If the layer was configured correctly, it should have a list attribute
    # (since it should have the custom class and config passed to it).
    new_model.summary()
    new_call_list = new_model.get_layer(
        name='transformer_scaffold')._attention_layer.list
    self.assertNotEmpty(new_call_list)
    self.assertTrue(new_call_list[0],
                    "The passed layer class wasn't instantiated.")


if __name__ == '__main__':
  tf.test.main()
