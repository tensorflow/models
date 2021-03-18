# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Keras-based one-hot embedding layer."""

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.keras_nlp.layers import on_device_embedding


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class OnDeviceEmbeddingTest(keras_parameterized.TestCase):

  def test_layer_creation(self):
    vocab_size = 31
    embedding_width = 27
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size, embedding_width=embedding_width)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # The output should be the same as the input, save that it has an extra
    # embedding_width dimension on the end.
    expected_output_shape = [None, sequence_length, embedding_width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    self.assertEqual(output_tensor.dtype, tf.float32)

  def test_layer_creation_with_mixed_precision(self):
    vocab_size = 31
    embedding_width = 27
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size, embedding_width=embedding_width, dtype=policy)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # The output should be the same as the input, save that it has an extra
    # embedding_width dimension on the end.
    expected_output_shape = [None, sequence_length, embedding_width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    self.assertEqual(output_tensor.dtype, tf.float16)

  def test_layer_invocation(self):
    vocab_size = 31
    embedding_width = 27
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size, embedding_width=embedding_width)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(input_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 3
    input_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    output = model.predict(input_data)
    self.assertEqual(tf.float32, output.dtype)

  def test_layer_invocation_with_mixed_precision(self):
    vocab_size = 31
    embedding_width = 27
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size, embedding_width=embedding_width, dtype=policy)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(input_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 3
    input_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    output = model.predict(input_data)
    self.assertEqual(tf.float16, output.dtype)

  def test_one_hot_layer_creation(self):
    vocab_size = 31
    embedding_width = 27
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        use_one_hot=True)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # The output should be the same as the input, save that it has an extra
    # embedding_width dimension on the end.
    expected_output_shape = [None, sequence_length, embedding_width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    self.assertEqual(output_tensor.dtype, tf.float32)

  def test_one_hot_layer_creation_with_mixed_precision(self):
    vocab_size = 31
    embedding_width = 27
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        dtype=policy,
        use_one_hot=True)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # The output should be the same as the input, save that it has an extra
    # embedding_width dimension on the end.
    expected_output_shape = [None, sequence_length, embedding_width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    self.assertEqual(output_tensor.dtype, tf.float16)

  def test_one_hot_layer_invocation(self):
    vocab_size = 31
    embedding_width = 27
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        use_one_hot=True)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(input_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 3
    input_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    output = model.predict(input_data)
    self.assertEqual(tf.float32, output.dtype)

  def test_one_hot_layer_invocation_with_mixed_precision(self):
    vocab_size = 31
    embedding_width = 27
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        dtype=policy,
        use_one_hot=True)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(input_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 3
    input_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    output = model.predict(input_data)
    self.assertEqual(tf.float16, output.dtype)

  def test_use_scale_layer_invocation(self):
    vocab_size = 31
    embedding_width = 27
    test_layer = on_device_embedding.OnDeviceEmbedding(
        vocab_size=vocab_size, embedding_width=embedding_width,
        scale_factor=embedding_width**0.5)
    # Create a 2-dimensional input (the first dimension is implicit).
    sequence_length = 23
    input_tensor = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    output_tensor = test_layer(input_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(input_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 3
    input_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    output = model.predict(input_data)
    self.assertEqual(tf.float32, output.dtype)


if __name__ == "__main__":
  tf.test.main()
