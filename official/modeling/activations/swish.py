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

"""Customized Swish activation."""

import tensorflow as tf, tf_keras


@tf_keras.utils.register_keras_serializable(package='Text')
def simple_swish(features):
  """Computes the Swish activation function.

  The tf.nn.swish operation uses a custom gradient to reduce memory usage.
  Since saving custom gradients in SavedModel is currently not supported, and
  one would not be able to use an exported TF-Hub module for fine-tuning, we
  provide this wrapper that can allow to select whether to use the native
  TensorFlow swish operation, or whether to use a customized operation that
  has uses default TensorFlow gradient computation.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return features * tf.nn.sigmoid(features)


@tf_keras.utils.register_keras_serializable(package='Text')
def hard_swish(features):
  """Computes a hard version of the swish function.

  This operation can be used to reduce computational cost and improve
  quantization for edge devices.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  fdtype = features.dtype
  return features * tf.nn.relu6(features + tf.cast(3., fdtype)) * (1. / 6.)


@tf_keras.utils.register_keras_serializable(package='Text')
def identity(features):
  """Computes the identity function.

  Useful for helping in quantization.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return tf.identity(features)
