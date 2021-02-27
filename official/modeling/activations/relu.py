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

"""Customized Relu activation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def relu6(features):
  """Computes the Relu6 activation function.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return tf.nn.relu6(features)
