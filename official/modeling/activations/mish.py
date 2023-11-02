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

"""Self Regularized Non-Monotonic Activation Function."""

import tensorflow as tf, tf_keras


@tf_keras.utils.register_keras_serializable(package='Text')
def mish(x) -> tf.Tensor:
  """Mish activation function.

     Mish: A Self Regularized Non-Monotonic Activation Function
     https://arxiv.org/pdf/1908.08681.pdf

     Mish(x) = x * tanh(ln(1+e^x))

  Args:
    x: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
  x = tf.convert_to_tensor(x)
  return x * tf.tanh(tf.nn.softplus(x))
