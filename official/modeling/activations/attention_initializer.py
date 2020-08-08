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
"""Attention Layer Initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def attention_initializer(hidden_size):
  """Weight Initializer of Attention Layer in Seq2Seq Transformer.

  Args:
    hidden_size: hidden size of input tensor

  Returns:
    Initialized weights based on hidden size
  """
  limit = math.sqrt(6.0 / (hidden_size + hidden_size))
  return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
