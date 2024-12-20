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

"""Dot product with margin layer."""
# pylint: disable=g-classes-have-attributes

from typing import Tuple
import tensorflow as tf, tf_keras

from official.modeling import tf_utils


@tf_keras.utils.register_keras_serializable(package='Text')
class MatMulWithMargin(tf_keras.layers.Layer):
  """This layer computs a dot product matrix given two encoded inputs.

  Args:
    logit_scale: The scaling factor of dot products when doing training.
    logit_margin: The margin value between the positive and negative examples
      when doing training.
  """

  def __init__(self,
               logit_scale=1.0,
               logit_margin=0.0,
               **kwargs):
    super().__init__(**kwargs)
    self.logit_scale = logit_scale
    self.logit_margin = logit_margin

  def call(self, left_encoded: tf.Tensor,
           right_encoded: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf_utils.get_shape_list(
        left_encoded, name='sequence_output_tensor')[0]

    # Left -> Right dot product.
    left_dot_products = tf.matmul(
        left_encoded, right_encoded, transpose_b=True)

    self.left_logits = self.logit_scale * (
        left_dot_products - self.logit_margin * tf.eye(batch_size))

    # Right -> Left dot product.
    self.right_logits = tf.transpose(self.left_logits)

    return (self.left_logits, self.right_logits)

  def get_config(self):
    config = {
        'logit_scale': self.logit_scale,
        'logit_margin': self.logit_margin}
    config.update(super().get_config())
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
