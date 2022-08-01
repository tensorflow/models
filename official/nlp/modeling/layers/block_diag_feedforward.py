# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Keras-based gated feedforward layer."""
# pylint: disable=g-classes-have-attributes
from typing import Optional

import tensorflow as tf

from official.modeling import tf_utils


class BlockDiagFeedforward(tf.keras.layers.Layer):
  """Block diagonal feedforward layer.

  This layer replaces the weight matrix of the output_dense layer with a block
  diagonal matrix to save layer parameters and FLOPs. A linear mixing layer can
  be added optionally to improve layer expressibility.

  Args:
    intermediate_size: Size of the intermediate layer.
    intermediate_activation: Activation for the intermediate layer.
    dropout: Dropout probability for the output dropout.
    num_blocks: The number of blocks for the block diagonal matrix of the
      output_dense layer.
    apply_mixing: Apply linear mixing if True.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(
      self,
      intermediate_size: int,
      intermediate_activation: str,
      dropout: float,
      num_blocks: int = 1,
      apply_mixing: bool = True,
      kernel_initializer: str = "glorot_uniform",
      bias_initializer: str = "zeros",
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
      bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
      **kwargs):  # pylint: disable=g-doc-args
    super().__init__(**kwargs)
    self._intermediate_size = intermediate_size
    self._intermediate_activation = intermediate_activation
    self._dropout = dropout
    self._num_blocks = num_blocks
    self._apply_mixing = apply_mixing

    if intermediate_size % num_blocks != 0:
      raise ValueError("Intermediate_size (%d) isn't a multiple of num_blocks "
                       "(%d)." % (intermediate_size, num_blocks))

    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    hidden_size = input_shape.as_list()[-1]

    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    self._intermediate_dense = tf.keras.layers.EinsumDense(
        "abc,cde->abde",
        output_shape=(None, self._num_blocks,
                      self._intermediate_size // self._num_blocks),
        bias_axes="de",
        name="intermediate",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)

    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._intermediate_activation, dtype=policy)

    self._output_dense = tf.keras.layers.EinsumDense(
        "abde,deo->abdo",
        output_shape=(None, self._num_blocks, hidden_size // self._num_blocks),
        bias_axes="do",
        name="output",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)

    if self._apply_mixing:
      self._output_mixing = tf.keras.layers.EinsumDense(
          "abdo,de->abeo",
          output_shape=(None, self._num_blocks,
                        hidden_size // self._num_blocks),
          name="output_mixing",
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
          **common_kwargs)
    self._output_reshape = tf.keras.layers.Reshape((-1, hidden_size))

    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout)

  def get_config(self):
    config = {
        "intermediate_size":
            self._intermediate_size,
        "intermediate_activation":
            self._intermediate_activation,
        "dropout":
            self._dropout,
        "num_blocks":
            self._num_blocks,
        "apply_mixing":
            self._apply_mixing,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint)
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    intermediate_output = self._intermediate_dense(inputs)
    intermediate_output = self._intermediate_activation_layer(
        intermediate_output)
    layer_output = self._output_dense(intermediate_output)
    if self._apply_mixing:
      layer_output = self._output_mixing(layer_output)
    layer_output = self._output_reshape(layer_output)
    layer_output = self._output_dropout(layer_output)

    return layer_output
