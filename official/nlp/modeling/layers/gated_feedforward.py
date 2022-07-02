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

import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling.layers import util


@tf.keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class GatedFeedforward(tf.keras.layers.Layer):
  """Gated linear feedforward layer.

  This layer follows the paper "GLU Variants Improve Transformer"
  (https://arxiv.org/abs/2002.05202). In additional, it allows to stack
  multiple feedforward blocks and specify the position of dropout layer.

  Args:
    intermediate_size: Size of the intermediate layer.
    intermediate_activation: Activation for the intermediate layer.
    dropout: Dropout probability for the output dropout.
    use_gate: Whether to use gated linear units. If True, assuming `GELU` as the
      activation and omitting bias, will apply
      `GEGLU(x, W, V, W_2) = (GEGLU(xW) * xV)W2`; if False, will follow
      "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) paper and
        apply `FFN(x, W, W_2) = GELU(xW_1)W_2.`
    num_blocks: The number of feedforward blocks to stack. Each block contains a
      (gated) linear layer and a fully connected layer followed by dropout,
      layer norm and residual.
    dropout_position: Where to apply the dropout, the value can be either
      `before_residual` or `after_residual`. If `before_residual`, will apply
      `layer_output = layer_norm(dropout(layer_output) + layer_input)`; if
      `after residual`, will apply
      `layer_output = dropout(layer_norm(layer_output + layer_input))`.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(self,
               inner_dim=768,
               inner_activation=tf_utils.get_activation("gelu"),
               dropout=0.0,
               use_gate=True,
               apply_output_layer_norm=True,
               num_blocks=1,
               dropout_position="before_residual",
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    inner_dim = kwargs.pop("intermediate_size", inner_dim)
    inner_activation = kwargs.pop("intermediate_activation", inner_activation)
    util.filter_kwargs(kwargs)
    super().__init__(**kwargs)
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._dropout = dropout
    self._use_gate = use_gate
    self._num_blocks = num_blocks
    self._apply_output_layer_norm = apply_output_layer_norm
    self._dropout_position = dropout_position
    if self._dropout_position not in ("before_residual", "after_residual"):
      raise ValueError(
          "The dropout_position should be either `before_residual` or"
          "`after_residual`, got: %s" % self._dropout_position)

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
    self._intermediate_dense = []
    self._inner_activation_layers = []
    self._gate_dense = []
    self._output_dense = []
    self._output_dropout = []
    self._output_layer_norm = []
    activation_policy = tf.keras.mixed_precision.global_policy()
    if activation_policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      activation_policy = tf.float32
    for i in range(self._num_blocks):
      self._intermediate_dense.append(
          tf.keras.layers.EinsumDense(
              "abc,cd->abd",
              output_shape=(None, self._inner_dim),
              bias_axes="d",
              name="intermediate_%d" % i,
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
              bias_initializer=tf_utils.clone_initializer(
                  self._bias_initializer),
              **common_kwargs))
      self._inner_activation_layers.append(
          tf.keras.layers.Activation(
              self._inner_activation, dtype=activation_policy))
      if self._use_gate:
        self._gate_dense.append(
            tf.keras.layers.EinsumDense(
                "abc,cd->abd",
                output_shape=(None, self._inner_dim),
                bias_axes="d",
                name="gate_%d" % i,
                kernel_initializer=tf_utils.clone_initializer(
                    self._kernel_initializer),
                bias_initializer=tf_utils.clone_initializer(
                    self._bias_initializer),
                **common_kwargs))
      self._output_dense.append(
          tf.keras.layers.EinsumDense(
              "abc,cd->abd",
              output_shape=(None, hidden_size),
              bias_axes="d",
              name="output_%d" % i,
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
              bias_initializer=tf_utils.clone_initializer(
                  self._bias_initializer),
              **common_kwargs))
      self._output_dropout.append(tf.keras.layers.Dropout(rate=self._dropout))
      # Use float32 in layernorm for numeric stability.
      if self._apply_output_layer_norm:
        self._output_layer_norm.append(
            tf.keras.layers.LayerNormalization(
                name="output_layer_norm_%d" % i,
                axis=-1,
                epsilon=1e-12,
                dtype=tf.float32))

  def get_config(self):
    config = {
        "inner_dim":
            self._inner_dim,
        "inner_activation":
            self._inner_activation,
        "dropout":
            self._dropout,
        "use_gate":
            self._use_gate,
        "num_blocks":
            self._num_blocks,
        "dropout_position":
            self._dropout_position,
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
    layer_output = inputs
    for i in range(self._num_blocks):
      layer_input = layer_output
      intermediate_output = self._intermediate_dense[i](layer_input)
      intermediate_output = self._inner_activation_layers[i](
          intermediate_output)
      if self._use_gate:
        gated_linear = self._gate_dense[i](layer_input)
        intermediate_output = intermediate_output * gated_linear

      layer_output = self._output_dense[i](intermediate_output)
      if self._dropout_position == "before_residual":
        layer_output = self._output_dropout[i](layer_output)

      # During mixed precision training, `layer_input` may be from layer norm.
      # If so, it is always fp32. Cast layer_output to fp32 for the subsequent
      # add.
      if layer_input.dtype == tf.float32:
        layer_output = tf.cast(layer_output, tf.float32)
      if self._apply_output_layer_norm:
        layer_output = self._output_layer_norm[i](layer_output + layer_input)
      if self._dropout_position == "after_residual":
        layer_output = self._output_dropout[i](layer_output)

    return layer_output
