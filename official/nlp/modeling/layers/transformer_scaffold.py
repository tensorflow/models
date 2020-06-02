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
"""Keras-based transformer scaffold layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import gin
import tensorflow as tf

from official.nlp.modeling.layers import attention


@tf.keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class TransformerScaffold(tf.keras.layers.Layer):
  """Transformer scaffold layer.

  This layer implements the Transformer from "Attention Is All You Need".
  (https://arxiv.org/abs/1706.03762), with a customizable attention layer and
  feedforward layer option. Users can pass a class to
  `attention_cls`/`feedforward_cls` and associated config to
  `attention_cfg`/`feedforward_cfg`, in which case the scaffold will
  instantiate the class with the config, or pass a class instance to
  `attention_cls`/`feedforward_cls`.

  Arguments:
    num_attention_heads: Number of attention heads.
    intermediate_size: Size of the intermediate layer.
    intermediate_activation: Activation for the intermediate layer.
    attention_cls: A class to instantiate attention layer, or a layer instance.
    attention_cfg: The config with which to instantiate `attention_cls`. Ignored
      if attention_cls is a layer instance or None. If `attention_cls` is a
      class, but `attention_cfg` is None, following kwargs will be used to
      instantiate the attention instance:
      {
        "num_heads": num_attention_heads,
        "key_size": int(hidden_size // num_attention_heads),
        "dropout": attention_dropout_rate,
        "name": "self_attention"
      }, where `hidden_size` is the input tensor's last dimension.
    feedforward_cls: A class to instantiate feedforward layer, or a layer
      instance. If None, will use the standard feedforward layer as described
      in "Attention Is All You Need" paper. If not None, the instantiated
      feedforward layer is expected to take the output of attention as input
      and its output is this transformer layer's output.
    feedforward_cfg: The config with which to instantiate `feedforward_cls`.
      Ignored if feedforward_cls is a layer instance or is None.
      If `feedforward_cls` is a class, but `feedforward_cfg` is None, following
      kwargs will be used to instantiate the feedforward instance:
      {
        "intermediate_size": intermediate_size,
        "intermediate_activation": intermediate_activation,
        "dropout": dropout_rate,
        "name": "feedforward"
      }.
    dropout_rate: Dropout probability for the post-attention and output dropout.
    attention_dropout_rate: Dropout probability for within the attention layer.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation,
               attention_cls=attention.MultiHeadAttention,
               attention_cfg=None,
               feedforward_cls=None,
               feedforward_cfg=None,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(TransformerScaffold, self).__init__(**kwargs)

    self._attention_cfg = attention_cfg
    self._attention_cls = attention_cls
    self._feedforward_cls = feedforward_cls
    self._feedforward_cfg = feedforward_cfg
    self._num_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._intermediate_activation = intermediate_activation
    self._attention_dropout_rate = attention_dropout_rate
    self._dropout_rate = dropout_rate
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    input_tensor = input_shape[0] if len(input_shape) == 2 else input_shape
    input_tensor_shape = tf.TensorShape(input_tensor)
    if len(input_tensor_shape) != 3:
      raise ValueError(
          "TransformerScaffold expects a three-dimensional input of "
          "shape [batch, sequence, width].")
    batch_size, sequence_length, hidden_size = input_tensor_shape

    if len(input_shape) == 2:
      mask_tensor_shape = tf.TensorShape(input_shape[1])
      expected_mask_tensor_shape = tf.TensorShape(
          [batch_size, sequence_length, sequence_length])
      if not expected_mask_tensor_shape.is_compatible_with(mask_tensor_shape):
        raise ValueError("When passing a mask tensor to TransformerLayer, the "
                         "mask tensor must be of shape [batch, "
                         "sequence_length, sequence_length] (here %s). Got a "
                         "mask tensor of shape %s." %
                         (expected_mask_tensor_shape, mask_tensor_shape))
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)

    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    def get_layer_instance(instance_or_cls, config, default_config):
      if isinstance(instance_or_cls, tf.keras.layers.Layer):
        return instance_or_cls
      else:
        if config is None:
          return instance_or_cls(**default_config)
        else:
          return instance_or_cls(**config)

    default_attention_cfg = {
        "num_heads": self._num_heads,
        "key_size": self._attention_head_size,
        "dropout": self._attention_dropout_rate,
        "name": "self_attention"
    }
    default_attention_cfg.update(common_kwargs)
    self._attention_layer = get_layer_instance(
        self._attention_cls,
        config=self._attention_cfg,
        default_config=default_attention_cfg)

    if self._feedforward_cls is not None:
      default_feedforward_cfg = {
          "intermediate_size": self._intermediate_size,
          "intermediate_activation": self._intermediate_activation,
          "dropout": self._dropout_rate,
          "name": "feedforward",
      }
      default_feedforward_cfg.update(common_kwargs)
      self._feedforward_block = get_layer_instance(
          self._feedforward_cls,
          config=self._feedforward_cfg,
          default_config=default_feedforward_cfg)
    else:
      self._feedforward_block = None

    self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            dtype=tf.float32))

    if self._feedforward_block is None:
      self._intermediate_dense = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, self._intermediate_size),
          bias_axes="d",
          activation=self._intermediate_activation,
          name="intermediate",
          **common_kwargs)
      self._output_dense = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, hidden_size),
          bias_axes="d",
          name="output",
          **common_kwargs)

    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

    super(TransformerScaffold, self).build(input_shape)

  def get_config(self):
    config = {
        "attention_cls":
            self._attention_layer,
        "feedforward_cls":
            self._feedforward_block,
        "num_attention_heads":
            self._num_heads,
        "intermediate_size":
            self._intermediate_size,
        "intermediate_activation":
            self._intermediate_activation,
        "dropout_rate":
            self._dropout_rate,
        "attention_dropout_rate":
            self._attention_dropout_rate,
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
    base_config = super(TransformerScaffold, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    attention_inputs = [input_tensor, input_tensor]

    attention_output = self._attention_layer(attention_inputs, attention_mask)
    attention_output = self._attention_dropout(attention_output)
    attention_output = self._attention_layer_norm(input_tensor +
                                                  attention_output)
    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      layer_output = self._output_dense(intermediate_output)
      layer_output = self._output_dropout(layer_output)
      # During mixed precision training, attention_output is from layer norm
      # and is always fp32 for now. Cast layer_output to fp32 for the subsequent
      # add.
      layer_output = tf.cast(layer_output, tf.float32)
      layer_output = self._output_layer_norm(layer_output + attention_output)
    else:
      layer_output = self._feedforward_block(attention_output)

    return layer_output
