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

"""Keras-based transformer scaffold layer."""
# pylint: disable=g-classes-have-attributes

from absl import logging
import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling.layers import attention
from official.nlp.modeling.layers import util


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

  Args:
    num_attention_heads: Number of attention heads.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network.
    attention_cls: A class to instantiate attention layer, or a layer instance.
    attention_cfg: The config with which to instantiate `attention_cls`. Ignored
      if attention_cls is a layer instance or None. If `attention_cls` is a
      class, but `attention_cfg` is None, following kwargs will be used to
      instantiate the attention instance: {
        "num_heads": num_attention_heads,
        "key_dim": int(hidden_size // num_attention_heads),
        "dropout": attention_dropout_rate,
        "name": "self_attention" }, where `hidden_size` is the input tensor's
          last dimension.
    feedforward_cls: A class to instantiate feedforward layer, or a layer
      instance. If None, will use the standard feedforward layer as described in
      "Attention Is All You Need" paper. If not None, the instantiated
      feedforward layer is expected to take the output of attention as input and
      its output is this transformer layer's output.
    feedforward_cfg: The config with which to instantiate `feedforward_cls`.
      Ignored if feedforward_cls is a layer instance or is None. If
      `feedforward_cls` is a class, but `feedforward_cfg` is None, following
      kwargs will be used to instantiate the feedforward instance: {
        "inner_dim": inner_dim,
        "inner_activation": inner_activation,
        "dropout": dropout_rate,
        "name": "feedforward" }.
    dropout_rate: Dropout probability for the post-attention and output dropout.
    attention_dropout_rate: Dropout probability for within the attention layer.
    norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
    norm_epsilon: Epsilon value to initialize normalization layers.
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
               inner_dim=768,
               inner_activation=tf_utils.get_activation("gelu"),
               attention_cls=attention.MultiHeadAttention,
               attention_cfg=None,
               feedforward_cls=None,
               feedforward_cfg=None,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               norm_first=False,
               norm_epsilon=1e-12,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    inner_dim = kwargs.pop("intermediate_size", inner_dim)
    inner_activation = kwargs.pop("inner_activation", inner_activation)
    util.filter_kwargs(kwargs)
    super().__init__(**kwargs)

    self._attention_cfg = attention_cfg
    self._attention_cls = attention_cls
    self._feedforward_cls = feedforward_cls
    self._feedforward_cfg = feedforward_cfg
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._attention_dropout_rate = attention_dropout_rate
    self._dropout_rate = dropout_rate
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError(
          "The type of input shape argument is not supported, got: %s" %
          type(input_shape))

    if len(input_tensor_shape.as_list()) != 3:
      raise ValueError(
          "TransformerScaffold expects a three-dimensional input of "
          "shape [batch, sequence, width].")
    hidden_size = input_tensor_shape[-1]
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)

    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    def get_layer_instance(instance_or_cls, config, default_config):
      if isinstance(instance_or_cls, tf.keras.layers.Layer):
        return instance_or_cls
      elif isinstance(instance_or_cls, dict):
        return get_layer_instance(
            tf.keras.utils.deserialize_keras_object(instance_or_cls),
            config,
            default_config,
        )
      else:
        if config is None:
          return instance_or_cls(**default_config)
        else:
          return instance_or_cls(**config)

    default_attention_cfg = {
        "kernel_initializer": tf_utils.clone_initializer(
            self._kernel_initializer),
        "bias_initializer": tf_utils.clone_initializer(self._bias_initializer),
        "num_heads": self._num_heads,
        "key_dim": self._attention_head_size,
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
          "kernel_initializer": tf_utils.clone_initializer(
              self._kernel_initializer),
          "bias_initializer": tf_utils.clone_initializer(
              self._bias_initializer),
          "inner_dim": self._inner_dim,
          "inner_activation": self._inner_activation,
          # TODO(hongkuny): try to update all ffn block args.
          "intermediate_size": self._inner_dim,
          "intermediate_activation": self._inner_activation,
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

    # self._dropout_rate controls dropout rates at two places:
    # after attention, and after FFN.
    self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))

    if self._feedforward_block is None:
      self._intermediate_dense = tf.keras.layers.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, self._inner_dim),
          bias_axes="d",
          name="intermediate",
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
          **common_kwargs)
      policy = tf.keras.mixed_precision.global_policy()
      if policy.name == "mixed_bfloat16":
        # bfloat16 causes BERT with the LAMB optimizer to not converge
        # as well, so we use float32.
        # TODO(b/154538392): Investigate this.
        policy = tf.float32
      self._intermediate_activation_layer = tf.keras.layers.Activation(
          self._inner_activation, dtype=policy)
      self._output_dense = tf.keras.layers.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, hidden_size),
          bias_axes="d",
          name="output",
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
          **common_kwargs)

    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)

    super().build(input_shape)
    logging.info("%s configs: %s", self.__class__.__name__, self.get_config())

  def get_config(self):
    config = {
        "attention_cls": self._attention_layer,
        "feedforward_cls": self._feedforward_block,
        "num_attention_heads": self._num_heads,
        "inner_dim": self._inner_dim,
        "inner_activation": self._inner_activation,
        "dropout_rate": self._dropout_rate,
        "attention_dropout_rate": self._attention_dropout_rate,
        "norm_first": self._norm_first,
        "norm_epsilon": self._norm_epsilon,
        "kernel_initializer": tf_utils.serialize_initializer(
            self._kernel_initializer, use_legacy_format=True
        ),
        "bias_initializer": tf_utils.serialize_initializer(
            self._bias_initializer, use_legacy_format=True
        ),
        "kernel_regularizer": tf_utils.serialize_regularizer(
            self._kernel_regularizer, use_legacy_format=True
        ),
        "bias_regularizer": tf_utils.serialize_regularizer(
            self._bias_regularizer, use_legacy_format=True
        ),
        "activity_regularizer": tf_utils.serialize_regularizer(
            self._activity_regularizer, use_legacy_format=True
        ),
        "kernel_constraint": tf_utils.serialize_constraint(
            self._kernel_constraint, use_legacy_format=True
        ),
        "bias_constraint": tf_utils.serialize_constraint(
            self._bias_constraint, use_legacy_format=True
        ),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError("Unexpected inputs to %s with length at %d" %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if key_value is None:
      key_value = input_tensor

    if self._norm_first:
      source_tensor = input_tensor
      input_tensor = self._attention_layer_norm(input_tensor, training=training)

    attention_output = self._attention_layer(
        query=input_tensor, value=key_value, attention_mask=attention_mask,
        training=training)
    attention_output = self._attention_dropout(attention_output,
                                               training=training)

    if self._norm_first:
      attention_output = source_tensor + attention_output
    else:
      attention_output = self._attention_layer_norm(input_tensor +
                                                    attention_output,
                                                    training=training)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output,
                                                 training=training)

    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output, training=training)
      layer_output = self._output_dropout(layer_output, training=training)
      # During mixed precision training, attention_output is from layer norm
      # and is always fp32 for now. Cast layer_output to fp32 for the subsequent
      # add.
      layer_output = tf.cast(layer_output, tf.float32)
      if self._norm_first:
        layer_output = source_attention_output + layer_output
      else:
        layer_output = self._output_layer_norm(layer_output + attention_output,
                                               training=training)
    else:
      if self._norm_first:
        # if norm_first, assume the feedforward block will not apply layer norm
        layer_output = self._feedforward_block(attention_output,
                                               training=training)
        layer_output += source_attention_output
      else:
        # Attention: if not norm_first, assume that the feedforwad does apply
        # layer norm. The feedford also apply residual connection. Please
        # read  the `GatedFeedforward` as a concrete example.
        layer_output = self._feedforward_block(attention_output,
                                               training=training)

    return layer_output
