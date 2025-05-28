# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Contains a collection of util functions for model construction."""

from typing import Any, Dict, Optional, Union

import tensorflow as tf, tf_keras

from official.projects.yt8m.modeling import yt8m_model_utils


class ContextGate(tf_keras.layers.Layer):
  """Context Gating. More details: https://arxiv.org/pdf/1706.06905.pdf."""

  def __init__(
      self,
      normalizer_fn=None,
      normalizer_params: Optional[Dict[str, Any]] = None,
      kernel_initializer: Union[
          str, tf_keras.regularizers.Regularizer
      ] = "glorot_uniform",
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_initializer: Union[str, tf_keras.regularizers.Regularizer] = "zeros",
      hidden_layer_size: int = 0,
      pooling_method: Optional[str] = None,
      additive_residual: bool = False,
      name: Optional[str] = None,
  ):
    """Initialization of context gate.

    Args:
      normalizer_fn: Normalization function to use instead of `biases` (e.g.
        tf.contrib.layers.batch_norm). If None, bias is added.
      normalizer_params: Normalization function parameters.
      kernel_initializer: Weight initializer to use instead of Xavier (e.g.
        tf.contrib.layers.variance_scaling_initializer).
      kernel_regularizer: Weight regularizer to use instead of None (e.g.,
        tf.contrib.layers.l2_regularizer(l2_penalty)).
      bias_initializer: Biases initializer to use (default tf.zeros_initializer)
      hidden_layer_size: Dimensionality of the context gating hidden layer size,
        if any. If None, will apply a fully-connected context gating layer with
        shape [input_size x input_size]. If set to an int N, will factorize the
        context gating layer into [input_size x N] x [N x input_size] as in the
        squeeze-and-excitation block from https://arxiv.org/pdf/1709.01507.pdf.
      pooling_method: Whether to perform global pooling of the local features
        before applying the context gating layer. This is relevant only if the
        input_features tensor has rank > 2, e.g., it's a sequence of frame
        features, [batch_size, num_frames, feature_dim], or spatial convolution
        features, [batch_size*num_frames, h, w, feature_dim]. If the inputs are
        a set of local features and pooling_method is not None, will pool
        features across all but the batch_size dimension using the specified
        pooling method, and pass the aggregated features as context to the
        gating layer. For a list of pooling methods, see the frame_pooling()
        function.
      additive_residual: If true, will use ReLu6-activated (additive) residual
        connections instead of Sigmoid-activated (multiplicative) connections
        when combining the input_features with the context gating branch.
      name: Optional `str` name of the module.

    Returns:
      A tensor with the same shape as input_features.
    """
    super().__init__(name=name)
    self._normalizer_fn = normalizer_fn
    self._normalizer_params = normalizer_params or {}
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_initializer = bias_initializer
    self._hidden_layer_size = hidden_layer_size
    self._pooling_method = pooling_method
    self._additive_residual = additive_residual

    if hidden_layer_size >= 2:
      self._gates_bottleneck = tf_keras.layers.Dense(
          hidden_layer_size,
          activation="relu6",
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          name="bottleneck",
      )
      if self._normalizer_fn:
        self._gates_bottleneck_norm = self._normalizer_fn(
            **self._normalizer_params,
            name="bottleneck_norm",
        )

  def build(self, input_shape):
    super().build(input_shape)
    feature_size = input_shape[-1]
    activation_fn = tf.nn.relu6 if self._additive_residual else tf.nn.sigmoid
    self._gates = tf_keras.layers.Dense(
        feature_size,
        activation=activation_fn,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        name="gates_dense",
    )
    if self._normalizer_fn:
      self._gates_norm = self._normalizer_fn(
          **self._normalizer_params,
          name="gates_norm",
      )

  def call(self, inputs: tf.Tensor):
    num_dimensions = len(inputs.shape.as_list())
    feature_size = inputs.shape.as_list()[-1]

    if self._pooling_method:
      assert num_dimensions > 2
      # Collapse the inner axes of the original features shape into a 3D tensor
      original_shape = tf.shape(inputs)
      # The last dimension will change after concatenating the context
      new_shape = tf.concat(
          [original_shape[:-1], tf.constant([2 * feature_size])], 0
      )
      batch_size = original_shape[0]
      reshaped_features = tf.reshape(inputs, [batch_size, -1, feature_size])
      num_features = tf.shape(reshaped_features)[1]
      # Pool the feature channels across the inner axes to get global context
      context_features = yt8m_model_utils.frame_pooling(
          reshaped_features, self._pooling_method
      )
      context_features = tf.expand_dims(context_features, 1)
      # Replicate the global context features and concat to the local features.
      context_features = tf.tile(context_features, [1, num_features, 1])
      context_features = tf.concat([reshaped_features, context_features], 2)
      context_features = tf.reshape(context_features, shape=new_shape)
    else:
      # num_dimensions should be 2
      context_features = tf.identity(inputs)

    if self._hidden_layer_size >= 2:
      gates_bottleneck = self._gates_bottleneck(context_features)
      if self._normalizer_fn:
        gates_bottleneck = self._gates_bottleneck_norm(gates_bottleneck)
    else:
      gates_bottleneck = tf.identity(context_features)

    gates = self._gates(gates_bottleneck)
    if self._normalizer_fn:
      gates = self._gates_norm(gates)

    if self._additive_residual:
      inputs += tf.cast(gates, inputs.dtype)
    else:
      inputs *= tf.cast(gates, inputs.dtype)

    return inputs
