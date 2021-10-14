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

"""Contains a collection of util functions for model construction."""
from typing import Any, Dict, Optional, Union

import tensorflow as tf


def sample_random_sequence(model_input, num_frames, num_samples):
  """Samples a random sequence of frames of size num_samples.

  Args:
    model_input: tensor of shape [batch_size x max_frames x feature_size]
    num_frames: tensor of shape [batch_size x 1]
    num_samples: a scalar indicating the number of samples

  Returns:
    reshaped model_input in [batch_size x 'num_samples' x feature_size]
  """

  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)


def sample_random_frames(model_input, num_frames, num_samples):
  """Samples a random set of frames of size num_samples.

  Args:
    model_input: tensor of shape [batch_size x max_frames x feature_size]
    num_frames: tensor of shape [batch_size x 1]
    num_samples (int): a scalar indicating the number of samples

  Returns:
    reshaped model_input in [batch_size x 'num_samples' x feature_size]
  """
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random.uniform([batch_size, num_samples]),
          tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)


def frame_pooling(frames, method):
  """Pools over the frames of a video.

  Args:
    frames: tensor of shape [batch_size, num_frames, feature_size].
    method: string indicating pooling method, one of: "average", "max",
      "attention", or "none".

  Returns:
    tensor of shape [batch_size, feature_size] for average, max, or
    attention pooling, and shape [batch_size*num_frames, feature_size]
    for none pooling.
  Raises:
    ValueError: if method is other than "average", "max", "attention", or
    "none".
  """
  if method == "average":
    reduced = tf.reduce_mean(frames, 1)
  elif method == "max":
    reduced = tf.reduce_max(frames, 1)
  elif method == "none":
    feature_size = frames.shape_as_list()[2]
    reduced = tf.reshape(frames, [-1, feature_size])
  else:
    raise ValueError("Unrecognized pooling method: %s" % method)

  return reduced


def context_gate(
    input_features,
    normalizer_fn=None,
    normalizer_params: Optional[Dict[str, Any]] = None,
    kernel_initializer: Union[
        str, tf.keras.regularizers.Regularizer] = "glorot_uniform",
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    bias_initializer: Union[str, tf.keras.regularizers.Regularizer] = "zeros",
    hidden_layer_size: int = 0,
    pooling_method: Optional[str] = None,
    additive_residual: bool = False):
  """Context Gating.

  More details: https://arxiv.org/pdf/1706.06905.pdf.

  Args:
    input_features: a tensor of at least rank 2.
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
      features, [batch_size*num_frames, h, w, feature_dim]. If the inputs are a
      set of local features and pooling_method is not None, will pool features
      across all but the batch_size dimension using the specified pooling
      method, and pass the aggregated features as context to the gating layer.
      For a list of pooling methods, see the frame_pooling() function.
    additive_residual: If true, will use ReLu6-activated (additive) residual
      connections instead of Sigmoid-activated (multiplicative) connections when
      combining the input_features with the context gating branch.

  Returns:
    A tensor with the same shape as input_features.
  """
  if normalizer_params is None:
    normalizer_params = {}
  with tf.name_scope("ContextGating"):
    num_dimensions = len(input_features.shape.as_list())
    feature_size = input_features.shape.as_list()[-1]
    if pooling_method:
      assert num_dimensions > 2
      # Collapse the inner axes of the original features shape into a 3D tensor
      original_shape = tf.shape(input_features)
      # The last dimension will change after concatenating the context
      new_shape = tf.concat(
          [original_shape[:-1],
           tf.constant([2 * feature_size])], 0)
      batch_size = original_shape[0]
      reshaped_features = tf.reshape(input_features,
                                     [batch_size, -1, feature_size])
      num_features = tf.shape(reshaped_features)[1]
      # Pool the feature channels across the inner axes to get global context
      context_features = frame_pooling(reshaped_features, pooling_method)
      context_features = tf.expand_dims(context_features, 1)
      # Replicate the global context features and concat to the local features.
      context_features = tf.tile(context_features, [1, num_features, 1])
      context_features = tf.concat([reshaped_features, context_features], 2)
      context_features = tf.reshape(context_features, shape=new_shape)
    else:
      context_features = input_features

    if hidden_layer_size >= 2:
      gates_bottleneck = tf.keras.layers.Dense(
          hidden_layer_size,
          activation="relu6",
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
      )(
          context_features)
      if normalizer_fn:
        gates_bottleneck = normalizer_fn(**normalizer_params)(gates_bottleneck)
    else:
      gates_bottleneck = context_features

    activation_fn = (tf.nn.relu6 if additive_residual else tf.nn.sigmoid)
    gates = tf.keras.layers.Dense(
        feature_size,
        activation=activation_fn,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
    )(
        gates_bottleneck)
    if normalizer_fn:
      gates = normalizer_fn(**normalizer_params)(gates)

    if additive_residual:
      input_features += gates
    else:
      input_features *= gates

    return input_features
