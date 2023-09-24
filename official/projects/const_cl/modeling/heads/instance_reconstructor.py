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

"""The instance feature reconstructor head."""

from typing import Mapping

import tensorflow as tf

from official.projects.const_cl.modeling.heads import transformer_decoder
from official.vision.modeling.layers import roi_aligner


def _get_shape(x):
  """Helper function to return shape of a given tensor."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class InstanceReconstructor(tf.keras.layers.Layer):
  """The SSL head for reconstructing contextualized instance representations."""

  def __init__(self,
               context_level: int = 1,
               # parameters for projector
               num_output_channels: int = 1024,
               # parameters for RoiAligner
               crop_size: int = 4,
               sample_offset: float = 0.5,
               # parameters for TxDecoder
               num_tx_channels: int = 128,
               num_tx_layers: int = 3,
               num_tx_heads: int = 3,
               use_bias: bool = True,
               activation: str = 'gelu',
               dropout_rate: float = 0.0,
               layer_norm_epsilon: float = 1e-6,
               use_positional_embedding: bool = True,
               normalize_inputs: bool = True,
               **kwargs):
    """InstanceReconstructor SSL head initializer.

    Args:
      context_level: the number of context frame to use.
      num_output_channels: the number of final output channels.
      crop_size: the ROI aligner crop size.
      sample_offset: the ROI aligner sample offset.
      num_tx_channels: the Transformer decoder head channels.
      num_tx_layers: the number of Transformer decoder layers.
      num_tx_heads: the number of Transformer decoder heads per layer.
      use_bias: whether to use bias.
      activation: the activation function to use.
      dropout_rate: the dropout rate.
      layer_norm_epsilon: the layer norm epsilon.
      use_positional_embedding: whether to use positional embedding.
      normalize_inputs: whether to normalize input embeddings.
      **kwargs: the kwargs.
    """

    super().__init__(**kwargs)
    self._normalize_inputs = normalize_inputs
    self._context_level = context_level
    self._num_output_channels = num_output_channels
    self._crop_size = crop_size
    self._sample_offset = sample_offset
    self._num_tx_channels = num_tx_channels
    self._num_tx_layers = num_tx_layers
    self._num_tx_heads = num_tx_heads
    self._use_bias = use_bias
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._layer_norm_epsilon = layer_norm_epsilon
    self._use_positional_embedding = use_positional_embedding

    self._roi_aligner = roi_aligner.MultilevelROIAligner(
        crop_size=crop_size,
        sample_offset=sample_offset)

    if self._use_positional_embedding:
      self._spatial_mlp = [
          tf.keras.layers.Dense(
              4, use_bias=True, activation='relu', name='spatial_mlp_l1'),
          tf.keras.layers.Dense(
              8, use_bias=True, name='spatial_mlp_l2')]
      self._temporal_mlp = [
          tf.keras.layers.Dense(
              4, use_bias=True, activation='relu', name='temporal_mlp_l1'),
          tf.keras.layers.Dense(
              8, use_bias=True, name='temporal_mlp_l2')]

    self._attention_decoder = transformer_decoder.TransformerDecoder(
        num_channels=num_tx_channels,
        num_layers=num_tx_layers,
        num_heads=num_tx_heads,
        use_bias=use_bias,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon)

    self._projection_layer = tf.keras.layers.Dense(num_output_channels)

  def _get_memory_embeddings(self, inputs: tf.Tensor) -> tf.Tensor:
    """Uniformly samples frames to construct memory embeddings."""
    if self._context_level % 2 == 0:
      raise ValueError('context_level should be specified as odd number.')

    num_frames = tf.shape(inputs)[1]
    keyframe_index = num_frames // 2
    stride = num_frames // self._context_level
    start = self._context_level // 2 * -1
    stop = self._context_level // 2 + 1  # exclusive

    memories = []
    for idx in range(start, stop):
      idx = idx * stride + keyframe_index
      memories.append(inputs[:, idx, ...])

    memories = tf.stack(memories, axis=1)
    return memories

  def _add_positional_embedding(self, inputs: tf.Tensor) -> tf.Tensor:
    """Adds positional embeddings to the inputs tensor."""
    # Compute the locations using meshgrid.
    b, t, h, w = _get_shape(inputs)[:4]
    mesh = tf.meshgrid(tf.range(t), tf.range(h), tf.range(w), indexing='ij')
    position = tf.cast(
        tf.tile(
            tf.expand_dims(tf.stack(mesh, axis=-1), axis=0), [b, 1, 1, 1, 1]),
        tf.float32)

    # Make the positions relative to center point.
    # The mean of all position coordinates would be the center point anyway
    center_position = tf.reduce_mean(position, axis=[1, 2, 3], keepdims=True)
    position -= center_position

    # Apply learneable layers.
    temporal_position = position[..., :1]
    for mlp in self._temporal_mlp:
      temporal_position = mlp(temporal_position)
    spatial_position = position[..., 1:]
    for mlp in self._spatial_mlp:
      spatial_position = mlp(spatial_position)

    return tf.concat([inputs, temporal_position, spatial_position], axis=-1)

  def _keyframe_roi_pooling(self,
                            features: tf.Tensor,
                            boxes: tf.Tensor,
                            training: bool = True) -> tf.Tensor:
    """Pools ROI features on the keyframe.

    Args:
      features: a 5D tensor in shape [B, T, H, W, C].
      boxes: normalized box coordinates, a 4D tensor in shape [B, T', N, 4].
      training: whether in training mode.

    Returns:
      roi_feature: pooled ROI-features in shape [B, N, C].
    """
    if features.shape.ndims != 5:
      raise ValueError('Expected features is a rank-5 tensor. Got shape %s' %
                       features.shape)

    keyframe_index = tf.shape(boxes)[1] // 2
    t, h, w = _get_shape(features)[1:4]
    roi_features = {'0': features[:, t // 2, ...]}
    keyframe_boxes = boxes[:, keyframe_index, ...]
    unnormalized_boxes = keyframe_boxes * tf.convert_to_tensor(
        [h, w, h, w], keyframe_boxes.dtype)
    # roi_features in shape [B, N, h, w, C]
    roi_features = self._roi_aligner(
        roi_features, unnormalized_boxes, training=training)

    roi_shape = _get_shape(roi_features)
    # Perform average_pooling on ROI-pooled features.
    roi_features = tf.reshape(roi_features, [-1] + roi_shape[2:])
    roi_features = tf.reduce_mean(roi_features, axis=[1, 2])
    roi_features = tf.reshape(roi_features, roi_shape[:2] + roi_shape[-1:])
    return roi_features

  def call(self,
           inputs: Mapping[str, tf.Tensor],
           training: bool = False) -> Mapping[str, tf.Tensor]:
    """Forward calls.

    Args:
      inputs: the inputs dictionary contains
        'features': the instance embeddings in shape [2*B, T', H, W, C].
        'instances_positions': the instance boxes in shape [2*B, T, N, 4].
        'instances_mask': the validity mask for each instance position, in
          [2*B, T, N].
      training: whether in training mode.

    Returns:
      the context-guided reconstructed instance representations.
    """

    dense_embeddings_raw = inputs['features']
    instances_position = inputs['instances_position']
    instances_mask = inputs['instances_mask']

    if self._normalize_inputs:
      dense_embeddings_raw = tf.math.l2_normalize(dense_embeddings_raw, axis=-1)

    def _keyframe_temporal_pooling(inputs):
      t = tf.shape(inputs)[1] // 2
      return inputs[:, t:t+1, ...]

    dense_embeddings = _keyframe_temporal_pooling(dense_embeddings_raw)
    instances_position = _keyframe_temporal_pooling(instances_position)
    instances_mask = _keyframe_temporal_pooling(instances_mask)
    instances_mask_a, instances_mask_b = tf.split(
        tf.squeeze(instances_mask, axis=1), num_or_size_splits=2, axis=0)

    inst_embeddings = self._keyframe_roi_pooling(
        features=dense_embeddings,
        boxes=instances_position,
        training=training)

    inst_embeddings_a, inst_embeddings_b = tf.split(inst_embeddings, 2, axis=0)
    memory = self._get_memory_embeddings(dense_embeddings_raw)

    # Add the positional embeddings before roi_pooling and tx_decoder.
    if self._use_positional_embedding:
      memory = self._add_positional_embedding(memory)
    memory_a, memory_b = tf.split(memory, 2, axis=0)

    # Reconstruct inst_a2b by querying in memory_b.
    inst_embeddings_a2b = self._attention_decoder(
        inputs=inst_embeddings_a, memory=memory_b, training=training)
    inst_embeddings_a2b = inst_embeddings_a2b['hidden_states'][-1]
    inst_embeddings_a2b = self._projection_layer(
        inst_embeddings_a2b, training=training)
    # Reconstruct inst_b2a by querying in memory_a.
    inst_embeddings_b2a = self._attention_decoder(
        inputs=inst_embeddings_b, memory=memory_a, training=training)
    inst_embeddings_b2a = inst_embeddings_b2a['hidden_states'][-1]
    inst_embeddings_b2a = self._projection_layer(
        inst_embeddings_b2a, training=training)

    outputs = {
        'inst_a2b': inst_embeddings_a2b,
        'inst_b2a': inst_embeddings_b2a,
        'inst_a': inst_embeddings_a,
        'inst_b': inst_embeddings_b,
        'masks_a': instances_mask_a,
        'masks_b': instances_mask_b,
    }
    return outputs
