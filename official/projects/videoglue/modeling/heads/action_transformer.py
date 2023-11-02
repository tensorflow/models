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

"""The implementation of action transformer head."""
from typing import Mapping, Optional

import tensorflow as tf, tf_keras

from official.projects.videoglue.modeling.heads import simple
from official.projects.videoglue.modeling.heads import transformer_decoder
from official.vision.modeling.layers import roi_aligner


def _get_shape(x: tf.Tensor):
  """Helper function to return shape of a given tensor."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class ActionTransformerHead(tf_keras.layers.Layer):
  """A Video Action Transformer Head.

  Reference: Girdhar, Rohit et. al. "Video action transformer network." In CVPR
    2019. https://arxiv.org/abs/1812.02707
  """

  def __init__(
      self,
      # parameters for classifier
      num_hidden_layers: int,
      num_hidden_channels: int,
      use_sync_bn: bool,
      num_classes: int,
      activation: str = 'relu',
      dropout_rate: float = 0.0,
      classifier_norm_epsilon: float = 1e-5,
      # parameters for RoiAligner
      crop_size: int = 7,
      sample_offset: float = 0.5,
      # parameters for TxDecoder
      num_tx_channels: int = 768,
      num_tx_layers: int = 12,
      num_tx_heads: int = 12,
      use_bias: bool = True,
      tx_activation: str = 'gelu',
      attention_dropout_rate: float = 0.0,
      layer_norm_epsilon: float = 1e-6,
      use_positional_embedding: bool = True,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      name: str = 'action_transformer_classifier',
      **kwargs,
  ):
    """Initializer.

    Args:
      num_hidden_layers: The number of hidden layer in the final classifier.
      num_hidden_channels: The number of hidden channels in the classifier.
      use_sync_bn: Whether to use the sync batch norm in the classifier.
      num_classes: The final number of classes prediction.
      activation: The activation used in the classifier.
      dropout_rate: The dropout rate for the classifier.
      classifier_norm_epsilon: Batchnorm epsilon for the classifier.
      crop_size: The RoI-Align output crop size.
      sample_offset: The RoI-Align sample offset.
      num_tx_channels: The number of channels in the transformer.
      num_tx_layers: The number of transformer layer.
      num_tx_heads: The number of transformer head.
      use_bias: Whether to use bias in the transformer decoder.
      tx_activation: The activation function to use in the transformer.
      attention_dropout_rate: The attention dropout rate.
      layer_norm_epsilon: The layer norm epsilon.
      use_positional_embedding: Whether to use positional embedding.
      kernel_regularizer: tf_keras.regularizers.Regularizer object.
      bias_regularizer: tf_keras.regularizers.Regularizer object.
      name: The head name.
      **kwargs: Keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._num_hidden_layers = num_hidden_layers
    self._num_hidden_channels = num_hidden_channels
    self._use_sync_bn = use_sync_bn
    self._num_classes = num_classes
    self._dropout_rate = dropout_rate
    self._use_positional_embedding = use_positional_embedding
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if self._use_positional_embedding:
      self._spatial_mlp = [
          tf_keras.layers.Dense(
              4,
              use_bias=True,
              activation='relu',
              name='spatial_mlp_l1',
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer),
          tf_keras.layers.Dense(
              8,
              use_bias=True,
              name='spatial_mlp_l2',
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer),
      ]
      self._temporal_mlp = [
          tf_keras.layers.Dense(
              4,
              use_bias=True,
              activation='relu',
              name='temporal_mlp_l1',
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer),
          tf_keras.layers.Dense(
              8,
              use_bias=True,
              name='temporal_mlp_l2',
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer),
      ]

    self._roi_aligner = roi_aligner.MultilevelROIAligner(
        crop_size=crop_size,
        sample_offset=sample_offset)
    self._max_pooler = tf_keras.layers.MaxPool2D(
        pool_size=(crop_size, crop_size),
        strides=1,
        padding='valid')

    if num_tx_layers > 0:
      self._attention_decoder = transformer_decoder.TransformerDecoder(
          num_channels=num_tx_channels,
          num_layers=num_tx_layers,
          num_heads=num_tx_heads,
          use_bias=use_bias,
          activation=tx_activation,
          dropout_rate=attention_dropout_rate,
          layer_norm_epsilon=layer_norm_epsilon,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)
    else:
      self._attention_decoder = None

    self._dropout_layer = tf_keras.layers.Dropout(dropout_rate)
    self._classifier = simple.MLP(
        num_hidden_layers=self._num_hidden_layers,
        num_hidden_channels=self._num_hidden_channels,
        num_output_channels=self._num_classes,
        use_sync_bn=self._use_sync_bn,
        norm_epsilon=classifier_norm_epsilon,
        activation=activation,
        normalize_inputs=False,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def _add_positional_embedding(self, inputs):
    """Adds positional embeddings to the inputs tensor."""
    # Compute the locations using meshgrid.
    b, t, h, w = _get_shape(inputs)[:4]

    mesh = tf.meshgrid(tf.range(t), tf.range(h), tf.range(w), indexing='ij')
    position = tf.cast(
        tf.tile(
            tf.expand_dims(tf.stack(mesh, axis=-1), axis=0), [b, 1, 1, 1, 1]),
        tf.float32)

    # Make the positions relative to center point
    # The mean of all position coordinates would be the center point anyway.
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

  def call(self,
           inputs: Mapping[str, tf.Tensor],
           training: bool = False) -> Mapping[str, tf.Tensor]:
    """Forward calls.

    Args:
      inputs: the inputs dictionary contains
        'features': the instance embeddings in shape [B, T', H, W, C].
        'instances_positions': the instance boxes in shape [B, T, N, 4].
        'instances_mask': the validity mask for each instance position, in
          [B, T, N].
      training: whether in training mode.

    Returns:
      the final action classification.
    """
    features = inputs['features']
    instances_position = inputs['instances_position']
    if features.shape.ndims != 5:
      raise ValueError('Expected features is a rank-5 tensor. Got shape %s' %
                       features.shape)

    if self._use_positional_embedding:
      features = self._add_positional_embedding(features)

    # Perform RoI-pooling.
    h, w = _get_shape(features)[2:4]
    roi_features = {'0': tf.reduce_mean(features, axis=1)}
    unnormalized_boxes = instances_position * tf.convert_to_tensor(
        [h, w, h, w], instances_position.dtype)
    # roi_features in shape [B, N, h, w, C]
    roi_features = self._roi_aligner(
        roi_features, unnormalized_boxes, training=training)
    # Perform average_pooling on ROI-pooled features.
    b, n, ch, cw, cc = _get_shape(roi_features)
    roi_features = tf.reshape(roi_features, [b * n, ch, cw, cc])
    roi_features = self._max_pooler(roi_features)
    roi_features = tf.reshape(roi_features, [b, n, cc])

    if self._attention_decoder is None:
      predictions = roi_features
    else:
      outputs = self._attention_decoder(inputs=roi_features,
                                        memory=features,
                                        training=training)
      # Get last hidden states and perform final classification.
      predictions = outputs['hidden_states'][-1]

    predictions = self._dropout_layer(predictions, training=training)
    outputs = self._classifier(predictions, training=training)
    return outputs
