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

"""Builds the Video Action Transformer Network."""
from typing import Mapping, Optional, Tuple

import tensorflow as tf, tf_keras

from official.projects.videoglue.configs import spatiotemporal_action_localization as cfg
from official.projects.videoglue.modeling.backbones import vit_3d  # pylint: disable=unused-import
from official.projects.videoglue.modeling.heads import action_transformer
from official.vision.modeling import backbones
from official.vision.modeling import factory_3d as model_factory


@tf_keras.utils.register_keras_serializable(package='Vision')
class VideoActionTransformerModel(tf_keras.Model):
  """A Video Action Transformer Network.

  Reference: Girdhar, Rohit et. al. "Video action transformer network." In CVPR
    2019. https://arxiv.org/abs/1812.02707
  """

  def __init__(
      self,
      backbone: tf_keras.Model,
      num_classes: int,
      endpoint_name: str,
      # parameters for classifier
      num_hidden_layers: int,
      num_hidden_channels: int,
      use_sync_bn: bool,
      activation: str = 'relu',
      dropout_rate: float = 0.0,
      # parameters for RoiAligner
      crop_size: int = 7,
      sample_offset: float = 0.5,
      # parameters for TxDecoder
      num_tx_channels: int = 128,
      num_tx_layers: int = 3,
      num_tx_heads: int = 3,
      use_bias: bool = True,
      tx_activation: str = 'gelu',
      attention_dropout_rate: float = 0.0,
      layer_norm_epsilon: float = 1e-6,
      use_positional_embedding: bool = True,
      input_specs: Optional[Mapping[str, tf_keras.layers.InputSpec]] = None,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialization function.

    Args:
      backbone: A backbone network.
      num_classes: The final number of classes prediction.
      endpoint_name: The endpoint name from the backbone to extract features.
      num_hidden_layers: The number of hidden layer in the final classifier.
      num_hidden_channels: The number of hidden channels in the classifier.
      use_sync_bn: Whether to use the sync batch norm in the classifier.
      activation: The activation used in the classifier.
      dropout_rate: The dropout rate for the classifier.
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
      input_specs: Specs of the input tensor.
      kernel_regularizer: tf_keras.regularizers.Regularizer object.
      bias_regularizer: tf_keras.regularizers.Regularizer object.
      **kwargs: Keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image':
              tf_keras.layers.InputSpec(shape=[None, None, None, None, 3]),
          'instances_position':
              tf_keras.layers.InputSpec(shape=[None, None, 4]),
      }

    self._num_classes = num_classes
    self._endpoint_name = endpoint_name
    self._num_hidden_layers = num_hidden_layers
    self._num_hidden_channels = num_hidden_channels
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._crop_size = crop_size
    self._sample_offset = sample_offset
    self._num_tx_channels = num_tx_channels
    self._num_tx_layers = num_tx_layers
    self._num_tx_heads = num_tx_heads
    self._use_bias = use_bias
    self._tx_activation = tx_activation
    self._attention_dropout_rate = attention_dropout_rate
    self._layer_norm_epsilon = layer_norm_epsilon
    self._use_positional_embedding = use_positional_embedding
    self._input_specs = input_specs
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    inputs, outputs = self._build_model(backbone, input_specs)
    super().__init__(inputs=inputs, outputs=outputs, **kwargs)
    # Move backbone after super() call so Keras is happy.
    self._backbone = backbone

  def _build_model(
      self, backbone: tf_keras.Model,
      input_specs: Mapping[str, tf_keras.layers.InputSpec]
  ) -> Tuple[Mapping[str, tf.Tensor], tf.Tensor]:
    """Builds the model network.

    Args:
      backbone: the model backbone.
      input_specs: the model input spec to use.

    Returns:
      Inputs and outputs as a tuple. Inputs are expected to be a dict with
      base input and positions. Outputs are predictions per instance.
    """

    inputs = {
        k: tf_keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    endpoints = backbone(inputs['image'])
    features = endpoints[self._endpoint_name]

    tx_inputs = {
        'features': features,
        'instances_position': inputs['instances_position'],
    }
    outputs = action_transformer.ActionTransformerHead(
        num_hidden_layers=self._num_hidden_layers,
        num_hidden_channels=self._num_hidden_channels,
        use_sync_bn=self._use_sync_bn,
        num_classes=self._num_classes,
        activation=self._activation,
        dropout_rate=self._dropout_rate,
        # parameters for RoiAligner
        crop_size=self._crop_size,
        sample_offset=self._sample_offset,
        # parameters for TxDecoder
        num_tx_channels=self._num_tx_channels,
        num_tx_layers=self._num_tx_layers,
        num_tx_heads=self._num_tx_heads,
        use_bias=self._use_bias,
        tx_activation=self._tx_activation,
        attention_dropout_rate=self._attention_dropout_rate,
        layer_norm_epsilon=self._layer_norm_epsilon,
        use_positional_embedding=self._use_positional_embedding,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(tx_inputs)
    return inputs, outputs

  @property
  def backbone(self) -> tf_keras.Model:
    """Returns the backbone of the model."""
    return self._backbone


@model_factory.register_model_builder('video_action_transformer_model')
def build_video_action_transformer_model(
    input_specs_dict: Mapping[str, tf_keras.layers.InputSpec],
    model_config: cfg.VideoActionTransformerModel,
    num_classes: int,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> VideoActionTransformerModel:
  """Builds the video action localziation model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs_dict['image'],
      backbone_config=model_config.backbone,
      norm_activation_config=model_config.norm_activation,
      l2_regularizer=l2_regularizer)

  # Norm layer type in the MLP head should same with backbone.
  if (model_config.norm_activation.use_sync_bn
      != model_config.head.use_sync_bn):
    raise ValueError('Should use the same batch normalization type.')

  return VideoActionTransformerModel(
      backbone=backbone,
      input_specs=input_specs_dict,
      num_classes=num_classes,
      endpoint_name=model_config.endpoint_name,
      # parameters for classifier
      num_hidden_layers=model_config.head.num_hidden_layers,
      num_hidden_channels=model_config.head.num_hidden_channels,
      use_sync_bn=model_config.head.use_sync_bn,
      activation=model_config.head.activation,
      dropout_rate=model_config.head.dropout_rate,
      crop_size=model_config.head.crop_size,
      sample_offset=model_config.head.sample_offset,
      num_tx_channels=model_config.head.num_tx_channels,
      num_tx_layers=model_config.head.num_tx_layers,
      num_tx_heads=model_config.head.num_tx_heads,
      use_bias=model_config.head.use_bias,
      tx_activation=model_config.head.tx_activation,
      attention_dropout_rate=model_config.head.attention_dropout_rate,
      layer_norm_epsilon=model_config.head.layer_norm_epsilon,
      use_positional_embedding=model_config.head.use_positional_embedding,
      kernel_regularizer=l2_regularizer)
