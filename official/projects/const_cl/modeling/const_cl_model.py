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

"""Builds ConST-CL SSL models."""
from typing import Mapping, Optional

import tensorflow as tf, tf_keras

from official.projects.const_cl.configs import const_cl as const_cl_cfg
from official.projects.const_cl.modeling.heads import instance_reconstructor
from official.projects.const_cl.modeling.heads import simple

from official.vision.modeling import backbones
from official.vision.modeling import factory_3d as model_factory

layers = tf_keras.layers


class ConstCLModel(tf_keras.Model):
  """A ConST-CL SSL model class builder."""

  def __init__(
      self,
      backbone,
      input_specs: Optional[Mapping[str, tf_keras.layers.InputSpec]] = None,
      # global_head
      num_hidden_layers: int = 3,
      num_hidden_channels: int = 1024,
      num_output_channels: int = 128,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 1e-5,
      activation: Optional[str] = None,
      normalize_global_features: bool = False,
      # local_head
      context_level: int = 1,
      num_tx_output_channels: int = 1024,
      crop_size: int = 4,
      sample_offset: float = 0.5,
      num_tx_channels: int = 128,
      num_tx_layers: int = 3,
      num_tx_heads: int = 3,
      use_bias: bool = True,
      tx_activation: str = 'gelu',
      dropout_rate: float = 0.0,
      layer_norm_epsilon: float = 1e-6,
      use_positional_embedding: bool = True,
      normalize_local_features: bool = True,
      **kwargs):
    """Video Classification initialization function.

    Args:
      backbone: a 3d backbone network.
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
      num_hidden_layers: the number of hidden layers in the MLP.
      num_hidden_channels: the number of hidden nodes in the MLP.
      num_output_channels: the number of final output nodes in the MLP.
      use_sync_bn: whether to use sync batch norm in the MLP.
      norm_momentum: the MLP batch norm momentum.
      norm_epsilon: the MLP batch norm epsilon.
      activation: the MLP activation function.
      normalize_global_features: whether to normalize inputs to the MLP.
      context_level: the number of context frame to use.
      num_tx_output_channels: the number of final output channels for instance
        reconstrcutor.
      crop_size: the ROI aligner crop size.
      sample_offset: the ROI aligner sample offset.
      num_tx_channels: the Transformer decoder head channels.
      num_tx_layers: the number of Transformer decoder layers.
      num_tx_heads: the number of Transformer decoder heads per layer.
      use_bias: whether to use bias in the Transformer.
      tx_activation: the activation function to use in the Transformer.
      dropout_rate: the dropout rate for Transformer.
      layer_norm_epsilon: the layer norm epsilon.
      use_positional_embedding: whether to use positional embedding.
      normalize_local_features: whether to normalize input embeddings.
      **kwargs: keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3]),
          'instances_position': layers.InputSpec(shape=[None, None, None, 4]),
          'instances_mask': layers.InputSpec(shape=[None, None, None]),
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'num_hidden_layers': num_hidden_layers,
        'num_hidden_channels': num_hidden_channels,
        'num_output_channels': num_output_channels,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'normalize_global_features': normalize_global_features,
        'context_level': context_level,
        'num_tx_output_channels': num_tx_output_channels,
        'crop_size': crop_size,
        'sample_offset': sample_offset,
        'num_tx_channels': num_tx_channels,
        'num_tx_layers': num_tx_layers,
        'num_tx_heads': num_tx_heads,
        'use_bias': use_bias,
        'tx_activation': tx_activation,
        'dropout_rate': dropout_rate,
        'layer_norm_epsilon': layer_norm_epsilon,
        'use_positional_embedding': use_positional_embedding,
        'normalize_local_features': normalize_local_features,
    }
    self._input_specs = input_specs
    self._backbone = backbone

    inputs = {
        k: tf_keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    endpoints = backbone(inputs['image'])

    res5 = endpoints['5']
    res5 = tf_keras.layers.GlobalAveragePooling3D()(res5)
    res5_1 = endpoints['5_1']

    global_embeddings = simple.MLP(
        num_hidden_layers=num_hidden_layers,
        num_hidden_channels=num_hidden_channels,
        num_output_channels=num_output_channels,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        activation=activation,
        normalize_inputs=normalize_global_features)(res5)

    instance_inputs = {
        'features': res5_1,
        'instances_position': inputs['instances_position'],
        'instances_mask': inputs['instances_mask'],
    }
    instances_outputs = instance_reconstructor.InstanceReconstructor(
        context_level=context_level,
        # parameters for projector
        num_output_channels=num_tx_output_channels,
        # parameters for RoiAligner
        crop_size=crop_size,
        sample_offset=sample_offset,
        # parameters for TxDecoder
        num_tx_channels=num_tx_channels,
        num_tx_layers=num_tx_layers,
        num_tx_heads=num_tx_heads,
        use_bias=use_bias,
        activation=tx_activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        use_positional_embedding=use_positional_embedding,
        normalize_inputs=normalize_local_features)(instance_inputs)

    outputs = instances_outputs
    outputs['global_embeddings'] = global_embeddings
    super().__init__(inputs=inputs, outputs=outputs, **kwargs)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self):
    return self._backbone

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@model_factory.register_model_builder('const_cl_model')
def build_const_cl_pretrain_model(
    input_specs_dict: Mapping[str, tf_keras.layers.InputSpec],
    model_config: const_cl_cfg.ConstCLModel,
    num_classes: int,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> ConstCLModel:
  """Builds the ConST-CL video ssl model."""
  del num_classes
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs_dict['image'],
      backbone_config=model_config.backbone,
      norm_activation_config=model_config.norm_activation,
      l2_regularizer=l2_regularizer)

  # Norm layer type in the MLP head should same with backbone
  if (model_config.norm_activation.use_sync_bn
      != model_config.global_head.use_sync_bn):
    raise ValueError('Should use the same batch normalization type.')

  return ConstCLModel(
      backbone=backbone,
      input_specs=input_specs_dict,
      # global_head
      num_hidden_channels=model_config.global_head.num_hidden_channels,
      num_hidden_layers=model_config.global_head.num_hidden_layers,
      num_output_channels=model_config.global_head.num_output_channels,
      use_sync_bn=model_config.global_head.use_sync_bn,
      norm_momentum=model_config.global_head.norm_momentum,
      norm_epsilon=model_config.global_head.norm_epsilon,
      activation=model_config.global_head.activation,
      normalize_global_features=model_config.global_head.normalize_inputs,
      # local_head
      context_level=model_config.local_head.context_level,
      num_tx_output_channels=model_config.local_head.num_output_channels,
      crop_size=model_config.local_head.crop_size,
      sample_offset=model_config.local_head.sample_offset,
      num_tx_channels=model_config.local_head.num_tx_channels,
      num_tx_layers=model_config.local_head.num_tx_layers,
      num_tx_heads=model_config.local_head.num_tx_heads,
      use_bias=model_config.local_head.use_bias,
      tx_activation=model_config.local_head.activation,
      dropout_rate=model_config.local_head.dropout_rate,
      layer_norm_epsilon=model_config.local_head.layer_norm_epsilon,
      use_positional_embedding=model_config.local_head.use_positional_embedding,
      normalize_local_features=model_config.local_head.normalize_inputs)
