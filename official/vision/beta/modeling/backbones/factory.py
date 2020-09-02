# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""factory method."""
# Import libraries
import tensorflow as tf

from official.vision.beta.modeling import backbones
from official.vision.beta.modeling.backbones import spinenet


def build_backbone(input_specs: tf.keras.layers.InputSpec,
                   model_config,
                   l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds backbone from a config.

  Args:
    input_specs: tf.keras.layers.InputSpec.
    model_config: a OneOfConfig. Model config.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    tf.keras.Model instance of the backbone.
  """
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation

  if backbone_type == 'resnet':
    backbone = backbones.ResNet(
        model_id=backbone_cfg.model_id,
        input_specs=input_specs,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  elif backbone_type == 'efficientnet':
    backbone = backbones.EfficientNet(
        model_id=backbone_cfg.model_id,
        input_specs=input_specs,
        stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate,
        se_ratio=backbone_cfg.se_ratio,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  elif backbone_type == 'spinenet':
    model_id = backbone_cfg.model_id
    if model_id not in spinenet.SCALING_MAP:
      raise ValueError(
          'SpineNet-{} is not a valid architecture.'.format(model_id))
    scaling_params = spinenet.SCALING_MAP[model_id]

    backbone = backbones.SpineNet(
        input_specs=input_specs,
        min_level=model_config.min_level,
        max_level=model_config.max_level,
        endpoints_num_filters=scaling_params['endpoints_num_filters'],
        resample_alpha=scaling_params['resample_alpha'],
        block_repeats=scaling_params['block_repeats'],
        filter_size_scale=scaling_params['filter_size_scale'],
        kernel_regularizer=l2_regularizer,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon)
  elif backbone_type == 'revnet':
    backbone = backbones.RevNet(
        model_id=backbone_cfg.model_id,
        input_specs=input_specs,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  else:
    raise ValueError('Backbone {!r} not implement'.format(backbone_type))

  return backbone


def build_backbone_3d(input_specs: tf.keras.layers.InputSpec,
                      model_config,
                      l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds 3d backbone from a config.

  Args:
    input_specs: tf.keras.layers.InputSpec.
    model_config: a OneOfConfig. Model config.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    tf.keras.Model instance of the backbone.
  """
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation

  # Flatten configs before passing to the backbone.
  temporal_strides = []
  temporal_kernel_sizes = []
  use_self_gating = []
  for block_spec in backbone_cfg.block_specs:
    temporal_strides.append(block_spec.temporal_strides)
    temporal_kernel_sizes.append(block_spec.temporal_kernel_sizes)
    use_self_gating.append(block_spec.use_self_gating)

  if backbone_type == 'resnet_3d':
    backbone = backbones.ResNet3D(
        model_id=backbone_cfg.model_id,
        temporal_strides=temporal_strides,
        temporal_kernel_sizes=temporal_kernel_sizes,
        use_self_gating=use_self_gating,
        input_specs=input_specs,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  else:
    raise ValueError('Backbone {!r} not implement'.format(backbone_type))

  return backbone
