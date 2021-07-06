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

from typing import List

import tensorflow as tf

from official.vision.beta.modeling.backbones import factory
from official.vision.beta.projects.centernet.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package='centernet')
class Hourglass(tf.keras.Model):
  """
  CenterNet Hourglass backbone
  """
  
  def __init__(
      self,
      input_channel_dims: int,
      channel_dims_per_stage: List[int],
      blocks_per_stage: List[int],
      num_hourglasses: int,
      initial_downsample: bool = True,
      norm_momentum=0.1,
      norm_episilon=1e-5,
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      **kwargs):
    """
    Args:
        input_channel_dims: integer, number of filters used to downsample the 
          input image
        channel_dims_per_stage: list, containing of number of filters for the
          residual blocks in the hourglass blocks
        blocks_per_stage: list of residual block repetitions to use in the
          hourglass blocks
        num_hourglasses: integer, number of hourglass blocks in backbone
        initial_downsample: bool, whether or not to downsample the input
        norm_momentum: float, momentum for the batch normalization layers
        norm_episilon: float, epsilon for the batch normalization layers
    """
    # yapf: disable
    input = tf.keras.layers.Input(shape=input_specs.shape[1:], name='input')
    
    inp_filters = channel_dims_per_stage[0]
    
    # Downsample the input
    if initial_downsample:
      prelayer_kernel_size = 7
      prelayer_strides = 2
    else:
      prelayer_kernel_size = 3
      prelayer_strides = 1
    
    x_downsampled = nn_blocks.CenterNetConvBN(
        filters=input_channel_dims,
        kernel_size=prelayer_kernel_size,
        strides=prelayer_strides,
        padding='valid',
        activation='relu',
        use_sync_bn=True,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_episilon)(input)
    
    x_downsampled = nn_blocks.CenterNetResidualBlock(
        filters=inp_filters,
        use_projection=True,
        strides=prelayer_strides,
        use_sync_bn=True,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_episilon)(x_downsampled)
    
    all_heatmaps = []
    for i in range(num_hourglasses):
      # Create an hourglass stack
      x_hg = nn_blocks.HourglassBlock(
          channel_dims_per_stage=channel_dims_per_stage,
          blocks_per_stage=blocks_per_stage,
      )(x_downsampled)
      
      x_hg = nn_blocks.CenterNetConvBN(
          filters=inp_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='valid',
          activation='relu',
          use_sync_bn=True,
          norm_momentum=norm_momentum,
          norm_epsilon=norm_episilon
      )(x_hg)
      
      all_heatmaps.append(x_hg)
      
      # Intermediate conv and residual layers between hourglasses
      if i < num_hourglasses - 1:
        inter_hg_conv1 = nn_blocks.CenterNetConvBN(
            filters=inp_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='identity',
            use_sync_bn=True,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_episilon
        )(x_downsampled)
        
        inter_hg_conv2 = nn_blocks.CenterNetConvBN(
            filters=inp_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='identity',
            use_sync_bn=True,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_episilon
        )(x_hg)
        
        x_downsampled = tf.keras.layers.Add()([inter_hg_conv1, inter_hg_conv2])
        x_downsampled = tf.keras.layers.ReLU()(x_downsampled)
        
        x_downsampled = nn_blocks.CenterNetResidualBlock(
            filters=inp_filters, use_projection=False, strides=1,
            use_sync_bn=True, norm_momentum=norm_momentum,
            norm_epsilon=norm_episilon
        )(x_downsampled)
    # yapf: enable
    
    super().__init__(inputs=input, outputs=all_heatmaps, **kwargs)
    
    self._input_channel_dims = input_channel_dims
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._num_hourglasses = num_hourglasses
    self._initial_downsample = initial_downsample
    self._norm_momentum = norm_momentum
    self._norm_episilon = norm_episilon
    self._output_specs = [hm.get_shape() for hm in all_heatmaps]
  
  def get_config(self):
    layer_config = {
        'input_channel_dims': self._input_channel_dims,
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'num_hourglasses': self._num_hourglasses,
        'initial_downsample': self._initial_downsample,
        'norm_momentum': self._norm_momentum,
        'norm_episilon': self._norm_episilon
    }
    layer_config.update(super().get_config())
    return layer_config
  
  @property
  def output_specs(self):
    return self._output_specs


@factory.register_backbone_builder('hourglass')
def build_hourglass(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds Hourglass backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  assert backbone_type == 'hourglass', (f'Inconsistent backbone type '
                                        f'{backbone_type}')
  
  return Hourglass(
      input_channel_dims=backbone_cfg.input_channel_dims,
      channel_dims_per_stage=backbone_cfg.channel_dims_per_stage,
      blocks_per_stage=backbone_cfg.blocks_per_stage,
      num_hourglasses=backbone_cfg.num_hourglasses,
      initial_downsample=backbone_cfg.initial_downsample,
      norm_momentum=backbone_cfg.norm_momentum,
      norm_episilon=backbone_cfg.norm_episilon,
      input_specs=input_specs)
