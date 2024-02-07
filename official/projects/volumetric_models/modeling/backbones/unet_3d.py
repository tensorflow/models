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

"""Contains definitions of 3D UNet Model encoder part.

[1] Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf
Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse
Annotation. arXiv:1606.06650.
"""

from typing import Any, Mapping, Sequence

# Import libraries
import tensorflow as tf, tf_keras
from official.modeling import hyperparams
from official.projects.volumetric_models.modeling import nn_blocks_3d
from official.vision.modeling.backbones import factory

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class UNet3D(tf_keras.Model):
  """Class to build 3D UNet backbone."""

  def __init__(
      self,
      model_id: int,
      input_specs: layers = layers.InputSpec(shape=[None, None, None, None, 3]),
      pool_size: Sequence[int] = (2, 2, 2),
      kernel_size: Sequence[int] = (3, 3, 3),
      base_filters: int = 32,
      kernel_regularizer: tf_keras.regularizers.Regularizer = None,
      activation: str = 'relu',
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      use_sync_bn: bool = False,
      use_batch_normalization: bool = False,  # type: ignore  # typed-keras
      **kwargs):
    """3D UNet backbone initialization function.

    Args:
      model_id: The depth of UNet3D backbone model. The greater the depth, the
        more max pooling layers will be added to the model. Lowering the depth
        may reduce the amount of memory required for training.
      input_specs: The specs of the input tensor. It specifies a 5D input of
        [batch, height, width, volume, channel] for `channel_last` data format
        or [batch, channel, height, width, volume] for `channel_first` data
        format.
      pool_size: The pooling size for the max pooling operations.
      kernel_size: The kernel size for 3D convolution.
      base_filters: The number of filters that the first layer in the
        convolution network will have. Following layers will contain a multiple
        of this number. Lowering this number will likely reduce the amount of
        memory required to train the model.
      kernel_regularizer: A tf_keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      activation: The name of the activation function.
      norm_momentum: The normalization momentum for the moving average.
      norm_epsilon: A float added to variance to avoid dividing by zero.
      use_sync_bn: If True, use synchronized batch normalization.
      use_batch_normalization: If set to True, use batch normalization after
        convolution and before activation. Default to False.
      **kwargs: Keyword arguments to be passed.
    """

    self._model_id = model_id
    self._input_specs = input_specs
    self._pool_size = pool_size
    self._kernel_size = kernel_size
    self._activation = activation
    self._base_filters = base_filters
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._use_sync_bn = use_sync_bn
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._kernel_regularizer = kernel_regularizer
    self._use_batch_normalization = use_batch_normalization

    # Build 3D UNet.
    inputs = tf_keras.Input(
        shape=input_specs.shape[1:], dtype=input_specs.dtype)
    x = inputs
    endpoints = {}

    # Add levels with max pooling to downsample input.
    for layer_depth in range(model_id):
      # Two convoluions are applied sequentially without downsampling.
      filter_num = base_filters * (2**layer_depth)
      x2 = nn_blocks_3d.BasicBlock3DVolume(
          filters=[filter_num, filter_num * 2],
          strides=(1, 1, 1),
          kernel_size=self._kernel_size,
          kernel_regularizer=self._kernel_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          use_batch_normalization=self._use_batch_normalization)(
              x)
      if layer_depth < model_id - 1:
        x = layers.MaxPool3D(
            pool_size=pool_size,
            strides=(2, 2, 2),
            padding='valid',
            data_format=tf_keras.backend.image_data_format())(
                x2)
      else:
        x = x2
      endpoints[str(layer_depth + 1)] = x2

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(UNet3D, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def get_config(self) -> Mapping[str, Any]:
    return {
        'model_id': self._model_id,
        'pool_size': self._pool_size,
        'kernel_size': self._kernel_size,
        'activation': self._activation,
        'base_filters': self._base_filters,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'use_sync_bn': self._use_sync_bn,
        'kernel_regularizer': self._kernel_regularizer,
        'use_batch_normalization': self._use_batch_normalization
    }

  @classmethod
  def from_config(cls, config: Mapping[str, Any], custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """Returns a dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('unet_3d')
def build_unet3d(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds 3D UNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'unet_3d', (f'Inconsistent backbone type '
                                      f'{backbone_type}')

  return UNet3D(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      pool_size=backbone_cfg.pool_size,
      base_filters=backbone_cfg.base_filters,
      kernel_regularizer=l2_regularizer,
      activation=norm_activation_config.activation,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_batch_normalization=backbone_cfg.use_batch_normalization)
