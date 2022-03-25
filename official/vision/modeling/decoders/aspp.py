# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of Atrous Spatial Pyramid Pooling (ASPP) decoder."""
from typing import Any, List, Mapping, Optional, Union

# Import libraries

import tensorflow as tf

from official.modeling import hyperparams
from official.vision.modeling.decoders import factory
from official.vision.modeling.layers import deeplab
from official.vision.modeling.layers import nn_layers

TensorMapUnion = Union[tf.Tensor, Mapping[str, tf.Tensor]]


@tf.keras.utils.register_keras_serializable(package='Vision')
class ASPP(tf.keras.layers.Layer):
  """Creates an Atrous Spatial Pyramid Pooling (ASPP) layer."""

  def __init__(
      self,
      level: int,
      dilation_rates: List[int],
      num_filters: int = 256,
      pool_kernel_size: Optional[int] = None,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout_rate: float = 0.0,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = False,
      spp_layer_version: str = 'v1',
      output_tensor: bool = False,
      **kwargs):
    """Initializes an Atrous Spatial Pyramid Pooling (ASPP) layer.

    Args:
      level: An `int` level to apply ASPP.
      dilation_rates: A `list` of dilation rates.
      num_filters: An `int` number of output filters in ASPP.
      pool_kernel_size: A `list` of [height, width] of pooling kernel size or
        None. Pooling size is with respect to original image size, it will be
        scaled down by 2**level. If None, global average pooling is used.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      activation: A `str` activation to be used in ASPP.
      dropout_rate: A `float` rate for dropout regularization.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      interpolation: A `str` of interpolation method. It should be one of
        `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
        `gaussian`, or `mitchellcubic`.
      use_depthwise_convolution: If True depthwise separable convolutions will
        be added to the Atrous spatial pyramid pooling.
     spp_layer_version: A `str` of spatial pyramid pooling layer version.
     output_tensor: Whether to output a single tensor or a dictionary of tensor.
       Default is false.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._config_dict = {
        'level': level,
        'dilation_rates': dilation_rates,
        'num_filters': num_filters,
        'pool_kernel_size': pool_kernel_size,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'interpolation': interpolation,
        'use_depthwise_convolution': use_depthwise_convolution,
        'spp_layer_version': spp_layer_version,
        'output_tensor': output_tensor
    }
    self._aspp_layer = deeplab.SpatialPyramidPooling if self._config_dict[
        'spp_layer_version'] == 'v1' else nn_layers.SpatialPyramidPooling

  def build(self, input_shape):
    pool_kernel_size = None
    if self._config_dict['pool_kernel_size']:
      pool_kernel_size = [
          int(p_size // 2**self._config_dict['level'])
          for p_size in self._config_dict['pool_kernel_size']  # pytype: disable=attribute-error  # trace-all-classes
      ]

    self.aspp = self._aspp_layer(
        output_channels=self._config_dict['num_filters'],
        dilation_rates=self._config_dict['dilation_rates'],
        pool_kernel_size=pool_kernel_size,
        use_sync_bn=self._config_dict['use_sync_bn'],
        batchnorm_momentum=self._config_dict['norm_momentum'],
        batchnorm_epsilon=self._config_dict['norm_epsilon'],
        activation=self._config_dict['activation'],
        dropout=self._config_dict['dropout_rate'],
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        interpolation=self._config_dict['interpolation'],
        use_depthwise_convolution=self._config_dict['use_depthwise_convolution']
    )

  def call(self, inputs: TensorMapUnion) -> TensorMapUnion:
    """Calls the Atrous Spatial Pyramid Pooling (ASPP) layer on an input.

    The output of ASPP will be a dict of {`level`, `tf.Tensor`} even if only one
    level is present, if output_tensor is false. Hence, this will be compatible
    with the rest of the segmentation model interfaces.
    If output_tensor is true, a single tensot is output.

    Args:
      inputs: A `tf.Tensor` of shape [batch, height_l, width_l, filter_size] or
        a `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of shape [batch, height_l, width_l,
          filter_size].

    Returns:
      A `tf.Tensor` of shape [batch, height_l, width_l, filter_size] or a `dict`
        of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of output of ASPP module.
    """
    outputs = {}
    level = str(self._config_dict['level'])
    backbone_output = inputs[level] if isinstance(inputs, dict) else inputs
    outputs = self.aspp(backbone_output)
    return outputs if self._config_dict['output_tensor'] else {level: outputs}

  def get_config(self) -> Mapping[str, Any]:
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(self._config_dict.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@factory.register_decoder_builder('aspp')
def build_aspp_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds ASPP decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone. Note this is for consistent
        interface, and is not used by ASPP decoder.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf.keras.regularizers.Regularizer` instance. Default to
      None.

  Returns:
    A `tf.keras.Model` instance of the ASPP decoder.

  Raises:
    ValueError: If the model_config.decoder.type is not `aspp`.
  """
  del input_specs  # input_specs is not used by ASPP decoder.
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  if decoder_type != 'aspp':
    raise ValueError(f'Inconsistent decoder type {decoder_type}. '
                     'Need to be `aspp`.')

  norm_activation_config = model_config.norm_activation
  return ASPP(
      level=decoder_cfg.level,
      dilation_rates=decoder_cfg.dilation_rates,
      num_filters=decoder_cfg.num_filters,
      use_depthwise_convolution=decoder_cfg.use_depthwise_convolution,
      pool_kernel_size=decoder_cfg.pool_kernel_size,
      dropout_rate=decoder_cfg.dropout_rate,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      activation=norm_activation_config.activation,
      kernel_regularizer=l2_regularizer,
      spp_layer_version=decoder_cfg.spp_layer_version,
      output_tensor=decoder_cfg.output_tensor)
