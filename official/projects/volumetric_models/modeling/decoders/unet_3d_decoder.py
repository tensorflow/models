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

"""Contains definitions of 3D UNet Model decoder part.

[1] Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf
Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse
Annotation. arXiv:1606.06650.
"""

from typing import Any, Dict, Mapping, Optional, Sequence

import tensorflow as tf

from official.modeling import hyperparams
from official.projects.volumetric_models.modeling import nn_blocks_3d
from official.projects.volumetric_models.modeling.decoders import factory

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class UNet3DDecoder(tf.keras.Model):
  """Class to build 3D UNet decoder."""

  def __init__(self,
               model_id: int,
               input_specs: Mapping[str, tf.TensorShape],
               pool_size: Sequence[int] = (2, 2, 2),
               kernel_size: Sequence[int] = (3, 3, 3),
               kernel_regularizer: tf.keras.regularizers.Regularizer = None,
               activation: str = 'relu',
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               use_sync_bn: bool = False,
               use_batch_normalization: bool = False,
               use_deconvolution: bool = False,  # pytype: disable=annotation-type-mismatch  # typed-keras
               **kwargs):
    """3D UNet decoder initialization function.

    Args:
      model_id: The depth of UNet3D backbone model. The greater the depth, the
        more max pooling layers will be added to the model. Lowering the depth
        may reduce the amount of memory required for training.
      input_specs: The input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      pool_size: The pooling size for the max pooling operations.
      kernel_size: The kernel size for 3D convolution.
      kernel_regularizer: A tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      activation: The name of the activation function.
      norm_momentum: The normalization momentum for the moving average.
      norm_epsilon: A float added to variance to avoid dividing by zero.
      use_sync_bn: If True, use synchronized batch normalization.
      use_batch_normalization: If set to True, use batch normalization after
        convolution and before activation. Default to False.
      use_deconvolution: If set to True, the model will use transpose
        convolution (deconvolution) instead of up-sampling. This increases the
        amount memory required during training. Default to False.
      **kwargs: Keyword arguments to be passed.
    """
    self._config_dict = {
        'model_id': model_id,
        'input_specs': input_specs,
        'pool_size': pool_size,
        'kernel_size': kernel_size,
        'kernel_regularizer': kernel_regularizer,
        'activation': activation,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'use_sync_bn': use_sync_bn,
        'use_batch_normalization': use_batch_normalization,
        'use_deconvolution': use_deconvolution
    }
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._use_batch_normalization = use_batch_normalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      channel_dim = -1
    else:
      channel_dim = 1

    # Build 3D UNet.
    inputs = self._build_input_pyramid(input_specs, model_id)  # pytype: disable=wrong-arg-types  # dynamic-method-lookup

    # Add levels with up-convolution or up-sampling.
    x = inputs[str(model_id)]
    for layer_depth in range(model_id - 1, 0, -1):
      # Apply deconvolution or upsampling.
      if use_deconvolution:
        x = layers.Conv3DTranspose(
            filters=x.get_shape().as_list()[channel_dim],
            kernel_size=pool_size,
            strides=(2, 2, 2))(
                x)
      else:
        x = layers.UpSampling3D(size=pool_size)(x)

      # Concatenate upsampled features with input features from one layer up.
      x = tf.concat([x, tf.cast(inputs[str(layer_depth)], dtype=x.dtype)],
                    axis=channel_dim)
      filter_num = inputs[str(layer_depth)].get_shape().as_list()[channel_dim]
      x = nn_blocks_3d.BasicBlock3DVolume(
          filters=[filter_num, filter_num],
          strides=(1, 1, 1),
          kernel_size=kernel_size,
          kernel_regularizer=kernel_regularizer,
          activation=activation,
          use_sync_bn=use_sync_bn,
          norm_momentum=norm_momentum,
          norm_epsilon=norm_epsilon,
          use_batch_normalization=use_batch_normalization)(
              x)

    feats = {'1': x}
    self._output_specs = {l: feats[l].get_shape() for l in feats}

    super(UNet3DDecoder, self).__init__(inputs=inputs, outputs=feats, **kwargs)

  def _build_input_pyramid(self, input_specs: Dict[str, tf.TensorShape],
                           depth: int) -> Dict[str, tf.Tensor]:
    """Builds input pyramid features."""
    assert isinstance(input_specs, dict)
    if len(input_specs.keys()) > depth:
      raise ValueError(
          'Backbone depth should be equal to 3D UNet decoder\'s depth.')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any], custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_decoder_builder('unet_3d_decoder')
def build_unet_3d_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds UNet3D decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf.keras.regularizers.Regularizer` instance. Default to
      None.

  Returns:
    A `tf.keras.Model` instance of the UNet3D decoder.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  assert decoder_type == 'unet_3d_decoder', (f'Inconsistent decoder type '
                                             f'{decoder_type}')
  norm_activation_config = model_config.norm_activation
  return UNet3DDecoder(
      model_id=decoder_cfg.model_id,
      input_specs=input_specs,
      pool_size=decoder_cfg.pool_size,
      kernel_regularizer=l2_regularizer,
      activation=norm_activation_config.activation,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_batch_normalization=decoder_cfg.use_batch_normalization,
      use_deconvolution=decoder_cfg.use_deconvolution)
