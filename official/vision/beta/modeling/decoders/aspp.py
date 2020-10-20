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
"""ASPP decoder."""

# Import libraries
import tensorflow as tf

from official.vision import keras_cv


@tf.keras.utils.register_keras_serializable(package='Vision')
class ASPP(tf.keras.layers.Layer):
  """ASPP."""

  def __init__(self,
               level,
               dilation_rates,
               num_filters=256,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='relu',
               dropout_rate=0.0,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               interpolation='bilinear',
               **kwargs):
    """ASPP initialization function.

    Args:
      level: `int` level to apply ASPP.
      dilation_rates: `list` of dilation rates.
      num_filters: `int` number of output filters in ASPP.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      activation: `str` activation to be used in ASPP.
      dropout_rate: `float` rate for dropout regularization.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      interpolation: interpolation method, one of bilinear, nearest, bicubic,
        area, lanczos3, lanczos5, gaussian, or mitchellcubic.
      **kwargs: keyword arguments to be passed.
    """
    super(ASPP, self).__init__(**kwargs)
    self._config_dict = {
        'level': level,
        'dilation_rates': dilation_rates,
        'num_filters': num_filters,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'interpolation': interpolation,
    }

  def build(self, input_shape):
    self.aspp = keras_cv.layers.SpatialPyramidPooling(
        output_channels=self._config_dict['num_filters'],
        dilation_rates=self._config_dict['dilation_rates'],
        use_sync_bn=self._config_dict['use_sync_bn'],
        batchnorm_momentum=self._config_dict['norm_momentum'],
        batchnorm_epsilon=self._config_dict['norm_epsilon'],
        activation=self._config_dict['activation'],
        dropout=self._config_dict['dropout_rate'],
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        interpolation=self._config_dict['interpolation'])

  def call(self, inputs):
    """ASPP call method.

    The output of ASPP will be a dict of level, Tensor even if only one
    level is present. Hence, this will be compatible with the rest of the
    segmentation model interfaces..

    Args:
      inputs: A dict of tensors
        - key: `str`, the level of the multilevel feature maps.
        - values: `Tensor`, [batch, height_l, width_l, filter_size].
    Returns:
      A dict of tensors
        - key: `str`, the level of the multilevel feature maps.
        - values: `Tensor`, output of ASPP module.
    """
    outputs = {}
    level = str(self._config_dict['level'])
    outputs[level] = self.aspp(inputs[level])
    return outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
