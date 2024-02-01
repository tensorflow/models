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

"""Backbone models for Pointpillars."""

from typing import Any, Mapping, Optional
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import layers
from official.projects.pointpillars.utils import utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class Backbone(tf_keras.Model):
  """The backbone to extract features from BEV pseudo image.

  The implementation is from the network architecture of PointPillars
  (https://arxiv.org/pdf/1812.05784.pdf). It downsamples the input image
  through convolutions and output features with multiple levels.
  """

  def __init__(
      self,
      input_specs: tf.TensorShape,
      min_level: int = 1,
      max_level: int = 3,
      num_convs: int = 4,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize the backbone.

    The output of the backbone is a multi-level features.
    1 <= min_level <= max_level,
    level_feature_size = input_image_size / 2 ^ level,
    e.g. input size (32, 32), feature size should be:
    (32, 32) at level 0, (16, 16) at level 1, (8, 8) at level 2, ...
    Args:
      input_specs: A `tf.TensorShape` of the input tensor.
      min_level: An `int` of min level for output multiscale features.
      max_level: An `int` of max level for output multiscale features.
      num_convs: An `int` number of convolution layers in a downsample group.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      endpoints: A `dict` of {level: Tensor} pairs for the model output.
      output_specs: A dict of {level: TensorShape} pairs for the model output.
    """
    utils.assert_channels_last()

    self._config_dict = {
        'input_specs': input_specs,
        'min_level': min_level,
        'max_level': max_level,
        'num_convs': num_convs,
        'kernel_regularizer': kernel_regularizer,
    }
    # Onlly allow to output from level 1.
    if min_level < 1:
      raise ValueError(
          'The min_level must be >= 1, but {} found.'.format(min_level))

    input_channels = input_specs[-1]
    inputs = tf_keras.Input(shape=input_specs[1:])

    # build the net
    x = inputs
    net = {}
    scale = 1
    for level in range(1, max_level + 1):
      x = self._block_group(
          inputs=x,
          filters=input_channels * scale)
      scale *= 2
      net[level] = x

    # build endpoints
    endpoints = {}
    for level in range(min_level, max_level + 1):
      endpoints[str(level)] = net[level]

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    super(Backbone, self).__init__(inputs=inputs, outputs=endpoints)

  def _block_group(self,
                   inputs: tf.Tensor,
                   filters: int) -> tf.Tensor:
    """A group of convolution layers to downsample inputs.

    Args:
      inputs: A tensor to be downsampled.
      filters: An `int` number of filters of convolution.

    Returns:
      x: A tensor of downsampled feature.
    """
    x = layers.ConvBlock(
        filters=filters,
        kernel_size=3,
        strides=2,
        kernel_regularizer=self._config_dict['kernel_regularizer'])(inputs)
    for _ in range(1, self._config_dict['num_convs']):
      x = layers.ConvBlock(
          filters=filters,
          kernel_size=3,
          strides=1,
          kernel_regularizer=self._config_dict['kernel_regularizer'])(x)
    return x

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf_keras.Model:
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    return self._output_specs
