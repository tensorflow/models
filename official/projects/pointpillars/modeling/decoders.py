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

"""Decoder models for Pointpillars."""

from typing import Any, Mapping, Optional

import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import layers
from official.projects.pointpillars.utils import utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class Decoder(tf_keras.Model):
  """The decoder to process feature maps learned by a backbone.

  The implementation is from the network architecture of PointPillars
  (https://arxiv.org/pdf/1812.05784.pdf). It upsamples the feature image
  to the same size and combine them to be the output.
  """

  def __init__(
      self,
      input_specs: Mapping[str, tf.TensorShape],
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize the Decoder.

    Args:
      input_specs: A dict of {level: tf.TensorShape} of the input tensor.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      endpoints: A `dict` of {level: Tensor} pairs for the model output.
      output_specs: A dict of {level: TensorShape} pairs for the model output.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'kernel_regularizer': kernel_regularizer,
    }

    utils.assert_channels_last()

    # Only allow to process levels learned by a backbone.
    min_level = int(min(input_specs.keys()))
    max_level = int(max(input_specs.keys()))

    # Build inputs
    inputs = {}
    # Set min_level as the output level.
    output_level = min_level
    for level, shape in input_specs.items():
      # Set num_filters as 2c if the channels of backbone output level is c.
      if int(level) == output_level:
        num_filters = 2 * shape[-1]
      inputs[level] = tf_keras.Input(shape=shape[1:])

    # Build lateral features
    lateral_feats = {}
    for level in range(min_level, max_level + 1):
      lateral_feats[level] = inputs[str(level)]

    # Build scale-up path
    feats = []
    for level in range(min_level, max_level + 1):
      x = layers.ConvBlock(
          filters=num_filters,
          kernel_size=3,
          strides=int(2 ** (level - output_level)),
          use_transpose_conv=True,
          kernel_regularizer=kernel_regularizer)(
              lateral_feats[level])
      feats.append(x)

    # Fuse all levels feature into the output level.
    endpoints = {}
    endpoints[str(output_level)] = tf_keras.layers.Concatenate(axis=-1)(feats)

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    super(Decoder, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf_keras.Model:
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    return self._output_specs
