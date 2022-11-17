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

"""Head layers for Pointpillars."""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from official.projects.pointpillars.modeling import layers
from official.projects.pointpillars.utils import utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class SSDHead(tf.keras.layers.Layer):
  """A SSD head for PointPillars detection."""

  def __init__(
      self,
      num_classes: int,
      num_anchors_per_location: int,
      num_params_per_anchor: int = 4,
      attribute_heads: Optional[List[Dict[str, Any]]] = None,
      min_level: int = 1,
      max_level: int = 3,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize the SSD Head.

    Args:
      num_classes: An `int` number of classes to predict.
      num_anchors_per_location: An `int` number of anchors per location.
      num_params_per_anchor: An `int` number of parameters per anchor.
      attribute_heads: If not None, a list that contains a dict for each
        additional attribute head. Each dict consists of 3 key-value pairs:
        `name`, `type` ('regression' or 'classification'), and `size` (number
        of predicted values for each instance).
      min_level: An `int` of min level for output mutiscale features.
      max_level: An `int` of max level for output mutiscale features.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      endpoints: A `dict` of {level: Tensor} pairs for the model output.
      output_specs: A dict of {level: TensorShape} pairs for the model output.
    """
    super(SSDHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'num_anchors_per_location': num_anchors_per_location,
        'num_params_per_anchor': num_params_per_anchor,
        'attribute_heads': attribute_heads,
        'min_level': min_level,
        'max_level': max_level,
        'kernel_regularizer': kernel_regularizer,
    }

    utils.assert_channels_last()

  def build(self, input_specs: Mapping[str, tf.TensorShape]):
    self._decoder_output_level = int(min(input_specs.keys()))
    if self._config_dict['min_level'] < self._decoder_output_level:
      raise ValueError('The min_level should be >= decoder output '
                       'level, but {} < {}'.format(
                           self._config_dict['min_level'],
                           self._decoder_output_level))

    # Multi-level convs.
    # Set num_filters as the one of decoder's output level.
    num_filters = input_specs[str(self._decoder_output_level)].as_list()[-1]
    self._convs = {}
    for level in range(self._decoder_output_level + 1,
                       self._config_dict['max_level'] + 1):
      self._convs[str(level)] = layers.ConvBlock(
          filters=num_filters,
          kernel_size=3,
          strides=2,
          kernel_regularizer=self._config_dict['kernel_regularizer'])

    # Detection convs, share weights across multi levels.
    self._classifier = tf.keras.layers.Conv2D(
        filters=(self._config_dict['num_classes'] *
                 self._config_dict['num_anchors_per_location']),
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)))
    self._box_regressor = tf.keras.layers.Conv2D(
        filters=(self._config_dict['num_params_per_anchor'] *
                 self._config_dict['num_anchors_per_location']),
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_initializer=tf.zeros_initializer())
    if self._config_dict['attribute_heads']:
      self._att_predictors = {}
      for att_config in self._config_dict['attribute_heads']:
        att_name = att_config['name']
        att_type = att_config['type']
        att_size = att_config['size']
        if att_type != 'regression':
          raise ValueError('Unsupported head type: {}'.format(att_type))
        self._att_predictors[att_name] = tf.keras.layers.Conv2D(
            filters=(att_size * self._config_dict['num_anchors_per_location']),
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
            kernel_regularizer=self._config_dict['kernel_regularizer'],
            bias_initializer=tf.zeros_initializer())

    super(SSDHead, self).build(input_specs)

  def call(
      self, inputs: Mapping[str, tf.Tensor]
  ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[Any, Dict[str, Any]]]:
    # Build multi level features.
    feats = {}
    for level in range(self._decoder_output_level,
                       self._config_dict['max_level'] + 1):
      if level == self._decoder_output_level:
        x = inputs[str(level)]
      else:
        x = self._convs[str(level)](feats[level - 1])
      feats[level] = x

    # Get multi level detection.
    scores = {}
    boxes = {}
    if self._config_dict['attribute_heads']:
      attributes = {
          att_config['name']: {}
          for att_config in self._config_dict['attribute_heads']
      }
    else:
      attributes = {}

    for level in range(self._config_dict['min_level'],
                       self._config_dict['max_level'] + 1):
      # The branch to predict box classes.
      scores[str(level)] = self._classifier(feats[level])
      # The branch to predict boxes.
      boxes[str(level)] = self._box_regressor(feats[level])
      # The branches to predict box attributes.
      if self._config_dict['attribute_heads']:
        for att_config in self._config_dict['attribute_heads']:
          att_name = att_config['name']
          attributes[att_name][str(level)] = self._att_predictors[att_name](
              feats[level])

    return scores, boxes, attributes

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf.keras.layers.Layer:
    return cls(**config)
