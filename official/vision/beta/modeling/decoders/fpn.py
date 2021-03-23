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

"""Contains the definitions of Feature Pyramid Networks (FPN)."""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class FPN(tf.keras.Model):
  """Creates a Feature Pyramid Network (FPN).

  This implemets the paper:
  Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and
  Serge Belongie.
  Feature Pyramid Networks for Object Detection.
  (https://arxiv.org/pdf/1612.03144)
  """

  def __init__(self,
               input_specs,
               min_level=3,
               max_level=7,
               num_filters=256,
               use_separable_conv=False,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Initializes a Feature Pyramid Network (FPN).

    Args:
      input_specs: A `dict` of input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      min_level: An `int` of minimum level in FPN output feature maps.
      max_level: An `int` of maximum level in FPN output feature maps.
      num_filters: An `int` number of filters in FPN layers.
      use_separable_conv: A `bool`.  If True use separable convolution for
        convolution in FPN layers.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'min_level': min_level,
        'max_level': max_level,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if use_separable_conv:
      conv2d = tf.keras.layers.SeparableConv2D
    else:
      conv2d = tf.keras.layers.Conv2D
    if use_sync_bn:
      norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      norm = tf.keras.layers.BatchNormalization
    activation_fn = tf.keras.layers.Activation(
        tf_utils.get_activation(activation))

    # Build input feature pyramid.
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Get input feature pyramid from backbone.
    inputs = self._build_input_pyramid(input_specs, min_level)
    backbone_max_level = min(int(max(inputs.keys())), max_level)

    # Build lateral connections.
    feats_lateral = {}
    for level in range(min_level, backbone_max_level + 1):
      feats_lateral[str(level)] = conv2d(
          filters=num_filters,
          kernel_size=1,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              inputs[str(level)])

    # Build top-down path.
    feats = {str(backbone_max_level): feats_lateral[str(backbone_max_level)]}
    for level in range(backbone_max_level - 1, min_level - 1, -1):
      feats[str(level)] = spatial_transform_ops.nearest_upsampling(
          feats[str(level + 1)], 2) + feats_lateral[str(level)]

    # TODO(xianzhi): consider to remove bias in conv2d.
    # Build post-hoc 3x3 convolution kernel.
    for level in range(min_level, backbone_max_level + 1):
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=1,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              feats[str(level)])

    # TODO(xianzhi): consider to remove bias in conv2d.
    # Build coarser FPN levels introduced for RetinaNet.
    for level in range(backbone_max_level + 1, max_level + 1):
      feats_in = feats[str(level - 1)]
      if level > backbone_max_level + 1:
        feats_in = activation_fn(feats_in)
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=2,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              feats_in)

    # Apply batch norm layers.
    for level in range(min_level, max_level + 1):
      feats[str(level)] = norm(
          axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
              feats[str(level)])

    self._output_specs = {
        str(level): feats[str(level)].get_shape()
        for level in range(min_level, max_level + 1)
    }

    super(FPN, self).__init__(inputs=inputs, outputs=feats, **kwargs)

  def _build_input_pyramid(self, input_specs, min_level):
    assert isinstance(input_specs, dict)
    if min(input_specs.keys()) > str(min_level):
      raise ValueError(
          'Backbone min level should be less or equal to FPN min level')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
