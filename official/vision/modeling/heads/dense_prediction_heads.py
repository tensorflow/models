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

"""Contains definitions of dense prediction heads."""

from typing import Any, Dict, List, Mapping, Optional, Union

# Import libraries

import numpy as np
import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class RetinaNetHead(tf.keras.layers.Layer):
  """Creates a RetinaNet head."""

  def __init__(
      self,
      min_level: int,
      max_level: int,
      num_classes: int,
      num_anchors_per_location: int,
      num_convs: int = 4,
      num_filters: int = 256,
      attribute_heads: Optional[List[Dict[str, Any]]] = None,
      share_classification_heads: bool = False,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      num_params_per_anchor: int = 4,
      share_level_convs: bool = True,
      **kwargs,
  ):
    """Initializes a RetinaNet head.

    Args:
      min_level: An `int` number of minimum feature level.
      max_level: An `int` number of maximum feature level.
      num_classes: An `int` number of classes to predict.
      num_anchors_per_location: An `int` number of anchors per pixel location.
      num_convs: An `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: An `int` number that represents the number of filters of the
        intermediate conv layers.
      attribute_heads: If not None, a list that contains a dict for each
        additional attribute head. Each dict consists of 4 key-value pairs:
        `name`, `type` ('regression' or 'classification'), `size` (number of
        predicted values for each instance), and `prediction_tower_name`
        (optional, specifies shared prediction towers.)
      share_classification_heads: A `bool` that indicates whether sharing
        weights among the main and attribute classification heads.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      num_params_per_anchor: Number of parameters required to specify an anchor
        box. For example, `num_params_per_anchor` would be 4 for axis-aligned
        anchor boxes specified by their y-centers, x-centers, heights, and
        widths.
      share_level_convs: An optional bool to enable sharing convs
        across levels for classnet, boxnet, classifier and box regressor.
        If True, convs will be shared across all levels.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._config_dict = {
        'min_level': min_level,
        'max_level': max_level,
        'num_classes': num_classes,
        'num_anchors_per_location': num_anchors_per_location,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'attribute_heads': attribute_heads,
        'share_classification_heads': share_classification_heads,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'num_params_per_anchor': num_params_per_anchor,
        'share_level_convs': share_level_convs,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

    self._conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      self._conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.01),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })

    self._bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._classifier_kwargs = {
        'filters': (
            self._config_dict['num_classes']
            * self._config_dict['num_anchors_per_location']
        ),
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      self._classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })

    self._box_regressor_kwargs = {
        'filters': (
            self._config_dict['num_params_per_anchor']
            * self._config_dict['num_anchors_per_location']
        ),
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      self._box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })

    if self._config_dict['attribute_heads']:
      self._init_attribute_kwargs()

  def _conv_kwargs_new_kernel_init(self, conv_kwargs):
    if 'kernel_initializer' in conv_kwargs:
      conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
          conv_kwargs['kernel_initializer']
      )
    return conv_kwargs

  def _init_attribute_kwargs(self):
    self._attribute_kwargs = []
    for att_config in self._config_dict['attribute_heads']:
      att_type = att_config['type']
      att_size = att_config['size']
      att_prediction_tower_name = att_config['prediction_tower_name']

      att_predictor_kwargs = {
          'filters': att_size * self._config_dict['num_anchors_per_location'],
          'kernel_size': 3,
          'padding': 'same',
          'bias_initializer': tf.zeros_initializer(),
          'bias_regularizer': self._config_dict['bias_regularizer'],
      }
      if att_type == 'regression':
        att_predictor_kwargs.update(
            {'bias_initializer': tf.zeros_initializer()}
        )
      elif att_type == 'classification':
        att_predictor_kwargs.update(
            {
                'bias_initializer': tf.constant_initializer(
                    -np.log((1 - 0.01) / 0.01)
                )
            }
        )
      else:
        raise ValueError(
            'Attribute head type {} not supported.'.format(att_type)
        )

      if (
          att_prediction_tower_name
          and self._config_dict['share_classification_heads']
      ):
        raise ValueError(
            'share_classification_heads cannot be set as True when'
            ' att_prediction_tower_name is specified.'
        )

      if not self._config_dict['use_separable_conv']:
        att_predictor_kwargs.update({
            'kernel_initializer': tf.keras.initializers.RandomNormal(
                stddev=1e-5
            ),
            'kernel_regularizer': self._config_dict['kernel_regularizer'],
        })
      self._attribute_kwargs.append(att_predictor_kwargs)

  def _apply_prediction_tower(self, features, convs, norms) -> tf.Tensor:
    x = features
    for conv, norm in zip(convs, norms):
      x = conv(x)
      x = norm(x)
      x = self._activation(x)
    return x

  def _apply_attribute_net(
      self, attributes, level, level_idx, this_level_features, classnet_x
  ):
    prediction_tower_output = {}
    for att_config in self._config_dict['attribute_heads']:
      att_name = att_config['name']
      att_type = att_config['type']
      if (
          self._config_dict['share_classification_heads']
          and att_type == 'classification'
      ):
        attributes[att_name][str(level)] = self._att_predictors[att_name](
            classnet_x
        )
      else:

        def _apply_attribute_prediction_tower(
            atttribute_name, features, feature_level
        ):
          return self._apply_prediction_tower(
              features,
              self._att_convs[atttribute_name],
              self._att_norms[atttribute_name][feature_level],
          )

        prediction_tower_name = att_config['prediction_tower_name']
        if not prediction_tower_name:
          attributes[att_name][str(level)] = self._att_predictors[att_name](
              _apply_attribute_prediction_tower(
                  att_name, this_level_features, level_idx
              )
          )
        else:
          if prediction_tower_name not in prediction_tower_output:
            prediction_tower_output[prediction_tower_name] = (
                _apply_attribute_prediction_tower(
                    att_name, this_level_features, level_idx
                )
            )
          attributes[att_name][str(level)] = self._att_predictors[att_name](
              prediction_tower_output[prediction_tower_name]
          )

  def _build_prediction_tower(
      self, net_name, predictor_name, conv_op, bn_op, predictor_kwargs
  ):
    """Builds the prediction tower. Convs across levels can be shared or not."""
    convs = []
    norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1
    ):
      if not self._config_dict['share_level_convs']:
        this_level_convs = []
      this_level_norms = []
      for i in range(self._config_dict['num_convs']):
        conv_kwargs = self._conv_kwargs_new_kernel_init(self._conv_kwargs)
        if not self._config_dict['share_level_convs']:
          # Do not share convs.
          this_level_convs.append(
              conv_op(name=f'{net_name}-conv_{level}_{i}', **conv_kwargs)
          )
        elif level == self._config_dict['min_level']:
          convs.append(conv_op(name=f'{net_name}-conv_{i}', **conv_kwargs))
        this_level_norms.append(
            bn_op(name=f'{net_name}-conv-norm_{level}_{i}', **self._bn_kwargs)
        )
      norms.append(this_level_norms)
      if not self._config_dict['share_level_convs']:
        convs.append(this_level_convs)

    # Create predictors after additional convs.
    if self._config_dict['share_level_convs']:
      predictors = conv_op(name=predictor_name, **predictor_kwargs)
    else:
      predictors = []
      for level in range(
          self._config_dict['min_level'], self._config_dict['max_level'] + 1
      ):
        predictors.append(
            conv_op(name=f'{predictor_name}-{level}', **predictor_kwargs)
        )

    return convs, norms, predictors

  def _build_attribute_net(self, conv_op, bn_op):
    self._att_predictors = {}
    self._att_convs = {}
    self._att_norms = {}

    for att_config, att_predictor_kwargs in zip(
        self._config_dict['attribute_heads'], self._attribute_kwargs
    ):
      att_name = att_config['name']
      att_num_convs = (
          att_config.get('num_convs') or self._config_dict['num_convs']
      )
      att_num_filters = (
          att_config.get('num_filters') or self._config_dict['num_filters']
      )
      if att_num_convs < 0:
        raise ValueError(f'Invalid `num_convs` {att_num_convs} for {att_name}.')
      if att_num_filters < 0:
        raise ValueError(
            f'Invalid `num_filters` {att_num_filters} for {att_name}.'
        )
      att_conv_kwargs = self._conv_kwargs.copy()
      att_conv_kwargs['filters'] = att_num_filters
      att_convs_i = []
      att_norms_i = []

      # Build conv and norm layers.
      for level in range(
          self._config_dict['min_level'], self._config_dict['max_level'] + 1
      ):
        this_level_att_norms = []
        for i in range(att_num_convs):
          if level == self._config_dict['min_level']:
            att_conv_name = '{}-conv_{}'.format(att_name, i)
            att_convs_i.append(conv_op(name=att_conv_name, **att_conv_kwargs))
          att_norm_name = '{}-conv-norm_{}_{}'.format(att_name, level, i)
          this_level_att_norms.append(
              bn_op(name=att_norm_name, **self._bn_kwargs)
          )
        att_norms_i.append(this_level_att_norms)
      self._att_convs[att_name] = att_convs_i
      self._att_norms[att_name] = att_norms_i

      # Build the final prediction layer.
      self._att_predictors[att_name] = conv_op(
          name='{}_attributes'.format(att_name), **att_predictor_kwargs
      )

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_op = (
        tf.keras.layers.SeparableConv2D
        if self._config_dict['use_separable_conv']
        else tf.keras.layers.Conv2D
    )
    bn_op = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._config_dict['use_sync_bn']
        else tf.keras.layers.BatchNormalization
    )

    # Class net.
    self._cls_convs, self._cls_norms, self._classifier = (
        self._build_prediction_tower(
            'classnet', 'scores', conv_op, bn_op, self._classifier_kwargs
        )
    )

    # Box net.
    self._box_convs, self._box_norms, self._box_regressor = (
        self._build_prediction_tower(
            'boxnet', 'boxes', conv_op, bn_op, self._box_regressor_kwargs
        )
    )

    # Attribute learning nets.
    if self._config_dict['attribute_heads']:
      self._build_attribute_net(conv_op, bn_op)

    super().build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RetinaNet head.

    Args:
      features: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor`, the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].

    Returns:
      scores: A `dict` of `tf.Tensor` which includes scores of the predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: A `dict` of `tf.Tensor` which includes coordinates of the
        predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l,
             num_params_per_anchor * num_anchors_per_location].
      attributes: a dict of (attribute_name, attribute_prediction). Each
        `attribute_prediction` is a dict of:
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box scores predicted from a particular feature
            level, whose shape is
            [batch, height_l, width_l,
            attribute_size * num_anchors_per_location].
        Can be an empty dictionary if no attribute learning is required.
    """
    scores = {}
    boxes = {}
    if self._config_dict['attribute_heads']:
      attributes = {
          att_config['name']: {}
          for att_config in self._config_dict['attribute_heads']
      }
    else:
      attributes = {}

    for i, level in enumerate(
        range(self._config_dict['min_level'],
              self._config_dict['max_level'] + 1)):
      this_level_features = features[str(level)]

      if self._config_dict['share_level_convs']:
        cls_convs = self._cls_convs
        box_convs = self._box_convs
        classifier = self._classifier
        box_regressor = self._box_regressor
      else:
        cls_convs = self._cls_convs[i]
        box_convs = self._box_convs[i]
        classifier = self._classifier[i]
        box_regressor = self._box_regressor[i]

      # Apply class net.
      x = self._apply_prediction_tower(
          this_level_features, cls_convs, self._cls_norms[i]
      )
      scores[str(level)] = classifier(x)
      classnet_x = x

      # Apply box net.
      x = self._apply_prediction_tower(
          this_level_features, box_convs, self._box_norms[i]
      )
      boxes[str(level)] = box_regressor(x)

      # Apply attribute nets.
      if self._config_dict['attribute_heads']:
        self._apply_attribute_net(
            attributes, level, i, this_level_features, classnet_x
        )

    return scores, boxes, attributes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class RPNHead(tf.keras.layers.Layer):
  """Creates a Region Proposal Network (RPN) head."""

  def __init__(
      self,
      min_level: int,
      max_level: int,
      num_anchors_per_location: int,
      num_convs: int = 1,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a Region Proposal Network head.

    Args:
      min_level: An `int` number of minimum feature level.
      max_level: An `int` number of maximum feature level.
      num_anchors_per_location: An `int` number of number of anchors per pixel
        location.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the prediction.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(RPNHead, self).__init__(**kwargs)
    self._config_dict = {
        'min_level': min_level,
        'max_level': max_level,
        'num_anchors_per_location': num_anchors_per_location,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=0.01),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1):
      this_level_norms = []
      for i in range(self._config_dict['num_convs']):
        if level == self._config_dict['min_level']:
          conv_name = 'rpn-conv_{}'.format(i)
          if 'kernel_initializer' in conv_kwargs:
            conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                conv_kwargs['kernel_initializer'])
          self._convs.append(conv_op(name=conv_name, **conv_kwargs))
        norm_name = 'rpn-conv-norm_{}_{}'.format(level, i)
        this_level_norms.append(bn_op(name=norm_name, **bn_kwargs))
      self._norms.append(this_level_norms)

    classifier_kwargs = {
        'filters': self._config_dict['num_anchors_per_location'],
        'kernel_size': 1,
        'padding': 'valid',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._classifier = conv_op(name='rpn-scores', **classifier_kwargs)

    box_regressor_kwargs = {
        'filters': 4 * self._config_dict['num_anchors_per_location'],
        'kernel_size': 1,
        'padding': 'valid',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._box_regressor = conv_op(name='rpn-boxes', **box_regressor_kwargs)

    super(RPNHead, self).build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RPN head.

    Args:
      features: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor`, the feature map tensors, whose shape is [batch,
          height_l, width_l, channels].

    Returns:
      scores: A `dict` of `tf.Tensor` which includes scores of the predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: A `dict` of `tf.Tensor` which includes coordinates of the
        predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
    """
    scores = {}
    boxes = {}
    for i, level in enumerate(
        range(self._config_dict['min_level'],
              self._config_dict['max_level'] + 1)):
      x = features[str(level)]
      for conv, norm in zip(self._convs, self._norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      scores[str(level)] = self._classifier(x)
      boxes[str(level)] = self._box_regressor(x)
    return scores, boxes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
