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

"""Classes to build various prediction heads in all supported models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from official.legacy.detection.modeling.architecture import nn_ops
from official.legacy.detection.ops import spatial_transform_ops


class RpnHead(tf.keras.layers.Layer):
  """Region Proposal Network head."""

  def __init__(
      self,
      min_level,
      max_level,
      anchors_per_location,
      num_convs=2,
      num_filters=256,
      use_separable_conv=False,
      activation='relu',
      use_batch_norm=True,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build Region Proposal Network head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      anchors_per_location: `int` number of number of anchors per pixel
        location.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    super().__init__(autocast=False)

    self._min_level = min_level
    self._max_level = max_level
    self._anchors_per_location = anchors_per_location
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm

    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.keras.layers.Conv2D,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          bias_initializer=tf.zeros_initializer())

    self._rpn_conv = self._conv2d_op(
        num_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=(None if self._use_batch_norm else self._activation_op),
        padding='same',
        name='rpn')
    self._rpn_class_conv = self._conv2d_op(
        anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name='rpn-class')
    self._rpn_box_conv = self._conv2d_op(
        4 * anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name='rpn-box')

    self._norm_activations = {}
    if self._use_batch_norm:
      for level in range(self._min_level, self._max_level + 1):
        self._norm_activations[level] = norm_activation(name='rpn-l%d-bn' %
                                                        level)

  def _shared_rpn_heads(self, features, anchors_per_location, level,
                        is_training):
    """Shared RPN heads."""
    features = self._rpn_conv(features)
    if self._use_batch_norm:
      # The batch normalization layers are not shared between levels.
      features = self._norm_activations[level](
          features, is_training=is_training)
    # Proposal classification scores
    scores = self._rpn_class_conv(features)
    # Proposal bbox regression deltas
    bboxes = self._rpn_box_conv(features)

    return scores, bboxes

  def call(self, features, is_training=None):

    scores_outputs = {}
    box_outputs = {}

    with tf.name_scope('rpn_head'):
      for level in range(self._min_level, self._max_level + 1):
        scores_output, box_output = self._shared_rpn_heads(
            features[level], self._anchors_per_location, level, is_training)
        scores_outputs[level] = scores_output
        box_outputs[level] = box_output
      return scores_outputs, box_outputs


class OlnRpnHead(tf.keras.layers.Layer):
  """Region Proposal Network for Object Localization Network (OLN)."""

  def __init__(
      self,
      min_level,
      max_level,
      anchors_per_location,
      num_convs=2,
      num_filters=256,
      use_separable_conv=False,
      activation='relu',
      use_batch_norm=True,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build Region Proposal Network head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      anchors_per_location: `int` number of number of anchors per pixel
        location.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    self._min_level = min_level
    self._max_level = max_level
    self._anchors_per_location = anchors_per_location
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm

    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.keras.layers.Conv2D,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          bias_initializer=tf.zeros_initializer())

    self._rpn_conv = self._conv2d_op(
        num_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=(None if self._use_batch_norm else self._activation_op),
        padding='same',
        name='rpn')
    self._rpn_class_conv = self._conv2d_op(
        anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name='rpn-class')
    self._rpn_box_conv = self._conv2d_op(
        4 * anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name='rpn-box-lrtb')
    self._rpn_center_conv = self._conv2d_op(
        anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name='rpn-centerness')

    self._norm_activations = {}
    if self._use_batch_norm:
      for level in range(self._min_level, self._max_level + 1):
        self._norm_activations[level] = norm_activation(name='rpn-l%d-bn' %
                                                        level)

  def _shared_rpn_heads(self, features, anchors_per_location, level,
                        is_training):
    """Shared RPN heads."""
    features = self._rpn_conv(features)
    if self._use_batch_norm:
      # The batch normalization layers are not shared between levels.
      features = self._norm_activations[level](
          features, is_training=is_training)
    # Feature L2 normalization for training stability
    features = tf.math.l2_normalize(
        features,
        axis=-1,
        name='rpn-norm',)
    # Proposal classification scores
    scores = self._rpn_class_conv(features)
    # Proposal bbox regression deltas
    bboxes = self._rpn_box_conv(features)
    # Proposal centerness scores
    centers = self._rpn_center_conv(features)

    return scores, bboxes, centers

  def __call__(self, features, is_training=None):

    scores_outputs = {}
    box_outputs = {}
    center_outputs = {}

    with tf.name_scope('rpn_head'):
      for level in range(self._min_level, self._max_level + 1):
        scores_output, box_output, center_output = self._shared_rpn_heads(
            features[level], self._anchors_per_location, level, is_training)
        scores_outputs[level] = scores_output
        box_outputs[level] = box_output
        center_outputs[level] = center_output
      return scores_outputs, box_outputs, center_outputs


class FastrcnnHead(tf.keras.layers.Layer):
  """Fast R-CNN box head."""

  def __init__(
      self,
      num_classes,
      num_convs=0,
      num_filters=256,
      use_separable_conv=False,
      num_fcs=2,
      fc_dims=1024,
      activation='relu',
      use_batch_norm=True,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build Fast R-CNN box head.

    Args:
      num_classes: a integer for the number of classes.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the FC layers.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      num_fcs: `int` number that represents the number of FC layers before the
        predictions.
      fc_dims: `int` number that represents the number of dimension of the FC
        layers.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    super(FastrcnnHead, self).__init__(autocast=False)

    self._num_classes = num_classes

    self._num_convs = num_convs
    self._num_filters = num_filters
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.keras.layers.Conv2D,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer())

    self._num_fcs = num_fcs
    self._fc_dims = fc_dims
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._norm_activation = norm_activation

    self._conv_ops = []
    self._conv_bn_ops = []
    for i in range(self._num_convs):
      self._conv_ops.append(
          self._conv2d_op(
              self._num_filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='same',
              dilation_rate=(1, 1),
              activation=(None
                          if self._use_batch_norm else self._activation_op),
              name='conv_{}'.format(i)))
      if self._use_batch_norm:
        self._conv_bn_ops.append(self._norm_activation())

    self._fc_ops = []
    self._fc_bn_ops = []
    for i in range(self._num_fcs):
      self._fc_ops.append(
          tf.keras.layers.Dense(
              units=self._fc_dims,
              activation=(None
                          if self._use_batch_norm else self._activation_op),
              name='fc{}'.format(i)))
      if self._use_batch_norm:
        self._fc_bn_ops.append(self._norm_activation(fused=False))

    self._class_predict = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name='class-predict')
    self._box_predict = tf.keras.layers.Dense(
        self._num_classes * 4,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        name='box-predict')

  def call(self, roi_features, is_training=None):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape [batch_size, num_rois,
        height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """

    with tf.name_scope(
        'fast_rcnn_head'):
      # reshape inputs beofre FC.
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()

      net = tf.reshape(roi_features, [-1, height, width, filters])
      for i in range(self._num_convs):
        net = self._conv_ops[i](net)
        if self._use_batch_norm:
          net = self._conv_bn_ops[i](net, is_training=is_training)

      filters = self._num_filters if self._num_convs > 0 else filters
      net = tf.reshape(net, [-1, num_rois, height * width * filters])

      for i in range(self._num_fcs):
        net = self._fc_ops[i](net)
        if self._use_batch_norm:
          net = self._fc_bn_ops[i](net, is_training=is_training)

      class_outputs = self._class_predict(net)
      box_outputs = self._box_predict(net)
      return class_outputs, box_outputs


class OlnBoxScoreHead(tf.keras.layers.Layer):
  """Box head of Object Localization Network (OLN)."""

  def __init__(
      self,
      num_classes,
      num_convs=0,
      num_filters=256,
      use_separable_conv=False,
      num_fcs=2,
      fc_dims=1024,
      activation='relu',
      use_batch_norm=True,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build OLN box head.

    Args:
      num_classes: a integer for the number of classes.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the FC layers.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      num_fcs: `int` number that represents the number of FC layers before the
        predictions.
      fc_dims: `int` number that represents the number of dimension of the FC
        layers.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    self._num_classes = num_classes

    self._num_convs = num_convs
    self._num_filters = num_filters
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.keras.layers.Conv2D,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer())

    self._num_fcs = num_fcs
    self._fc_dims = fc_dims
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._norm_activation = norm_activation

    self._conv_ops = []
    self._conv_bn_ops = []
    for i in range(self._num_convs):
      self._conv_ops.append(
          self._conv2d_op(
              self._num_filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='same',
              dilation_rate=(1, 1),
              activation=(None
                          if self._use_batch_norm else self._activation_op),
              name='conv_{}'.format(i)))
      if self._use_batch_norm:
        self._conv_bn_ops.append(self._norm_activation())

    self._fc_ops = []
    self._fc_bn_ops = []
    for i in range(self._num_fcs):
      self._fc_ops.append(
          tf.keras.layers.Dense(
              units=self._fc_dims,
              activation=(None
                          if self._use_batch_norm else self._activation_op),
              name='fc{}'.format(i)))
      if self._use_batch_norm:
        self._fc_bn_ops.append(self._norm_activation(fused=False))

    self._class_predict = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name='class-predict')
    self._box_predict = tf.keras.layers.Dense(
        self._num_classes * 4,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        name='box-predict')
    self._score_predict = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name='score-predict')

  def __call__(self, roi_features, is_training=None):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape [batch_size, num_rois,
        height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """

    with tf.name_scope('fast_rcnn_head'):
      # reshape inputs beofre FC.
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()

      net = tf.reshape(roi_features, [-1, height, width, filters])
      for i in range(self._num_convs):
        net = self._conv_ops[i](net)
        if self._use_batch_norm:
          net = self._conv_bn_ops[i](net, is_training=is_training)

      filters = self._num_filters if self._num_convs > 0 else filters
      net = tf.reshape(net, [-1, num_rois, height * width * filters])

      for i in range(self._num_fcs):
        net = self._fc_ops[i](net)
        if self._use_batch_norm:
          net = self._fc_bn_ops[i](net, is_training=is_training)

      class_outputs = self._class_predict(net)
      box_outputs = self._box_predict(net)
      score_outputs = self._score_predict(net)
      return class_outputs, box_outputs, score_outputs


class MaskrcnnHead(tf.keras.layers.Layer):
  """Mask R-CNN head."""

  def __init__(
      self,
      num_classes,
      mask_target_size,
      num_convs=4,
      num_filters=256,
      use_separable_conv=False,
      activation='relu',
      use_batch_norm=True,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build Fast R-CNN head.

    Args:
      num_classes: a integer for the number of classes.
      mask_target_size: a integer that is the resolution of masks.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    super(MaskrcnnHead, self).__init__(autocast=False)
    self._num_classes = num_classes
    self._mask_target_size = mask_target_size

    self._num_convs = num_convs
    self._num_filters = num_filters
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.keras.layers.Conv2D,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer())
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._norm_activation = norm_activation
    self._conv2d_ops = []
    for i in range(self._num_convs):
      self._conv2d_ops.append(
          self._conv2d_op(
              self._num_filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='same',
              dilation_rate=(1, 1),
              activation=(None
                          if self._use_batch_norm else self._activation_op),
              name='mask-conv-l%d' % i))
    self._mask_conv_transpose = tf.keras.layers.Conv2DTranspose(
        self._num_filters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        activation=(None if self._use_batch_norm else self._activation_op),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        bias_initializer=tf.zeros_initializer(),
        name='conv5-mask')

    with tf.name_scope('mask_head'):
      self._mask_conv2d_op = self._conv2d_op(
          self._num_classes,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='valid',
          name='mask_fcn_logits')

  def call(self, roi_features, class_indices, is_training=None):
    """Mask branch for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape [batch_size, num_rois,
        height_l, width_l, num_filters].
      class_indices: a Tensor of shape [batch_size, num_rois], indicating which
        class the ROI is.
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      mask_outputs: a tensor with a shape of
        [batch_size, num_masks, mask_height, mask_width, num_classes],
        representing the mask predictions.
      fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
        representing the fg mask targets.
    Raises:
      ValueError: If boxes is not a rank-3 tensor or the last dimension of
        boxes is not 4.
    """

    with tf.name_scope('mask_head'):
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()
      net = tf.reshape(roi_features, [-1, height, width, filters])

      for i in range(self._num_convs):
        net = self._conv2d_ops[i](net)
        if self._use_batch_norm:
          net = self._norm_activation()(net, is_training=is_training)

      net = self._mask_conv_transpose(net)
      if self._use_batch_norm:
        net = self._norm_activation()(net, is_training=is_training)

      mask_outputs = self._mask_conv2d_op(net)
      mask_outputs = tf.reshape(mask_outputs, [
          -1, num_rois, self._mask_target_size, self._mask_target_size,
          self._num_classes
      ])

      with tf.name_scope('masks_post_processing'):
        # TODO(pengchong): Figure out the way not to use the static inferred
        # batch size.
        batch_size, num_masks = class_indices.get_shape().as_list()
        mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])
        # Constructs indices for gather.
        batch_indices = tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1), [1, num_masks])
        mask_indices = tf.tile(
            tf.expand_dims(tf.range(num_masks), axis=0), [batch_size, 1])
        gather_indices = tf.stack(
            [batch_indices, mask_indices, class_indices], axis=2)
        mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
      return mask_outputs


class RetinanetHead(object):
  """RetinaNet head."""

  def __init__(
      self,
      min_level,
      max_level,
      num_classes,
      anchors_per_location,
      num_convs=4,
      num_filters=256,
      use_separable_conv=False,
      norm_activation=nn_ops.norm_activation_builder(activation='relu')):
    """Initialize params to build RetinaNet head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      num_classes: `int` number of classification categories.
      anchors_per_location: `int` number of anchors per pixel location.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      num_filters: `int` number of filters used in the head architecture.
      use_separable_conv: `bool` to indicate whether to use separable
        convoluation.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    self._min_level = min_level
    self._max_level = max_level

    self._num_classes = num_classes
    self._anchors_per_location = anchors_per_location

    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_separable_conv = use_separable_conv
    with tf.name_scope('class_net') as scope_name:
      self._class_name_scope = tf.name_scope(scope_name)
    with tf.name_scope('box_net') as scope_name:
      self._box_name_scope = tf.name_scope(scope_name)
    self._build_class_net_layers(norm_activation)
    self._build_box_net_layers(norm_activation)

  def _class_net_batch_norm_name(self, i, level):
    return 'class-%d-%d' % (i, level)

  def _box_net_batch_norm_name(self, i, level):
    return 'box-%d-%d' % (i, level)

  def _build_class_net_layers(self, norm_activation):
    """Build re-usable layers for class prediction network."""
    if self._use_separable_conv:
      self._class_predict = tf.keras.layers.SeparableConv2D(
          self._num_classes * self._anchors_per_location,
          kernel_size=(3, 3),
          bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
          padding='same',
          name='class-predict')
    else:
      self._class_predict = tf.keras.layers.Conv2D(
          self._num_classes * self._anchors_per_location,
          kernel_size=(3, 3),
          bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
          padding='same',
          name='class-predict')
    self._class_conv = []
    self._class_norm_activation = {}
    for i in range(self._num_convs):
      if self._use_separable_conv:
        self._class_conv.append(
            tf.keras.layers.SeparableConv2D(
                self._num_filters,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                padding='same',
                name='class-' + str(i)))
      else:
        self._class_conv.append(
            tf.keras.layers.Conv2D(
                self._num_filters,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.01),
                activation=None,
                padding='same',
                name='class-' + str(i)))
      for level in range(self._min_level, self._max_level + 1):
        name = self._class_net_batch_norm_name(i, level)
        self._class_norm_activation[name] = norm_activation(name=name)

  def _build_box_net_layers(self, norm_activation):
    """Build re-usable layers for box prediction network."""
    if self._use_separable_conv:
      self._box_predict = tf.keras.layers.SeparableConv2D(
          4 * self._anchors_per_location,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name='box-predict')
    else:
      self._box_predict = tf.keras.layers.Conv2D(
          4 * self._anchors_per_location,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
          padding='same',
          name='box-predict')
    self._box_conv = []
    self._box_norm_activation = {}
    for i in range(self._num_convs):
      if self._use_separable_conv:
        self._box_conv.append(
            tf.keras.layers.SeparableConv2D(
                self._num_filters,
                kernel_size=(3, 3),
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-' + str(i)))
      else:
        self._box_conv.append(
            tf.keras.layers.Conv2D(
                self._num_filters,
                kernel_size=(3, 3),
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.01),
                padding='same',
                name='box-' + str(i)))
      for level in range(self._min_level, self._max_level + 1):
        name = self._box_net_batch_norm_name(i, level)
        self._box_norm_activation[name] = norm_activation(name=name)

  def __call__(self, fpn_features, is_training=None):
    """Returns outputs of RetinaNet head."""
    class_outputs = {}
    box_outputs = {}
    with tf.name_scope('retinanet_head'):
      for level in range(self._min_level, self._max_level + 1):
        features = fpn_features[level]

        class_outputs[level] = self.class_net(
            features, level, is_training=is_training)
        box_outputs[level] = self.box_net(
            features, level, is_training=is_training)
    return class_outputs, box_outputs

  def class_net(self, features, level, is_training):
    """Class prediction network for RetinaNet."""
    with self._class_name_scope:
      for i in range(self._num_convs):
        features = self._class_conv[i](features)
        # The convolution layers in the class net are shared among all levels,
        # but each level has its batch normlization to capture the statistical
        # difference among different levels.
        name = self._class_net_batch_norm_name(i, level)
        features = self._class_norm_activation[name](
            features, is_training=is_training)

      classes = self._class_predict(features)
    return classes

  def box_net(self, features, level, is_training=None):
    """Box regression network for RetinaNet."""
    with self._box_name_scope:
      for i in range(self._num_convs):
        features = self._box_conv[i](features)
        # The convolution layers in the box net are shared among all levels, but
        # each level has its batch normlization to capture the statistical
        # difference among different levels.
        name = self._box_net_batch_norm_name(i, level)
        features = self._box_norm_activation[name](
            features, is_training=is_training)

      boxes = self._box_predict(features)
    return boxes


# TODO(yeqing): Refactor this class when it is ready for var_scope reuse.
class ShapemaskPriorHead(object):
  """ShapeMask Prior head."""

  def __init__(self, num_classes, num_downsample_channels, mask_crop_size,
               use_category_for_mask, shape_prior_path):
    """Initialize params to build RetinaNet head.

    Args:
      num_classes: Number of output classes.
      num_downsample_channels: number of channels in mask branch.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      shape_prior_path: the path to load shape priors.
    """
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._shape_prior_path = shape_prior_path
    self._use_category_for_mask = use_category_for_mask

    self._shape_prior_fc = tf.keras.layers.Dense(
        self._num_downsample_channels, name='shape-prior-fc')

  def __call__(self, fpn_features, boxes, outer_boxes, classes, is_training):
    """Generate the detection priors from the box detections and FPN features.

    This corresponds to the Fig. 4 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      fpn_features: a dictionary of FPN features.
      boxes: a float tensor of shape [batch_size, num_instances, 4] representing
        the tight gt boxes from dataloader/detection.
      outer_boxes: a float tensor of shape [batch_size, num_instances, 4]
        representing the loose gt boxes from dataloader/detection.
      classes: a int Tensor of shape [batch_size, num_instances] of instance
        classes.
      is_training: training mode or not.

    Returns:
      instance_features: a float Tensor of shape [batch_size * num_instances,
          mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
          instance feature crop.
      detection_priors: A float Tensor of shape [batch_size * num_instances,
        mask_size, mask_size, 1].
    """
    with tf.name_scope('prior_mask'):
      batch_size, num_instances, _ = boxes.get_shape().as_list()
      outer_boxes = tf.cast(outer_boxes, tf.float32)
      boxes = tf.cast(boxes, tf.float32)
      instance_features = spatial_transform_ops.multilevel_crop_and_resize(
          fpn_features, outer_boxes, output_size=self._mask_crop_size)
      instance_features = self._shape_prior_fc(instance_features)

      shape_priors = self._get_priors()

      # Get uniform priors for each outer box.
      uniform_priors = tf.ones([
          batch_size, num_instances, self._mask_crop_size, self._mask_crop_size
      ])
      uniform_priors = spatial_transform_ops.crop_mask_in_target_box(
          uniform_priors, boxes, outer_boxes, self._mask_crop_size)

      # Classify shape priors using uniform priors + instance features.
      prior_distribution = self._classify_shape_priors(
          tf.cast(instance_features, tf.float32), uniform_priors, classes)

      instance_priors = tf.gather(shape_priors, classes)
      instance_priors *= tf.expand_dims(
          tf.expand_dims(tf.cast(prior_distribution, tf.float32), axis=-1),
          axis=-1)
      instance_priors = tf.reduce_sum(instance_priors, axis=2)
      detection_priors = spatial_transform_ops.crop_mask_in_target_box(
          instance_priors, boxes, outer_boxes, self._mask_crop_size)

      return instance_features, detection_priors

  def _get_priors(self):
    """Load shape priors from file."""
    # loads class specific or agnostic shape priors
    if self._shape_prior_path:
      # Priors are loaded into shape [mask_num_classes, num_clusters, 32, 32].
      priors = np.load(tf.io.gfile.GFile(self._shape_prior_path, 'rb'))
      priors = tf.convert_to_tensor(priors, dtype=tf.float32)
      self._num_clusters = priors.get_shape().as_list()[1]
    else:
      # If prior path does not exist, do not use priors, i.e., pirors equal to
      # uniform empty 32x32 patch.
      self._num_clusters = 1
      priors = tf.zeros([
          self._mask_num_classes, self._num_clusters, self._mask_crop_size,
          self._mask_crop_size
      ])
    return priors

  def _classify_shape_priors(self, features, uniform_priors, classes):
    """Classify the uniform prior by predicting the shape modes.

    Classify the object crop features into K modes of the clusters for each
    category.

    Args:
      features: A float Tensor of shape [batch_size, num_instances, mask_size,
        mask_size, num_channels].
      uniform_priors: A float Tensor of shape [batch_size, num_instances,
        mask_size, mask_size] representing the uniform detection priors.
      classes: A int Tensor of shape [batch_size, num_instances] of detection
        class ids.

    Returns:
      prior_distribution: A float Tensor of shape
        [batch_size, num_instances, num_clusters] representing the classifier
        output probability over all possible shapes.
    """

    batch_size, num_instances, _, _, _ = features.get_shape().as_list()
    features *= tf.expand_dims(uniform_priors, axis=-1)
    # Reduce spatial dimension of features. The features have shape
    # [batch_size, num_instances, num_channels].
    features = tf.reduce_mean(features, axis=(2, 3))
    logits = tf.keras.layers.Dense(
        self._mask_num_classes * self._num_clusters,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        name='classify-shape-prior-fc')(features)
    logits = tf.reshape(
        logits,
        [batch_size, num_instances, self._mask_num_classes, self._num_clusters])
    if self._use_category_for_mask:
      logits = tf.gather(logits, tf.expand_dims(classes, axis=-1), batch_dims=2)
      logits = tf.squeeze(logits, axis=2)
    else:
      logits = logits[:, :, 0, :]

    distribution = tf.nn.softmax(logits, name='shape_prior_weights')
    return distribution


class ShapemaskCoarsemaskHead(object):
  """ShapemaskCoarsemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_convs,
               norm_activation=nn_ops.norm_activation_builder()):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      norm_activation: an operation that includes a normalization layer followed
        by an optional activation layer.
    """
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._use_category_for_mask = use_category_for_mask
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._num_convs = num_convs
    self._norm_activation = norm_activation

    self._coarse_mask_fc = tf.keras.layers.Dense(
        self._num_downsample_channels, name='coarse-mask-fc')

    self._class_conv = []
    self._class_norm_activation = []

    for i in range(self._num_convs):
      self._class_conv.append(
          tf.keras.layers.Conv2D(
              self._num_downsample_channels,
              kernel_size=(3, 3),
              bias_initializer=tf.zeros_initializer(),
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  stddev=0.01),
              padding='same',
              name='coarse-mask-class-%d' % i))

      self._class_norm_activation.append(
          norm_activation(name='coarse-mask-class-%d-bn' % i))

    self._class_predict = tf.keras.layers.Conv2D(
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        padding='same',
        name='coarse-mask-class-predict')

  def __call__(self, features, detection_priors, classes, is_training):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      features: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
      detection_priors: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, 1]. This is the detection prior for the
        instance.
      classes: a int Tensor of shape [batch_size, num_instances] of instance
        classes.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size, num_instances, mask_size, mask_size].
    """
    with tf.name_scope('coarse_mask'):
      # Transform detection priors to have the same dimension as features.
      detection_priors = tf.expand_dims(detection_priors, axis=-1)
      detection_priors = self._coarse_mask_fc(detection_priors)

      features += detection_priors
      mask_logits = self.decoder_net(features, is_training)
      # Gather the logits with right input class.
      if self._use_category_for_mask:
        mask_logits = tf.transpose(mask_logits, [0, 1, 4, 2, 3])
        mask_logits = tf.gather(
            mask_logits, tf.expand_dims(classes, -1), batch_dims=2)
        mask_logits = tf.squeeze(mask_logits, axis=2)
      else:
        mask_logits = mask_logits[..., 0]

      return mask_logits

  def decoder_net(self, features, is_training=False):
    """Coarse mask decoder network architecture.

    Args:
      features: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.

    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels]
    """
    (batch_size, num_instances, height, width,
     num_channels) = features.get_shape().as_list()
    features = tf.reshape(
        features, [batch_size * num_instances, height, width, num_channels])
    for i in range(self._num_convs):
      features = self._class_conv[i](features)
      features = self._class_norm_activation[i](
          features, is_training=is_training)

    mask_logits = self._class_predict(features)
    mask_logits = tf.reshape(
        mask_logits,
        [batch_size, num_instances, height, width, self._mask_num_classes])
    return mask_logits


class ShapemaskFinemaskHead(object):
  """ShapemaskFinemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_convs,
               upsample_factor,
               norm_activation=nn_ops.norm_activation_builder()):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      upsample_factor: `int` number of fine mask upsampling factor.
      norm_activation: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._use_category_for_mask = use_category_for_mask
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._num_convs = num_convs
    self.up_sample_factor = upsample_factor

    self._fine_mask_fc = tf.keras.layers.Dense(
        self._num_downsample_channels, name='fine-mask-fc')

    self._upsample_conv = tf.keras.layers.Conv2DTranspose(
        self._num_downsample_channels,
        (self.up_sample_factor, self.up_sample_factor),
        (self.up_sample_factor, self.up_sample_factor),
        name='fine-mask-conv2d-tran')

    self._fine_class_conv = []
    self._fine_class_bn = []
    for i in range(self._num_convs):
      self._fine_class_conv.append(
          tf.keras.layers.Conv2D(
              self._num_downsample_channels,
              kernel_size=(3, 3),
              bias_initializer=tf.zeros_initializer(),
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  stddev=0.01),
              activation=None,
              padding='same',
              name='fine-mask-class-%d' % i))
      self._fine_class_bn.append(
          norm_activation(name='fine-mask-class-%d-bn' % i))

    self._class_predict_conv = tf.keras.layers.Conv2D(
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        padding='same',
        name='fine-mask-class-predict')

  def __call__(self, features, mask_logits, classes, is_training):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      features: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
      mask_logits: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size] indicating predicted mask logits.
      classes: a int Tensor of shape [batch_size, num_instances] of instance
        classes.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size, num_instances, mask_size, mask_size].
    """
    # Extract the foreground mean features
    # with tf.variable_scope('fine_mask', reuse=tf.AUTO_REUSE):
    with tf.name_scope('fine_mask'):
      mask_probs = tf.nn.sigmoid(mask_logits)
      # Compute instance embedding for hard average.
      binary_mask = tf.cast(tf.greater(mask_probs, 0.5), features.dtype)
      instance_embedding = tf.reduce_sum(
          features * tf.expand_dims(binary_mask, axis=-1), axis=(2, 3))
      instance_embedding /= tf.expand_dims(
          tf.reduce_sum(binary_mask, axis=(2, 3)) + 1e-20, axis=-1)
      # Take the difference between crop features and mean instance features.
      features -= tf.expand_dims(
          tf.expand_dims(instance_embedding, axis=2), axis=2)

      features += self._fine_mask_fc(tf.expand_dims(mask_probs, axis=-1))

      # Decoder to generate upsampled segmentation mask.
      mask_logits = self.decoder_net(features, is_training)
      if self._use_category_for_mask:
        mask_logits = tf.transpose(mask_logits, [0, 1, 4, 2, 3])
        mask_logits = tf.gather(
            mask_logits, tf.expand_dims(classes, -1), batch_dims=2)
        mask_logits = tf.squeeze(mask_logits, axis=2)
      else:
        mask_logits = mask_logits[..., 0]

    return mask_logits

  def decoder_net(self, features, is_training=False):
    """Fine mask decoder network architecture.

    Args:
      features: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.

    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels], where output size is self._gt_upsample_scale times
        that of input.
    """
    (batch_size, num_instances, height, width,
     num_channels) = features.get_shape().as_list()
    features = tf.reshape(
        features, [batch_size * num_instances, height, width, num_channels])
    for i in range(self._num_convs):
      features = self._fine_class_conv[i](features)
      features = self._fine_class_bn[i](features, is_training=is_training)

    if self.up_sample_factor > 1:
      features = self._upsample_conv(features)

    # Predict per-class instance masks.
    mask_logits = self._class_predict_conv(features)

    mask_logits = tf.reshape(mask_logits, [
        batch_size, num_instances, height * self.up_sample_factor,
        width * self.up_sample_factor, self._mask_num_classes
    ])
    return mask_logits
