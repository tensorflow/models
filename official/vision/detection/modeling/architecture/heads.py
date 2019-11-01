# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Classes to build various prediction heads in all supported models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.keras import backend
from official.vision.detection.modeling.architecture import nn_ops
from official.vision.detection.ops import spatial_transform_ops


class RpnHead(object):
  """Region Proposal Network head."""

  def __init__(self,
               min_level,
               max_level,
               anchors_per_location,
               batch_norm_relu=nn_ops.BatchNormRelu):
    """Initialize params to build Region Proposal Network head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      anchors_per_location: `int` number of number of anchors per pixel
        location.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._min_level = min_level
    self._max_level = max_level
    self._anchors_per_location = anchors_per_location
    self._rpn_conv = tf.keras.layers.Conv2D(
        256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        padding='same',
        name='rpn')
    self._rpn_class_conv = tf.keras.layers.Conv2D(
        anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        padding='valid',
        name='rpn-class')
    self._rpn_box_conv = tf.keras.layers.Conv2D(
        4 * anchors_per_location,
        kernel_size=(1, 1),
        strides=(1, 1),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        padding='valid',
        name='rpn-box')
    self._batch_norm_relus = {}
    for level in range(self._min_level, self._max_level + 1):
      self._batch_norm_relus[level] = batch_norm_relu(name='rpn%d-bn' % level)

  def _shared_rpn_heads(self, features, anchors_per_location, level,
                        is_training):
    """Shared RPN heads."""
    # TODO(chiachenc): check the channel depth of the first convoultion.
    features = self._rpn_conv(features)
    # The batch normalization layers are not shared between levels.
    features = self._batch_norm_relus[level](features, is_training=is_training)
    # Proposal classification scores
    scores = self._rpn_class_conv(features)
    # Proposal bbox regression deltas
    bboxes = self._rpn_box_conv(features)

    return scores, bboxes

  def __call__(self, features, is_training=None):

    scores_outputs = {}
    box_outputs = {}

    with backend.get_graph().as_default(), tf.name_scope('rpn_head'):
      for level in range(self._min_level, self._max_level + 1):
        scores_output, box_output = self._shared_rpn_heads(
            features[level], self._anchors_per_location, level, is_training)
        scores_outputs[level] = scores_output
        box_outputs[level] = box_output
      return scores_outputs, box_outputs


class FastrcnnHead(object):
  """Fast R-CNN box head."""

  def __init__(self,
               num_classes,
               mlp_head_dim,
               batch_norm_relu=nn_ops.BatchNormRelu):
    """Initialize params to build Fast R-CNN box head.

    Args:
      num_classes: a integer for the number of classes.
      mlp_head_dim: a integer that is the hidden dimension in the
        fully-connected layers.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._num_classes = num_classes
    self._mlp_head_dim = mlp_head_dim
    self._batch_norm_relu = batch_norm_relu

  def __call__(self, roi_features, is_training=None):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape
        [batch_size, num_rois, height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """

    with backend.get_graph().as_default(), tf.name_scope('fast_rcnn_head'):
      # reshape inputs beofre FC.
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()
      roi_features = tf.reshape(roi_features,
                                [-1, num_rois, height * width * filters])
      net = tf.keras.layers.Dense(
          units=self._mlp_head_dim, activation=None, name='fc6')(
              roi_features)

      net = self._batch_norm_relu(fused=False)(net, is_training=is_training)
      net = tf.keras.layers.Dense(
          units=self._mlp_head_dim, activation=None, name='fc7')(
              net)
      net = self._batch_norm_relu(fused=False)(net, is_training=is_training)

      class_outputs = tf.keras.layers.Dense(
          self._num_classes,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          bias_initializer=tf.zeros_initializer(),
          name='class-predict')(
              net)
      box_outputs = tf.keras.layers.Dense(
          self._num_classes * 4,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
          bias_initializer=tf.zeros_initializer(),
          name='box-predict')(
              net)
      return class_outputs, box_outputs


class MaskrcnnHead(object):
  """Mask R-CNN head."""

  def __init__(self,
               num_classes,
               mask_target_size,
               batch_norm_relu=nn_ops.BatchNormRelu):
    """Initialize params to build Fast R-CNN head.

    Args:
      num_classes: a integer for the number of classes.
      mask_target_size: a integer that is the resolution of masks.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._num_classes = num_classes
    self._mask_target_size = mask_target_size
    self._batch_norm_relu = batch_norm_relu

  def __call__(self, roi_features, class_indices, is_training=None):
    """Mask branch for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape
        [batch_size, num_rois, height_l, width_l, num_filters].
      class_indices: a Tensor of shape [batch_size, num_rois], indicating
        which class the ROI is.
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

    def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
      """Returns the stddev of random normal initialization as MSRAFill."""
      # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463  # pylint: disable=line-too-long
      # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
      # stddev = (2/(3*3*256))^0.5 = 0.029
      return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

    with backend.get_graph().as_default():
      with tf.name_scope('mask_head'):
        _, num_rois, height, width, filters = roi_features.get_shape().as_list()
        net = tf.reshape(roi_features, [-1, height, width, filters])

        for i in range(4):
          kernel_size = (3, 3)
          fan_out = 256
          init_stddev = _get_stddev_equivalent_to_msra_fill(
              kernel_size, fan_out)
          net = tf.keras.layers.Conv2D(
              fan_out,
              kernel_size=kernel_size,
              strides=(1, 1),
              padding='same',
              dilation_rate=(1, 1),
              activation=None,
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  stddev=init_stddev),
              bias_initializer=tf.zeros_initializer(),
              name='mask-conv-l%d' % i)(
                  net)
          net = self._batch_norm_relu()(net, is_training=is_training)

        kernel_size = (2, 2)
        fan_out = 256
        init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        net = tf.keras.layers.Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=init_stddev),
            bias_initializer=tf.zeros_initializer(),
            name='conv5-mask')(
                net)
        net = self._batch_norm_relu()(net, is_training=is_training)

        kernel_size = (1, 1)
        fan_out = self._num_classes
        init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        mask_outputs = tf.keras.layers.Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=init_stddev),
            bias_initializer=tf.zeros_initializer(),
            name='mask_fcn_logits')(
                net)
        mask_outputs = tf.reshape(mask_outputs, [
            -1, num_rois, self._mask_target_size, self._mask_target_size,
            self._num_classes
        ])

        with tf.name_scope('masks_post_processing'):
          # TODO(pengchong): Figure out the way not to use the static inferred
          # batch size.
          batch_size, num_masks = class_indices.get_shape().as_list()
          mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])
          # Contructs indices for gather.
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

  def __init__(self,
               min_level,
               max_level,
               num_classes,
               anchors_per_location,
               num_convs=4,
               num_filters=256,
               batch_norm_relu=nn_ops.BatchNormRelu):
    """Initialize params to build RetinaNet head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      num_classes: `int` number of classification categories.
      anchors_per_location: `int` number of anchors per pixel location.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      num_filters: `int` number of filters used in the head architecture.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._min_level = min_level
    self._max_level = max_level

    self._num_classes = num_classes
    self._anchors_per_location = anchors_per_location

    self._num_convs = num_convs
    self._num_filters = num_filters

    with tf.name_scope('class_net') as scope_name:
      self._class_name_scope = tf.name_scope(scope_name)
    with tf.name_scope('box_net') as scope_name:
      self._box_name_scope = tf.name_scope(scope_name)
    self._build_class_net_layers(batch_norm_relu)
    self._build_box_net_layers(batch_norm_relu)

  def _class_net_batch_norm_name(self, i, level):
    return 'class-%d-%d' % (i, level)

  def _box_net_batch_norm_name(self, i, level):
    return 'box-%d-%d' % (i, level)

  def _build_class_net_layers(self, batch_norm_relu):
    """Build re-usable layers for class prediction network."""
    self._class_predict = tf.keras.layers.Conv2D(
        self._num_classes * self._anchors_per_location,
        kernel_size=(3, 3),
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        padding='same',
        name='class-predict')
    self._class_conv = []
    self._class_batch_norm_relu = {}
    for i in range(self._num_convs):
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
        self._class_batch_norm_relu[name] = batch_norm_relu(name=name)

  def _build_box_net_layers(self, batch_norm_relu):
    """Build re-usable layers for box prediction network."""
    self._box_predict = tf.keras.layers.Conv2D(
        4 * self._anchors_per_location,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        padding='same',
        name='box-predict')
    self._box_conv = []
    self._box_batch_norm_relu = {}
    for i in range(self._num_convs):
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
        self._box_batch_norm_relu[name] = batch_norm_relu(name=name)

  def __call__(self, fpn_features, is_training=None):
    """Returns outputs of RetinaNet head."""
    class_outputs = {}
    box_outputs = {}
    with backend.get_graph().as_default(), tf.name_scope('retinanet'):
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
        # The convolution layers in the class net are shared among all levels, but
        # each level has its batch normlization to capture the statistical
        # difference among different levels.
        name = self._class_net_batch_norm_name(i, level)
        features = self._class_batch_norm_relu[name](
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
        features = self._box_batch_norm_relu[name](
            features, is_training=is_training)

      boxes = self._box_predict(features)
    return boxes


# TODO(yeqing): Refactor this class when it is ready for var_scope reuse.
class ShapemaskPriorHead(object):
  """ShapeMask Prior head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_of_instances,
               min_mask_level,
               max_mask_level,
               num_clusters,
               temperature,
               shape_prior_path=None):
    """Initialize params to build RetinaNet head.

    Args:
      num_classes: Number of output classes.
      num_downsample_channels: number of channels in mask branch.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_of_instances: number of instances to sample in training time.
      min_mask_level: minimum FPN level to crop mask feature from.
      max_mask_level: maximum FPN level to crop mask feature from.
      num_clusters: number of clusters to use in K-Means.
      temperature: the temperature for shape prior learning.
      shape_prior_path: the path to load shape priors.
    """
    self._mask_num_classes = num_classes
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._use_category_for_mask = use_category_for_mask
    self._num_of_instances = num_of_instances
    self._min_mask_level = min_mask_level
    self._max_mask_level = max_mask_level
    self._num_clusters = num_clusters
    self._temperature = temperature
    self._shape_prior_path = shape_prior_path

  def __call__(self,
               fpn_features,
               boxes,
               outer_boxes,
               classes,
               is_training=None):
    """Generate the detection priors from the box detections and FPN features.

    This corresponds to the Fig. 4 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      fpn_features: a dictionary of FPN features.
      boxes: a float tensor of shape [batch_size, num_instances, 4]
        representing the tight gt boxes from dataloader/detection.
      outer_boxes: a float tensor of shape [batch_size, num_instances, 4]
        representing the loose gt boxes from dataloader/detection.
      classes: a int Tensor of shape [batch_size, num_instances]
        of instance classes.
      is_training: training mode or not.

    Returns:
      crop_features: a float Tensor of shape [batch_size * num_instances,
          mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
          instance feature crop.
      detection_priors: A float Tensor of shape [batch_size * num_instances,
        mask_size, mask_size, 1].
    """
    with backend.get_graph().as_default():
      # loads class specific or agnostic shape priors
      if self._shape_prior_path:
        if self._use_category_for_mask:
          fid = tf.io.gfile.GFile(self._shape_prior_path, 'rb')
          # The encoding='bytes' options is for incompatibility between python2
          # and python3 pickle.
          class_tups = pickle.load(fid, encoding='bytes')
          max_class_id = class_tups[-1][0] + 1
          class_masks = np.zeros((max_class_id, self._num_clusters,
                                  self._mask_crop_size, self._mask_crop_size),
                                 dtype=np.float32)
          for cls_id, _, cls_mask in class_tups:
            assert cls_mask.shape == (self._num_clusters,
                                      self._mask_crop_size**2)
            class_masks[cls_id] = cls_mask.reshape(self._num_clusters,
                                                   self._mask_crop_size,
                                                   self._mask_crop_size)

          self.class_priors = tf.convert_to_tensor(
              value=class_masks, dtype=tf.float32)
        else:
          npy_path = tf.io.gfile.GFile(self._shape_prior_path)
          class_np_masks = np.load(npy_path)
          assert class_np_masks.shape == (
              self._num_clusters, self._mask_crop_size,
              self._mask_crop_size), 'Invalid priors!!!'
          self.class_priors = tf.convert_to_tensor(
              value=class_np_masks, dtype=tf.float32)
      else:
        self.class_priors = tf.zeros(
            [self._num_clusters, self._mask_crop_size, self._mask_crop_size],
            tf.float32)

      batch_size = boxes.get_shape()[0]
      min_level_shape = fpn_features[self._min_mask_level].get_shape().as_list()
      self._max_feature_size = min_level_shape[1]
      detection_prior_levels = self._compute_box_levels(boxes)
      level_outer_boxes = outer_boxes / tf.pow(
          2., tf.expand_dims(detection_prior_levels, -1))
      detection_prior_levels = tf.cast(detection_prior_levels, tf.int32)
      uniform_priors = spatial_transform_ops.crop_mask_in_target_box(
          tf.ones([
              batch_size, self._num_of_instances, self._mask_crop_size,
              self._mask_crop_size
          ], tf.float32), boxes, outer_boxes, self._mask_crop_size)

      # Prepare crop features.
      multi_level_features = self._get_multilevel_features(fpn_features)
      crop_features = spatial_transform_ops.single_level_feature_crop(
          multi_level_features, level_outer_boxes, detection_prior_levels,
          self._min_mask_level, self._mask_crop_size)

      # Predict and fuse shape priors.
      shape_weights = self._classify_and_fuse_detection_priors(
          uniform_priors, classes, crop_features)
      fused_shape_priors = self._fuse_priors(shape_weights, classes)
      fused_shape_priors = tf.reshape(fused_shape_priors, [
          batch_size, self._num_of_instances, self._mask_crop_size,
          self._mask_crop_size
      ])
      predicted_detection_priors = spatial_transform_ops.crop_mask_in_target_box(
          fused_shape_priors, boxes, outer_boxes, self._mask_crop_size)
      predicted_detection_priors = tf.reshape(
          predicted_detection_priors,
          [-1, self._mask_crop_size, self._mask_crop_size, 1])

      return crop_features, predicted_detection_priors

  def _get_multilevel_features(self, fpn_features):
    """Get multilevel features from FPN feature dictionary into one tensor.

    Args:
      fpn_features: a dictionary of FPN features.

    Returns:
      features: a float tensor of shape [batch_size, num_levels,
        max_feature_size, max_feature_size, num_downsample_channels].
    """
    # TODO(yeqing): Recover reuse=tf.AUTO_REUSE logic.
    with tf.name_scope('masknet'):
      mask_feats = {}
      # Reduce the feature dimension at each FPN level by convolution.
      for feat_level in range(self._min_mask_level, self._max_mask_level + 1):
        mask_feats[feat_level] = tf.keras.layers.Conv2D(
            self._num_downsample_channels,
            kernel_size=(1, 1),
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            padding='same',
            name='mask-downsample')(
                fpn_features[feat_level])

      # Concat features through padding to the max size.
      features = [mask_feats[self._min_mask_level]]
      for feat_level in range(self._min_mask_level + 1,
                              self._max_mask_level + 1):
        features.append(tf.image.pad_to_bounding_box(
            mask_feats[feat_level], 0, 0,
            self._max_feature_size, self._max_feature_size))

      features = tf.stack(features, axis=1)

    return features

  def _compute_box_levels(self, boxes):
    """Compute the box FPN levels.

    Args:
      boxes: a float tensor of shape [batch_size, num_instances, 4].

    Returns:
      levels: a int tensor of shape [batch_size, num_instances].
    """
    object_sizes = tf.stack([
        boxes[:, :, 2] - boxes[:, :, 0],
        boxes[:, :, 3] - boxes[:, :, 1],
    ], axis=2)
    object_sizes = tf.reduce_max(input_tensor=object_sizes, axis=2)
    ratios = object_sizes / self._mask_crop_size
    levels = tf.math.ceil(tf.math.log(ratios) / tf.math.log(2.))
    levels = tf.maximum(tf.minimum(levels, self._max_mask_level),
                        self._min_mask_level)
    return levels

  def _classify_and_fuse_detection_priors(self, uniform_priors,
                                          detection_prior_classes,
                                          crop_features):
    """Classify the uniform prior by predicting the shape modes.

    Classify the object crop features into K modes of the clusters for each
    category.

    Args:
      uniform_priors: A float Tensor of shape [batch_size, num_instances,
        mask_size, mask_size] representing the uniform detection priors.
      detection_prior_classes: A int Tensor of shape [batch_size, num_instances]
        of detection class ids.
      crop_features: A float Tensor of shape [batch_size * num_instances,
        mask_size, mask_size, num_channels].

    Returns:
      shape_weights: A float Tensor of shape
        [batch_size * num_instances, num_clusters] representing the classifier
        output probability over all possible shapes.
    """
    location_detection_priors = tf.reshape(
        uniform_priors, [-1, self._mask_crop_size, self._mask_crop_size, 1])
    # Generate image embedding to shape.
    fused_shape_features = crop_features * location_detection_priors

    shape_embedding = tf.reduce_mean(
        input_tensor=fused_shape_features, axis=(1, 2))
    if not self._use_category_for_mask:
      # TODO(weicheng) use custom op for performance
      shape_logits = tf.keras.layers.Dense(
          self._num_clusters,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(
              shape_embedding)
      shape_logits = tf.reshape(shape_logits,
                                [-1, self._num_clusters]) / self._temperature
      shape_weights = tf.nn.softmax(shape_logits, name='shape_prior_weights')
    else:
      shape_logits = tf.keras.layers.Dense(
          self._mask_num_classes * self._num_clusters,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(
              shape_embedding)
      shape_logits = tf.reshape(
          shape_logits, [-1, self._mask_num_classes, self._num_clusters])
      training_classes = tf.reshape(detection_prior_classes, [-1])
      class_idx = tf.stack(
          [tf.range(tf.size(input=training_classes)), training_classes - 1],
          axis=1)
      shape_logits = tf.gather_nd(shape_logits, class_idx) / self._temperature
      shape_weights = tf.nn.softmax(shape_logits, name='shape_prior_weights')

    return shape_weights

  def _fuse_priors(self, shape_weights, detection_prior_classes):
    """Fuse shape priors by the predicted shape probability.

    Args:
      shape_weights: A float Tensor of shape [batch_size * num_instances,
        num_clusters] of predicted shape probability distribution.
      detection_prior_classes: A int Tensor of shape [batch_size, num_instances]
        of detection class ids.

    Returns:
      detection_priors: A float Tensor of shape [batch_size * num_instances,
        mask_size, mask_size, 1].
    """
    if self._use_category_for_mask:
      object_class_priors = tf.gather(
          self.class_priors, detection_prior_classes)
    else:
      num_batch_instances = shape_weights.get_shape()[0]
      object_class_priors = tf.tile(
          tf.expand_dims(self.class_priors, 0),
          [num_batch_instances, 1, 1, 1])

    vector_class_priors = tf.reshape(
        object_class_priors,
        [-1, self._num_clusters,
         self._mask_crop_size * self._mask_crop_size])
    detection_priors = tf.matmul(
        tf.expand_dims(shape_weights, 1), vector_class_priors)[:, 0, :]
    detection_priors = tf.reshape(
        detection_priors, [-1, self._mask_crop_size, self._mask_crop_size, 1])
    return detection_priors


class ShapemaskCoarsemaskHead(object):
  """ShapemaskCoarsemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_convs):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
    """
    self._mask_num_classes = num_classes
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._use_category_for_mask = use_category_for_mask
    self._num_convs = num_convs
    if not use_category_for_mask:
      assert num_classes == 1

  def __call__(self,
               crop_features,
               detection_priors,
               inst_classes,
               is_training=None):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      crop_features: a float Tensor of shape [batch_size * num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
      detection_priors: a float Tensor of shape [batch_size * num_instances,
        mask_crop_size, mask_crop_size, 1]. This is the detection prior for
        the instance.
      inst_classes: a int Tensor of shape [batch_size, num_instances]
        of instance classes.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size * num_instances, mask_size, mask_size, num_classes].
    """
    # Embed the anchor map into some feature space for anchor conditioning.
    detection_prior_features = tf.keras.layers.Conv2D(
        self._num_downsample_channels,
        kernel_size=(1, 1),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0., stddev=0.01),
        padding='same',
        name='anchor-conv')(
            detection_priors)

    prior_conditioned_features = crop_features + detection_prior_features
    coarse_output_features = self.coarsemask_decoder_net(
        prior_conditioned_features, is_training)

    coarse_mask_classes = tf.keras.layers.Conv2D(
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0, stddev=0.01),
        padding='same',
        name='class-predict')(
            coarse_output_features)

    if self._use_category_for_mask:
      inst_classes = tf.cast(tf.reshape(inst_classes, [-1]), tf.int32)
      coarse_mask_classes_t = tf.transpose(
          a=coarse_mask_classes, perm=(0, 3, 1, 2))
      # pylint: disable=g-long-lambda
      coarse_mask_logits = tf.cond(
          pred=tf.size(input=inst_classes) > 0,
          true_fn=lambda: tf.gather_nd(
              coarse_mask_classes_t,
              tf.stack(
                  [tf.range(tf.size(input=inst_classes)), inst_classes - 1],
                  axis=1)),
          false_fn=lambda: coarse_mask_classes_t[:, 0, :, :])
      # pylint: enable=g-long-lambda
      coarse_mask_logits = tf.expand_dims(coarse_mask_logits, -1)
    else:
      coarse_mask_logits = coarse_mask_classes

    coarse_class_probs = tf.nn.sigmoid(coarse_mask_logits)
    class_probs = tf.cast(coarse_class_probs, prior_conditioned_features.dtype)

    return coarse_mask_classes, class_probs, prior_conditioned_features

  def coarsemask_decoder_net(self,
                             images,
                             is_training=None,
                             batch_norm_relu=nn_ops.BatchNormRelu):
    """Coarse mask decoder network architecture.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels]
    """
    for i in range(self._num_convs):
      images = tf.keras.layers.Conv2D(
          self._num_downsample_channels,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          activation=None,
          padding='same',
          name='coarse-class-%d' % i)(
              images)
      images = batch_norm_relu(name='coarse-class-%d-bn' % i)(
          images, is_training=is_training)

    return images


class ShapemaskFinemaskHead(object):
  """ShapemaskFinemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               num_convs,
               coarse_mask_thr,
               gt_upsample_scale,
               batch_norm_relu=nn_ops.BatchNormRelu):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      coarse_mask_thr: the threshold for suppressing noisy coarse prediction.
      gt_upsample_scale: scale for upsampling groundtruths.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
    """
    self._mask_num_classes = num_classes
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._num_convs = num_convs
    self._coarse_mask_thr = coarse_mask_thr
    self._gt_upsample_scale = gt_upsample_scale

    self._class_predict_conv = tf.keras.layers.Conv2D(
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0, stddev=0.01),
        padding='same',
        name='affinity-class-predict')
    self._upsample_conv = tf.keras.layers.Conv2DTranspose(
        self._num_downsample_channels // 2,
        (self._gt_upsample_scale, self._gt_upsample_scale),
        (self._gt_upsample_scale, self._gt_upsample_scale))
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
              name='fine-class-%d' % i))
      self._fine_class_bn.append(batch_norm_relu(name='fine-class-%d-bn' % i))

  def __call__(self, prior_conditioned_features, class_probs, is_training=None):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      prior_conditioned_features: a float Tensor of shape [batch_size *
        num_instances, mask_crop_size, mask_crop_size, num_downsample_channels].
        This is the instance feature crop.
      class_probs: a float Tensor of shape [batch_size * num_instances,
        mask_crop_size, mask_crop_size, 1]. This is the class probability of
        instance segmentation.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size * num_instances, mask_size, mask_size, num_classes].
    """
    with backend.get_graph().as_default(), tf.name_scope('affinity-masknet'):
      # Extract the foreground mean features
      point_samp_prob_thr = 1. / (1. + tf.exp(-self._coarse_mask_thr))
      point_samp_prob_thr = tf.cast(point_samp_prob_thr, class_probs.dtype)
      class_probs = tf.where(
          tf.greater(class_probs, point_samp_prob_thr), class_probs,
          tf.zeros_like(class_probs))
      weighted_features = class_probs * prior_conditioned_features
      sum_class_vector = tf.reduce_sum(
          input_tensor=class_probs, axis=(1, 2)) + tf.constant(
              1e-20, class_probs.dtype)
      instance_embedding = tf.reduce_sum(
          input_tensor=weighted_features, axis=(1, 2)) / sum_class_vector

      # Take the difference between crop features and mean instance features.
      instance_features = prior_conditioned_features - tf.reshape(
          instance_embedding, (-1, 1, 1, self._num_downsample_channels))

      # Decoder to generate upsampled segmentation mask.
      affinity_output_features = self.finemask_decoder_net(
          instance_features, is_training)

      # Predict per-class instance masks.
      affinity_mask_classes = self._class_predict_conv(affinity_output_features)

      return affinity_mask_classes

  def finemask_decoder_net(self, images, is_training=None):
    """Fine mask decoder network architecture.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.

    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels], where output size is self._gt_upsample_scale times
        that of input.
    """
    for i in range(self._num_convs):
      images = self._fine_class_conv[i](images)
      images = self._fine_class_bn[i](images, is_training=is_training)

    if self._gt_upsample_scale > 1:
      images = self._upsample_conv(images)

    return images
