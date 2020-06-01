# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Convolutional Box Predictors with and without weight sharing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.core import box_predictor
from object_detection.utils import shape_utils
from object_detection.utils import static_shape

slim = contrib_slim

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(box_predictor.BoxPredictor):
  """Convolutional Box Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               num_layers_before_predictor,
               min_depth,
               max_depth):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes.
      class_prediction_head: The head that predicts the classes.
      other_heads: A dictionary mapping head names to convolutional
        head classes.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      min_depth: Minimum feature depth prior to predicting box encodings
        and class predictions.
      max_depth: Maximum feature depth prior to predicting box encodings
        and class predictions. If max_depth is set to 0, no additional
        feature map will be inserted before location and class predictions.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      A dictionary containing:
        box_encodings: A list of float tensors of shape
          [batch_size, num_anchors_i, q, code_size] representing the location of
          the objects, where q is 1 or the number of classes. Each entry in the
          list corresponds to a feature map in the input `image_features` list.
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
        (optional) Predictions from other heads.
    """
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]
    for (image_feature,
         num_predictions_per_location, box_predictor_scope) in zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes):
      net = image_feature
      with box_predictor_scope:
        with slim.arg_scope(self._conv_hyperparams_fn()):
          with slim.arg_scope([slim.dropout], is_training=self._is_training):
            # Add additional conv layers before the class predictor.
            features_depth = static_shape.get_depth(image_feature.get_shape())
            depth = max(min(features_depth, self._max_depth), self._min_depth)
            tf.logging.info('depth of additional conv before box predictor: {}'.
                            format(depth))
            if depth > 0 and self._num_layers_before_predictor > 0:
              for i in range(self._num_layers_before_predictor):
                net = slim.conv2d(
                    net,
                    depth, [1, 1],
                    reuse=tf.AUTO_REUSE,
                    scope='Conv2d_%d_1x1_%d' % (i, depth))
            sorted_keys = sorted(self._other_heads.keys())
            sorted_keys.append(BOX_ENCODINGS)
            sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
            for head_name in sorted_keys:
              if head_name == BOX_ENCODINGS:
                head_obj = self._box_prediction_head
              elif head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
                head_obj = self._class_prediction_head
              else:
                head_obj = self._other_heads[head_name]
              prediction = head_obj.predict(
                  features=net,
                  num_predictions_per_location=num_predictions_per_location)
              predictions[head_name].append(prediction)
    return predictions


# TODO(rathodv): Replace with slim.arg_scope_func_key once its available
# externally.
def _arg_scope_func_key(op):
  """Returns a key that can be used to index arg_scope dictionary."""
  return getattr(op, '_key_op', str(op))


# TODO(rathodv): Merge the implementation with ConvolutionalBoxPredictor above
# since they are very similar.
class WeightSharedConvolutionalBoxPredictor(box_predictor.BoxPredictor):
  """Convolutional Box Predictor with weight sharing.

  Defines the box predictor as defined in
  https://arxiv.org/abs/1708.02002. This class differs from
  ConvolutionalBoxPredictor in that it shares weights and biases while
  predicting from different feature maps. However, batch_norm parameters are not
  shared because the statistics of the activations vary among the different
  feature maps.

  Also note that separate multi-layer towers are constructed for the box
  encoding and class predictors respectively.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               depth,
               num_layers_before_predictor,
               kernel_size=3,
               apply_batch_norm=False,
               share_prediction_tower=False,
               use_depthwise=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes.
      class_prediction_head: The head that predicts the classes.
      other_heads: A dictionary mapping head names to convolutional
        head classes.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      kernel_size: Size of final convolution kernel.
      apply_batch_norm: Whether to apply batch normalization to conv layers in
        this predictor.
      share_prediction_tower: Whether to share the multi-layer tower among box
        prediction head, class prediction head and other heads.
      use_depthwise: Whether to use depthwise separable conv2d instead of
       regular conv2d.
    """
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(is_training,
                                                                num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm
    self._share_prediction_tower = share_prediction_tower
    self._use_depthwise = use_depthwise

  @property
  def num_classes(self):
    return self._num_classes

  def _insert_additional_projection_layer(self, image_feature,
                                          inserted_layer_counter,
                                          target_channel):
    if inserted_layer_counter < 0:
      return image_feature, inserted_layer_counter
    image_feature = slim.conv2d(
        image_feature,
        target_channel, [1, 1],
        stride=1,
        padding='SAME',
        activation_fn=None,
        normalizer_fn=(tf.identity if self._apply_batch_norm else None),
        scope='ProjectionLayer/conv2d_{}'.format(
            inserted_layer_counter))
    if self._apply_batch_norm:
      image_feature = slim.batch_norm(
          image_feature,
          scope='ProjectionLayer/conv2d_{}/BatchNorm'.format(
              inserted_layer_counter))
    inserted_layer_counter += 1
    return image_feature, inserted_layer_counter

  def _compute_base_tower(self, tower_name_scope, image_feature, feature_index):
    net = image_feature
    for i in range(self._num_layers_before_predictor):
      if self._use_depthwise:
        conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
      else:
        conv_op = slim.conv2d
      net = conv_op(
          net,
          self._depth, [self._kernel_size, self._kernel_size],
          stride=1,
          padding='SAME',
          activation_fn=None,
          normalizer_fn=(tf.identity if self._apply_batch_norm else None),
          scope='{}/conv2d_{}'.format(tower_name_scope, i))
      if self._apply_batch_norm:
        net = slim.batch_norm(
            net,
            scope='{}/conv2d_{}/BatchNorm/feature_{}'.
            format(tower_name_scope, i, feature_index))
      net = tf.nn.relu6(net)
    return net

  def _predict_head(self, head_name, head_obj, image_feature, box_tower_feature,
                    feature_index, num_predictions_per_location):
    if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
      tower_name_scope = 'ClassPredictionTower'
    else:
      tower_name_scope = head_name + 'PredictionTower'
    if self._share_prediction_tower:
      head_tower_feature = box_tower_feature
    else:
      head_tower_feature = self._compute_base_tower(
          tower_name_scope=tower_name_scope,
          image_feature=image_feature,
          feature_index=feature_index)
    return head_obj.predict(
        features=head_tower_feature,
        num_predictions_per_location=num_predictions_per_location)

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels] containing features for a batch of images. Note that
        when not all tensors in the list have the same number of channels, an
        additional projection layer will be added on top the tensor to generate
        feature map with number of channels consitent with the majority.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map. Note that all values must be the same since the weights are
        shared.

    Returns:
      A dictionary containing:
        box_encodings: A list of float tensors of shape
          [batch_size, num_anchors_i, code_size] representing the location of
          the objects. Each entry in the list corresponds to a feature map in
          the input `image_features` list.
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
        (optional) Predictions from other heads.
          E.g., mask_predictions: A list of float tensors of shape
          [batch_size, num_anchord_i, num_classes, mask_height, mask_width].


    Raises:
      ValueError: If the num predictions per locations differs between the
        feature maps.
    """
    if len(set(num_predictions_per_location_list)) > 1:
      raise ValueError('num predictions per location must be same for all'
                       'feature maps, found: {}'.format(
                           num_predictions_per_location_list))
    feature_channels = [
        shape_utils.get_dim_as_int(image_feature.shape[3])
        for image_feature in image_features
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, appending additional projection '
                      'layers to bring all feature maps to uniformly have {} '
                      'channels.'.format(feature_channels, target_channel))
    else:
      # Place holder variables if has_different_feature_channels is False.
      target_channel = -1
      inserted_layer_counter = -1
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    for feature_index, (image_feature,
                        num_predictions_per_location) in enumerate(
                            zip(image_features,
                                num_predictions_per_location_list)):
      with tf.variable_scope('WeightSharedConvolutionalBoxPredictor',
                             reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._conv_hyperparams_fn()):
          # TODO(wangjiang) Pass is_training to the head class directly.
          with slim.arg_scope([slim.dropout], is_training=self._is_training):
            (image_feature,
             inserted_layer_counter) = self._insert_additional_projection_layer(
                 image_feature, inserted_layer_counter, target_channel)
            if self._share_prediction_tower:
              box_tower_scope = 'PredictionTower'
            else:
              box_tower_scope = 'BoxPredictionTower'
            box_tower_feature = self._compute_base_tower(
                tower_name_scope=box_tower_scope,
                image_feature=image_feature,
                feature_index=feature_index)
            box_encodings = self._box_prediction_head.predict(
                features=box_tower_feature,
                num_predictions_per_location=num_predictions_per_location)
            predictions[BOX_ENCODINGS].append(box_encodings)
            sorted_keys = sorted(self._other_heads.keys())
            sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
            for head_name in sorted_keys:
              if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
                head_obj = self._class_prediction_head
              else:
                head_obj = self._other_heads[head_name]
              prediction = self._predict_head(
                  head_name=head_name,
                  head_obj=head_obj,
                  image_feature=image_feature,
                  box_tower_feature=box_tower_feature,
                  feature_index=feature_index,
                  num_predictions_per_location=num_predictions_per_location)
              predictions[head_name].append(prediction)
    return predictions


