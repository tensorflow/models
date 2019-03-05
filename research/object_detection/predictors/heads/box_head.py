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

"""Box Head.

Contains Box prediction head classes for different meta architectures.
All the box prediction heads have a predict function that receives the
`features` as the first argument and returns `box_encodings`.
"""
import functools
import tensorflow as tf

from object_detection.predictors.heads import head

slim = tf.contrib.slim


class MaskRCNNBoxHead(head.Head):
  """Box prediction head.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

  def __init__(self,
               is_training,
               num_classes,
               fc_hyperparams_fn,
               use_dropout,
               dropout_keep_prob,
               box_code_size,
               share_box_across_classes=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for fully connected ops.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      box_code_size: Size of encoding for each box.
      share_box_across_classes: Whether to share boxes across classes rather
        than use a different box for each class.
    """
    super(MaskRCNNBoxHead, self).__init__()
    self._is_training = is_training
    self._num_classes = num_classes
    self._fc_hyperparams_fn = fc_hyperparams_fn
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._box_code_size = box_code_size
    self._share_box_across_classes = share_box_across_classes

  def predict(self, features, num_predictions_per_location=1):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: Int containing number of predictions per
        location.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, 1, num_classes, code_size] representing the location of the
        objects.

    Raises:
      ValueError: If num_predictions_per_location is not 1.
    """
    if num_predictions_per_location != 1:
      raise ValueError('Only num_predictions_per_location=1 is supported')
    spatial_averaged_roi_pooled_features = tf.reduce_mean(
        features, [1, 2], keep_dims=True, name='AvgPool')
    flattened_roi_pooled_features = slim.flatten(
        spatial_averaged_roi_pooled_features)
    if self._use_dropout:
      flattened_roi_pooled_features = slim.dropout(
          flattened_roi_pooled_features,
          keep_prob=self._dropout_keep_prob,
          is_training=self._is_training)
    number_of_boxes = 1
    if not self._share_box_across_classes:
      number_of_boxes = self._num_classes

    with slim.arg_scope(self._fc_hyperparams_fn()):
      box_encodings = slim.fully_connected(
          flattened_roi_pooled_features,
          number_of_boxes * self._box_code_size,
          activation_fn=None,
          scope='BoxEncodingPredictor')
    box_encodings = tf.reshape(box_encodings,
                               [-1, 1, number_of_boxes, self._box_code_size])
    return box_encodings


class ConvolutionalBoxHead(head.Head):
  """Convolutional box prediction head."""

  def __init__(self,
               is_training,
               box_code_size,
               kernel_size,
               use_depthwise=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxHead, self).__init__()
    self._is_training = is_training
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.

    Returns:
      box_encodings: A float tensors of shape
        [batch_size, num_anchors, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes.
    """
    net = features
    if self._use_depthwise:
      box_encodings = slim.separable_conv2d(
          net, None, [self._kernel_size, self._kernel_size],
          padding='SAME', depth_multiplier=1, stride=1,
          rate=1, scope='BoxEncodingPredictor_depthwise')
      box_encodings = slim.conv2d(
          box_encodings,
          num_predictions_per_location * self._box_code_size, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='BoxEncodingPredictor')
    else:
      box_encodings = slim.conv2d(
          net, num_predictions_per_location * self._box_code_size,
          [self._kernel_size, self._kernel_size],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='BoxEncodingPredictor')
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    box_encodings = tf.reshape(box_encodings,
                               [batch_size, -1, 1, self._box_code_size])
    return box_encodings


# TODO(alirezafathi): See if possible to unify Weight Shared with regular
# convolutional box head.
class WeightSharedConvolutionalBoxHead(head.Head):
  """Weight shared convolutional box prediction head.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

  def __init__(self,
               box_code_size,
               kernel_size=3,
               use_depthwise=False,
               box_encodings_clip_range=None):
    """Constructor.

    Args:
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.
      use_depthwise: Whether to use depthwise convolutions for prediction steps.
        Default is False.
      box_encodings_clip_range: Min and max values for clipping box_encodings.
    """
    super(WeightSharedConvolutionalBoxHead, self).__init__()
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise
    self._box_encodings_clip_range = box_encodings_clip_range

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, num_anchors, code_size] representing the location of
        the objects.
    """
    box_encodings_net = features
    if self._use_depthwise:
      conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
    else:
      conv_op = slim.conv2d
    box_encodings = conv_op(
        box_encodings_net,
        num_predictions_per_location * self._box_code_size,
        [self._kernel_size, self._kernel_size],
        activation_fn=None, stride=1, padding='SAME',
        normalizer_fn=None,
        scope='BoxPredictor')
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    # Clipping the box encodings to make the inference graph TPU friendly.
    if self._box_encodings_clip_range is not None:
      box_encodings = tf.clip_by_value(
          box_encodings, self._box_encodings_clip_range.min,
          self._box_encodings_clip_range.max)
    box_encodings = tf.reshape(box_encodings,
                               [batch_size, -1, self._box_code_size])
    return box_encodings
