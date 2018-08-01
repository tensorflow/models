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

"""Mask R-CNN Mask Head."""
import math
import tensorflow as tf

from object_detection.predictors.mask_rcnn_heads import mask_rcnn_head

slim = tf.contrib.slim


class MaskHead(mask_rcnn_head.MaskRCNNHead):
  """Mask RCNN mask prediction head."""

  def __init__(self,
               num_classes,
               conv_hyperparams_fn=None,
               mask_height=14,
               mask_width=14,
               mask_prediction_num_conv_layers=2,
               mask_prediction_conv_depth=256,
               masks_are_class_agnostic=False):
    """Constructor.

    Args:
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      mask_height: Desired output mask height. The default value is 14.
      mask_width: Desired output mask width. The default value is 14.
      mask_prediction_num_conv_layers: Number of convolution layers applied to
        the image_features in mask prediction branch.
      mask_prediction_conv_depth: The depth for the first conv2d_transpose op
        applied to the image_features in the mask prediction branch. If set
        to 0, the depth of the convolution layers will be automatically chosen
        based on the number of object classes and the number of channels in the
        image features.
      masks_are_class_agnostic: Boolean determining if the mask-head is
        class-agnostic or not.

    Raises:
      ValueError: conv_hyperparams_fn is None.
    """
    super(MaskHead, self).__init__()
    self._num_classes = num_classes
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._mask_prediction_num_conv_layers = mask_prediction_num_conv_layers
    self._mask_prediction_conv_depth = mask_prediction_conv_depth
    self._masks_are_class_agnostic = masks_are_class_agnostic
    if conv_hyperparams_fn is None:
      raise ValueError('conv_hyperparams_fn is None.')

  def _get_mask_predictor_conv_depth(self,
                                     num_feature_channels,
                                     num_classes,
                                     class_weight=3.0,
                                     feature_weight=2.0):
    """Computes the depth of the mask predictor convolutions.

    Computes the depth of the mask predictor convolutions given feature channels
    and number of classes by performing a weighted average of the two in
    log space to compute the number of convolution channels. The weights that
    are used for computing the weighted average do not need to sum to 1.

    Args:
      num_feature_channels: An integer containing the number of feature
        channels.
      num_classes: An integer containing the number of classes.
      class_weight: Class weight used in computing the weighted average.
      feature_weight: Feature weight used in computing the weighted average.

    Returns:
      An integer containing the number of convolution channels used by mask
        predictor.
    """
    num_feature_channels_log = math.log(float(num_feature_channels), 2.0)
    num_classes_log = math.log(float(num_classes), 2.0)
    weighted_num_feature_channels_log = (
        num_feature_channels_log * feature_weight)
    weighted_num_classes_log = num_classes_log * class_weight
    total_weight = feature_weight + class_weight
    num_conv_channels_log = round(
        (weighted_num_feature_channels_log + weighted_num_classes_log) /
        total_weight)
    return int(math.pow(2.0, num_conv_channels_log))

  def _predict(self, roi_pooled_features):
    """Performs mask prediction.

    Args:
      roi_pooled_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      instance_masks: A float tensor of shape
          [batch_size, 1, num_classes, mask_height, mask_width].
    """
    num_conv_channels = self._mask_prediction_conv_depth
    if num_conv_channels == 0:
      num_feature_channels = roi_pooled_features.get_shape().as_list()[3]
      num_conv_channels = self._get_mask_predictor_conv_depth(
          num_feature_channels, self._num_classes)
    with slim.arg_scope(self._conv_hyperparams_fn()):
      upsampled_features = tf.image.resize_bilinear(
          roi_pooled_features, [self._mask_height, self._mask_width],
          align_corners=True)
      for _ in range(self._mask_prediction_num_conv_layers - 1):
        upsampled_features = slim.conv2d(
            upsampled_features,
            num_outputs=num_conv_channels,
            kernel_size=[3, 3])
      num_masks = 1 if self._masks_are_class_agnostic else self._num_classes
      mask_predictions = slim.conv2d(
          upsampled_features,
          num_outputs=num_masks,
          activation_fn=None,
          kernel_size=[3, 3])
      return tf.expand_dims(
          tf.transpose(mask_predictions, perm=[0, 3, 1, 2]),
          axis=1,
          name='MaskPredictor')
