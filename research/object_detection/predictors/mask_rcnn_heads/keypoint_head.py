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

"""Mask R-CNN Keypoint Head."""
import tensorflow as tf

from object_detection.predictors.mask_rcnn_heads import mask_rcnn_head
slim = tf.contrib.slim


class KeypointHead(mask_rcnn_head.MaskRCNNHead):
  """Mask RCNN keypoint prediction head."""

  def __init__(self,
               num_keypoints=17,
               conv_hyperparams_fn=None,
               keypoint_heatmap_height=56,
               keypoint_heatmap_width=56,
               keypoint_prediction_num_conv_layers=8,
               keypoint_prediction_conv_depth=512):
    """Constructor.

    Args:
      num_keypoints: (int scalar) number of keypoints.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      keypoint_heatmap_height: Desired output mask height. The default value
        is 14.
      keypoint_heatmap_width: Desired output mask width. The default value
        is 14.
      keypoint_prediction_num_conv_layers: Number of convolution layers applied
        to the image_features in mask prediction branch.
      keypoint_prediction_conv_depth: The depth for the first conv2d_transpose
        op applied to the image_features in the mask prediction branch. If set
        to 0, the depth of the convolution layers will be automatically chosen
        based on the number of object classes and the number of channels in the
        image features.
    """
    super(KeypointHead, self).__init__()
    self._num_keypoints = num_keypoints
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._keypoint_heatmap_height = keypoint_heatmap_height
    self._keypoint_heatmap_width = keypoint_heatmap_width
    self._keypoint_prediction_num_conv_layers = (
        keypoint_prediction_num_conv_layers)
    self._keypoint_prediction_conv_depth = keypoint_prediction_conv_depth

  def _predict(self, roi_pooled_features):
    """Performs keypoint prediction.

    Args:
      roi_pooled_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      instance_masks: A float tensor of shape
          [batch_size, 1, num_keypoints, heatmap_height, heatmap_width].
    """
    with slim.arg_scope(self._conv_hyperparams_fn()):
      net = slim.conv2d(
          roi_pooled_features,
          self._keypoint_prediction_conv_depth, [3, 3],
          scope='conv_1')
      for i in range(1, self._keypoint_prediction_num_conv_layers):
        net = slim.conv2d(
            net,
            self._keypoint_prediction_conv_depth, [3, 3],
            scope='conv_%d' % (i + 1))
      net = slim.conv2d_transpose(
          net, self._num_keypoints, [2, 2], scope='deconv1')
      heatmaps_mask = tf.image.resize_bilinear(
          net, [self._keypoint_heatmap_height, self._keypoint_heatmap_width],
          align_corners=True,
          name='upsample')
      return tf.expand_dims(
          tf.transpose(heatmaps_mask, perm=[0, 3, 1, 2]),
          axis=1,
          name='KeypointPredictor')
