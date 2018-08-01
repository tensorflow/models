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

"""Mask R-CNN Class Head."""
import tensorflow as tf

from object_detection.predictors.mask_rcnn_heads import mask_rcnn_head

slim = tf.contrib.slim


class ClassHead(mask_rcnn_head.MaskRCNNHead):
  """Mask RCNN class prediction head."""

  def __init__(self, is_training, num_classes, fc_hyperparams_fn,
               use_dropout, dropout_keep_prob):
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
    """
    super(ClassHead, self).__init__()
    self._is_training = is_training
    self._num_classes = num_classes
    self._fc_hyperparams_fn = fc_hyperparams_fn
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob

  def _predict(self, roi_pooled_features):
    """Predicts boxes and class scores.

    Args:
      roi_pooled_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      class_predictions_with_background: A float tensor of shape
        [batch_size, 1, num_classes + 1] representing the class predictions for
        the proposals.
    """
    spatial_averaged_roi_pooled_features = tf.reduce_mean(
        roi_pooled_features, [1, 2], keep_dims=True, name='AvgPool')
    flattened_roi_pooled_features = slim.flatten(
        spatial_averaged_roi_pooled_features)
    if self._use_dropout:
      flattened_roi_pooled_features = slim.dropout(
          flattened_roi_pooled_features,
          keep_prob=self._dropout_keep_prob,
          is_training=self._is_training)

    with slim.arg_scope(self._fc_hyperparams_fn()):
      class_predictions_with_background = slim.fully_connected(
          flattened_roi_pooled_features,
          self._num_classes + 1,
          activation_fn=None,
          scope='ClassPredictor')
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [-1, 1, self._num_classes + 1])
    return class_predictions_with_background
