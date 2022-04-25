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

"""Losses used for segmentation models."""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils

EPSILON = 1e-5


class SegmentationLoss:
  """Semantic segmentation loss."""

  def __init__(self, label_smoothing, class_weights, ignore_label,
               use_groundtruth_dimension, top_k_percent_pixels=1.0):
    self._top_k_percent_pixels = top_k_percent_pixels
    self._class_weights = class_weights
    self._ignore_label = ignore_label
    self._use_groundtruth_dimension = use_groundtruth_dimension
    self._label_smoothing = label_smoothing

  def __call__(self, logits, labels, **kwargs):
    _, height, width, num_classes = logits.get_shape().as_list()

    if self._use_groundtruth_dimension:
      # TODO(arashwan): Test using align corners to match deeplab alignment.
      logits = tf.image.resize(
          logits, tf.shape(labels)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    else:
      labels = tf.image.resize(
          labels, (height, width),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    valid_mask = tf.not_equal(labels, self._ignore_label)
    normalizer = tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + EPSILON
    # Assign pixel with ignore label to class 0 (background). The loss on the
    # pixel will later be masked out.
    labels = tf.where(valid_mask, labels, tf.zeros_like(labels))

    labels = tf.squeeze(tf.cast(labels, tf.int32), axis=3)
    valid_mask = tf.squeeze(tf.cast(valid_mask, tf.float32), axis=3)
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=self.get_labels_with_prob(labels, logits, **kwargs),
        logits=logits)

    if not self._class_weights:
      class_weights = [1] * num_classes
    else:
      class_weights = self._class_weights

    if num_classes != len(class_weights):
      raise ValueError(
          'Length of class_weights should be {}'.format(num_classes))

    weight_mask = tf.einsum('...y,y->...',
                            tf.one_hot(labels, num_classes, dtype=tf.float32),
                            tf.constant(class_weights, tf.float32))
    valid_mask *= weight_mask
    cross_entropy_loss *= tf.cast(valid_mask, tf.float32)

    if self._top_k_percent_pixels >= 1.0:
      loss = tf.reduce_sum(cross_entropy_loss) / normalizer
    else:
      cross_entropy_loss = tf.reshape(cross_entropy_loss, shape=[-1])
      top_k_pixels = tf.cast(
          self._top_k_percent_pixels *
          tf.cast(tf.size(cross_entropy_loss), tf.float32), tf.int32)
      top_k_losses, _ = tf.math.top_k(
          cross_entropy_loss, k=top_k_pixels, sorted=True)
      normalizer = tf.reduce_sum(
          tf.cast(tf.not_equal(top_k_losses, 0.0), tf.float32)) + EPSILON
      loss = tf.reduce_sum(top_k_losses) / normalizer

    return loss

  def get_labels_with_prob(self, labels, logits, **unused_kwargs):
    """Get a tensor representing the probability of each class for each pixel.

    This method can be overridden in subclasses for customizing loss function.

    Args:
      labels: A float tensor in shape (batch_size, height, width), which is the
        label map of the ground truth.
      logits: A float tensor in shape (batch_size, height, width, num_classes)
        which is the output of the network.
      **unused_kwargs: Unused keyword arguments.

    Returns:
       A float tensor in shape (batch_size, height, width, num_classes).
    """
    num_classes = logits.get_shape().as_list()[-1]
    onehot_labels = tf.one_hot(labels, num_classes)
    return onehot_labels * (
        1 - self._label_smoothing) + self._label_smoothing / num_classes


def get_actual_mask_scores(logits, labels, ignore_label):
  """Gets actual mask scores."""
  _, height, width, num_classes = logits.get_shape().as_list()
  batch_size = tf.shape(logits)[0]
  logits = tf.stop_gradient(logits)
  labels = tf.image.resize(
      labels, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  predicted_labels = tf.argmax(logits, -1, output_type=tf.int32)
  flat_predictions = tf.reshape(predicted_labels, [batch_size, -1])
  flat_labels = tf.cast(tf.reshape(labels, [batch_size, -1]), tf.int32)

  one_hot_predictions = tf.one_hot(
      flat_predictions, num_classes, on_value=True, off_value=False)
  one_hot_labels = tf.one_hot(
      flat_labels, num_classes, on_value=True, off_value=False)
  keep_mask = tf.not_equal(flat_labels, ignore_label)
  keep_mask = tf.expand_dims(keep_mask, 2)

  overlap = tf.logical_and(one_hot_predictions, one_hot_labels)
  overlap = tf.logical_and(overlap, keep_mask)
  overlap = tf.reduce_sum(tf.cast(overlap, tf.float32), axis=1)
  union = tf.logical_or(one_hot_predictions, one_hot_labels)
  union = tf.logical_and(union, keep_mask)
  union = tf.reduce_sum(tf.cast(union, tf.float32), axis=1)
  actual_scores = tf.divide(overlap, tf.maximum(union, EPSILON))
  return actual_scores


class MaskScoringLoss:
  """Mask Scoring loss."""

  def __init__(self, ignore_label):
    self._ignore_label = ignore_label
    self._mse_loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)

  def __call__(self, predicted_scores, logits, labels):
    actual_scores = get_actual_mask_scores(logits, labels, self._ignore_label)
    loss = tf_utils.safe_mean(self._mse_loss(actual_scores, predicted_scores))
    return loss
