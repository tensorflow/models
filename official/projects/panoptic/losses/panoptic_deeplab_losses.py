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

"""Losses used for panoptic deeplab model."""

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.panoptic.ops import mask_ops

EPSILON = 1e-5


class WeightedBootstrappedCrossEntropyLoss:
  """Weighted semantic segmentation loss."""

  def __init__(self, label_smoothing, class_weights, ignore_label,
               top_k_percent_pixels=1.0):
    self._top_k_percent_pixels = top_k_percent_pixels
    self._class_weights = class_weights
    self._ignore_label = ignore_label
    self._label_smoothing = label_smoothing

  def __call__(self, logits, labels, sample_weight=None):
    _, _, _, num_classes = logits.get_shape().as_list()

    logits = tf.image.resize(
        logits, tf.shape(labels)[1:3],
        method=tf.image.ResizeMethod.BILINEAR)

    valid_mask = tf.not_equal(labels, self._ignore_label)
    normalizer = tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + EPSILON
    # Assign pixel with ignore label to class 0 (background). The loss on the
    # pixel will later be masked out.
    labels = tf.where(valid_mask, labels, tf.zeros_like(labels))

    labels = tf.squeeze(tf.cast(labels, tf.int32), axis=3)
    valid_mask = tf.squeeze(tf.cast(valid_mask, tf.float32), axis=3)
    onehot_labels = tf.one_hot(labels, num_classes)
    onehot_labels = onehot_labels * (
        1 - self._label_smoothing) + self._label_smoothing / num_classes
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=onehot_labels, logits=logits)

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

    if sample_weight is not None:
      valid_mask *= sample_weight

    cross_entropy_loss *= tf.cast(valid_mask, tf.float32)

    if self._top_k_percent_pixels >= 1.0:
      loss = tf.reduce_sum(cross_entropy_loss) / normalizer
    else:
      loss = self._compute_top_k_loss(cross_entropy_loss)
    return loss

  def _compute_top_k_loss(self, loss):
    """Computs top k loss."""
    batch_size = tf.shape(loss)[0]
    loss = tf.reshape(loss, shape=[batch_size, -1])

    top_k_pixels = tf.cast(
        self._top_k_percent_pixels *
        tf.cast(tf.shape(loss)[-1], dtype=tf.float32),
        dtype=tf.int32)

    # shape: [batch_size, top_k_pixels]
    per_sample_top_k_loss = tf.map_fn(
        fn=lambda x: tf.nn.top_k(x, k=top_k_pixels, sorted=False)[0],
        elems=loss,
        parallel_iterations=32,
        fn_output_signature=tf.float32)

    # shape: [batch_size]
    per_sample_normalizer = tf.reduce_sum(
        tf.cast(
            tf.not_equal(per_sample_top_k_loss, 0.0),
            dtype=tf.float32),
        axis=-1) + EPSILON
    per_sample_normalized_loss = tf.reduce_sum(
        per_sample_top_k_loss, axis=-1) / per_sample_normalizer

    normalized_loss = tf_utils.safe_mean(per_sample_normalized_loss)
    return normalized_loss


class CenterHeatmapLoss:
  """Center heatmap loss."""

  def __init__(self):
    self._loss_fn = tf.losses.mean_squared_error

  def __call__(self, logits, labels, sample_weight=None):
    _, height, width, _ = labels.get_shape().as_list()
    logits = tf.image.resize(
        logits,
        size=[height, width],
        method=tf.image.ResizeMethod.BILINEAR)

    loss = self._loss_fn(y_true=labels, y_pred=logits)

    if sample_weight is not None:
      loss *= sample_weight

    return tf_utils.safe_mean(loss)


class CenterOffsetLoss:
  """Center offset loss."""

  def __init__(self):
    self._loss_fn = tf.losses.mean_absolute_error

  def __call__(self, logits, labels, sample_weight=None):
    _, height, width, _ = labels.get_shape().as_list()
    logits = mask_ops.resize_and_rescale_offsets(
        logits, target_size=[height, width])

    loss = self._loss_fn(y_true=labels, y_pred=logits)

    if sample_weight is not None:
      loss *= sample_weight

    return tf_utils.safe_mean(loss)
