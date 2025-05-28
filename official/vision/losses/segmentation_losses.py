# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.dataloaders import utils

EPSILON = 1e-5


class SegmentationLoss:
  """Semantic segmentation loss."""

  def __init__(self,
               label_smoothing,
               class_weights,
               ignore_label,
               use_groundtruth_dimension,
               use_binary_cross_entropy=False,
               top_k_percent_pixels=1.0,
               gt_is_matting_map=False):
    """Initializes `SegmentationLoss`.

    Args:
      label_smoothing: A float, if > 0., smooth out one-hot probability by
        spreading the amount of probability to all other label classes.
      class_weights: A float list containing the weight of each class.
      ignore_label: An integer specifying the ignore label.
      use_groundtruth_dimension: A boolean, whether to resize the output to
        match the dimension of the ground truth.
      use_binary_cross_entropy: A boolean, if true, use binary cross entropy
        loss, otherwise, use categorical cross entropy.
      top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its
        value < 1., only compute the loss for the top k percent pixels. This is
        useful for hard pixel mining.
      gt_is_matting_map: If or not the groundtruth mask is a matting map. Note
        that the matting map is only supported for 2 class segmentation.
    """
    self._label_smoothing = label_smoothing
    self._class_weights = class_weights
    self._ignore_label = ignore_label
    self._use_groundtruth_dimension = use_groundtruth_dimension
    self._use_binary_cross_entropy = use_binary_cross_entropy
    self._top_k_percent_pixels = top_k_percent_pixels
    self._gt_is_matting_map = gt_is_matting_map

  def __call__(self, logits, labels, **kwargs):
    """Computes `SegmentationLoss`.

    Args:
      logits: A float tensor in shape (batch_size, height, width, num_classes)
        which is the output of the network.
      labels: A tensor in shape (batch_size, height, width, num_layers), which
        is the label masks of the ground truth. The num_layers can be > 1 if the
        pixels are labeled as multiple classes.
      **kwargs: additional keyword arguments.

    Returns:
       A 0-D float which stores the overall loss of the batch.
    """
    _, height, width, num_classes = logits.get_shape().as_list()
    output_dtype = logits.dtype
    num_layers = labels.get_shape().as_list()[-1]
    if not self._use_binary_cross_entropy:
      if num_layers > 1:
        raise ValueError(
            'Groundtruth mask must have only 1 layer if using categorical'
            'cross entropy, but got {} layers.'.format(num_layers))
    if self._gt_is_matting_map:
      if num_classes != 2:
        raise ValueError(
            'Groundtruth matting map only supports 2 classes, but got {} '
            'classes.'.format(num_classes))
      if num_layers > 1:
        raise ValueError(
            'Groundtruth matting map must have only 1 layer, but got {} '
            'layers.'.format(num_layers))

    class_weights = (
        self._class_weights if self._class_weights else [1] * num_classes)
    if num_classes != len(class_weights):
      raise ValueError(
          'Length of class_weights should be {}'.format(num_classes))
    class_weights = tf.constant(class_weights, dtype=output_dtype)

    if not self._gt_is_matting_map:
      labels = tf.cast(labels, tf.int32)
    if self._use_groundtruth_dimension:
      # TODO(arashwan): Test using align corners to match deeplab alignment.
      logits = tf.image.resize(
          logits, tf.shape(labels)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    else:
      labels = tf.image.resize(
          labels, (height, width),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    valid_mask = tf.not_equal(tf.cast(labels, tf.int32), self._ignore_label)

    # (batch_size, height, width, num_classes)
    labels_with_prob = self.get_labels_with_prob(logits, labels, valid_mask,
                                                 **kwargs)

    # (batch_size, height, width)
    valid_mask = tf.cast(tf.reduce_any(valid_mask, axis=-1), dtype=output_dtype)

    if self._use_binary_cross_entropy:
      # (batch_size, height, width, num_classes)
      cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_with_prob, logits=logits)
      # (batch_size, height, width, num_classes)
      cross_entropy_loss *= class_weights
      num_valid_values = tf.reduce_sum(valid_mask) * tf.cast(
          num_classes, output_dtype)
      # (batch_size, height, width, num_classes)
      cross_entropy_loss *= valid_mask[..., tf.newaxis]
    else:
      # (batch_size, height, width)
      cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels_with_prob, logits=logits)

      # If groundtruth is matting map, binarize the value to create the weight
      # mask
      if self._gt_is_matting_map:
        labels = utils.binarize_matting_map(labels)

      # (batch_size, height, width)
      weight_mask = tf.einsum(
          '...y,y->...',
          tf.one_hot(
              tf.cast(tf.squeeze(labels, axis=-1), tf.int32),
              depth=num_classes,
              dtype=output_dtype), class_weights)
      cross_entropy_loss *= weight_mask
      num_valid_values = tf.reduce_sum(valid_mask)
      cross_entropy_loss *= valid_mask

    if self._top_k_percent_pixels < 1.0:
      return self.aggregate_loss_top_k(cross_entropy_loss, num_valid_values)
    else:
      return tf.reduce_sum(cross_entropy_loss) / (num_valid_values + EPSILON)

  def get_labels_with_prob(self, logits, labels, valid_mask, **unused_kwargs):
    """Get a tensor representing the probability of each class for each pixel.

    This method can be overridden in subclasses for customizing loss function.

    Args:
      logits: A float tensor in shape (batch_size, height, width, num_classes)
        which is the output of the network.
      labels: A tensor in shape (batch_size, height, width, num_layers), which
        is the label masks of the ground truth. The num_layers can be > 1 if the
        pixels are labeled as multiple classes.
      valid_mask: A bool tensor in shape (batch_size, height, width, num_layers)
        which indicates the ignored labels in each ground truth layer.
      **unused_kwargs: Unused keyword arguments.

    Returns:
       A float tensor in shape (batch_size, height, width, num_classes).
    """
    num_classes = logits.get_shape().as_list()[-1]

    if self._gt_is_matting_map:
      # (batch_size, height, width, num_classes=2)
      train_labels = tf.concat([1 - labels, labels], axis=-1)
    else:
      labels = tf.cast(labels, tf.int32)
      # Assign pixel with ignore label to class -1, which will be ignored by
      # tf.one_hot operation.
      # (batch_size, height, width, num_masks)
      labels = tf.where(valid_mask, labels, -tf.ones_like(labels))

      if self._use_binary_cross_entropy:
        # (batch_size, height, width, num_masks, num_classes)
        one_hot_labels_per_mask = tf.one_hot(
            labels,
            depth=num_classes,
            on_value=True,
            off_value=False,
            dtype=tf.bool,
            axis=-1)
        # Aggregate all one-hot labels to get a binary mask in shape
        # (batch_size, height, width, num_classes), which represents all the
        # classes that a pixel is labeled as.
        # For example, if a pixel is labeled as "window" (id=1) and also being a
        # part of the "building" (id=3), then its train_labels are [0,1,0,1].
        train_labels = tf.cast(
            tf.reduce_any(one_hot_labels_per_mask, axis=-2), dtype=logits.dtype)
      else:
        # (batch_size, height, width, num_classes)
        train_labels = tf.one_hot(
            tf.squeeze(labels, axis=-1), depth=num_classes, dtype=logits.dtype)

    return train_labels * (
        1 - self._label_smoothing) + self._label_smoothing / num_classes

  def aggregate_loss_top_k(self, pixelwise_loss, num_valid_pixels=None):
    """Aggregate the top-k greatest pixelwise loss.

    Args:
      pixelwise_loss: a float tensor in shape (batch_size, height, width) which
        stores the loss of each pixel.
      num_valid_pixels: the number of pixels which are not ignored. If None, all
        the pixels are valid.

    Returns:
       A 0-D float which stores the overall loss of the batch.
    """
    pixelwise_loss = tf.reshape(pixelwise_loss, shape=[-1])
    top_k_pixels = tf.cast(
        self._top_k_percent_pixels
        * tf.cast(tf.size(pixelwise_loss), tf.float32),
        tf.int32,
    )
    top_k_losses, _ = tf.math.top_k(pixelwise_loss, k=top_k_pixels)
    normalizer = tf.cast(top_k_pixels, top_k_losses.dtype)
    if num_valid_pixels is not None:
      normalizer = tf.minimum(normalizer,
                              tf.cast(num_valid_pixels, top_k_losses.dtype))
    return tf.reduce_sum(top_k_losses) / (normalizer + EPSILON)


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
    self._mse_loss = tf_keras.losses.MeanSquaredError(
        reduction=tf_keras.losses.Reduction.NONE)

  def __call__(self, predicted_scores, logits, labels):
    actual_scores = get_actual_mask_scores(logits, labels, self._ignore_label)
    loss = tf_utils.safe_mean(self._mse_loss(actual_scores, predicted_scores))
    return loss
