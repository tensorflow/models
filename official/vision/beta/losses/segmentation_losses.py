# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Losses used for segmentation models."""

# Import libraries
import tensorflow as tf

EPSILON = 1e-5


class SegmentationLoss:
  """Semantic segmentation loss."""

  def __init__(self, label_smoothing, class_weights,
               ignore_label, use_groundtruth_dimension):
    self._class_weights = class_weights
    self._ignore_label = ignore_label
    self._use_groundtruth_dimension = use_groundtruth_dimension
    self._label_smoothing = label_smoothing

  def __call__(self, logits, labels):
    _, height, width, num_classes = logits.get_shape().as_list()

    if self._use_groundtruth_dimension:
      # TODO(arashwan): Test using align corners to match deeplab alignment.
      logits = tf.image.resize(
          logits, tf.shape(labels)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
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
    cross_entropy_loss *= tf.cast(valid_mask, tf.float32)
    loss = tf.reduce_sum(cross_entropy_loss) / normalizer
    return loss
