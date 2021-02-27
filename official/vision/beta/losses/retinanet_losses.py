# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Losses used for detection models."""

# Import libraries
import tensorflow as tf


def focal_loss(logits, targets, alpha, gamma):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, d_1, ..., d_k, n_classes].
    targets: A float32 tensor of size
      [batch, d_1, ..., d_k, n_classes].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.

  Returns:
    loss: A float32 Tensor of size
      [batch, d_1, ..., d_k, n_classes] representing
      normalized loss on the prediction map.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)

  return weighted_loss


class FocalLoss(tf.keras.losses.Loss):
  """Implements a Focal loss for classification problems.

  Reference:
    [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  """

  def __init__(self,
               alpha,
               gamma,
               num_classes,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    """Initializes `FocalLoss`.

    Args:
      alpha: The `alpha` weight factor for binary class imbalance.
      gamma: The `gamma` focusing parameter to re-weight loss.
      num_classes: Number of foreground classes.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op. Defaults to 'retinanet_class_loss'.
    """
    self._num_classes = num_classes
    self._alpha = alpha
    self._gamma = gamma
    super(FocalLoss, self).__init__(reduction=reduction, name=name)

  def call(self, y_true, y_pred):
    """Invokes the `FocalLoss`.

    Args:
      y_true: Ordered Dict with level to [batch, height, width, num_anchors].
        for example,
        {3: tf.Tensor(shape=[32, 512, 512, 9], dtype=tf.float32),
         4: tf.Tensor([shape=32, 256, 256, 9, dtype=tf.float32])}
      y_pred: Ordered Dict with level to [batch, height, width, num_anchors *
        num_classes]. for example,
        {3: tf.Tensor(shape=[32, 512, 512, 9], dtype=tf.int64),
         4: tf.Tensor(shape=[32, 256, 256, 9 * 21], dtype=tf.int64)}

    Returns:
      Summed loss float `Tensor`.
    """
    flattened_cls_outputs = []
    flattened_labels = []
    batch_size = None
    for level in y_pred.keys():
      cls_output = y_pred[level]
      label = y_true[level]
      if batch_size is None:
        batch_size = cls_output.shape[0] or tf.shape(cls_output)[0]
      flattened_cls_outputs.append(
          tf.reshape(cls_output, [batch_size, -1, self._num_classes]))
      flattened_labels.append(tf.reshape(label, [batch_size, -1]))
    cls_outputs = tf.concat(flattened_cls_outputs, axis=1)
    labels = tf.concat(flattened_labels, axis=1)

    cls_targets_one_hot = tf.one_hot(labels, self._num_classes)
    return focal_loss(
        tf.cast(cls_outputs, dtype=tf.float32),
        tf.cast(cls_targets_one_hot, dtype=tf.float32), self._alpha,
        self._gamma)

  def get_config(self):
    config = {
        'alpha': self._alpha,
        'gamma': self._gamma,
        'num_classes': self._num_classes,
    }
    base_config = super(FocalLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class RetinanetBoxLoss(tf.keras.losses.Loss):
  """RetinaNet box Huber loss."""

  def __init__(self,
               delta,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    """Initializes `RetinanetBoxLoss`.

    Args:
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op. Defaults to 'retinanet_class_loss'.
    """
    self._huber_loss = tf.keras.losses.Huber(
        delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    self._delta = delta
    super(RetinanetBoxLoss, self).__init__(reduction=reduction, name=name)

  def call(self, y_true, y_pred):
    """Computes box detection loss.

    Computes total detection loss including box and class loss from all levels.

    Args:
      y_true: Ordered Dict with level to [batch, height, width,
        num_anchors * 4] for example,
        {3: tf.Tensor(shape=[32, 512, 512, 9 * 4], dtype=tf.float32),
         4: tf.Tensor([shape=32, 256, 256, 9 * 4, dtype=tf.float32])}
      y_pred: Ordered Dict with level to [batch, height, width,
        num_anchors * 4]. for example,
        {3: tf.Tensor(shape=[32, 512, 512, 9 * 4], dtype=tf.int64),
         4: tf.Tensor(shape=[32, 256, 256, 9 * 4], dtype=tf.int64)}

    Returns:
      an integer tensor representing total box regression loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training

    flattened_box_outputs = []
    flattened_labels = []
    batch_size = None
    for level in y_pred.keys():
      box_output = y_pred[level]
      label = y_true[level]
      if batch_size is None:
        batch_size = box_output.shape[0] or tf.shape(box_output)[0]
      flattened_box_outputs.append(tf.reshape(box_output, [batch_size, -1, 4]))
      flattened_labels.append(tf.reshape(label, [batch_size, -1, 4]))
    box_outputs = tf.concat(flattened_box_outputs, axis=1)
    labels = tf.concat(flattened_labels, axis=1)
    loss = self._huber_loss(labels, box_outputs)
    return loss

  def get_config(self):
    config = {
        'delta': self._delta,
    }
    base_config = super(RetinanetBoxLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
