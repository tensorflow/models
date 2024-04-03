# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Losses used for detection models."""

import tensorflow as tf 
import keras


class FocalLoss(keras.losses.Loss):
  """Implements a Focal loss for classification problems.

  Reference:
    [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  """

  def __init__(self,
               alpha,
               gamma,
               reduction="sum_over_batch_size",
               name=None):
    """Initializes `FocalLoss`.

    Args:
      alpha: The `alpha` weight factor for binary class imbalance.
      gamma: The `gamma` focusing parameter to re-weight loss.
      reduction: (Optional) 
        However, loss class instances feature a reduction constructor argument, 
        which defaults to "sum_over_batch_size" (i.e. average). 
        Allowable values are "sum_over_batch_size", "sum", and "none":
          "sum_over_batch_size" means the loss instance will return the 
            average of the per-sample losses in the batch.
          "sum" means the loss instance will return the sum of the per-sample losses in the batch.
          "none" means the loss instance will return the full array of per-sample losses.
      name: Optional name for the op. Defaults to 'retinanet_class_loss'.
    """
    self._alpha = alpha
    self._gamma = gamma
    super(FocalLoss, self).__init__(reduction=reduction, name=name)

  def call(self, y_true, y_pred):
    """Invokes the `FocalLoss`.

    Args:
      y_true: A tensor of size [batch, num_anchors, num_classes]
      y_pred: A tensor of size [batch, num_anchors, num_classes]

    Returns:
      Summed loss float `Tensor`.
    """
    with tf.name_scope('focal_loss'):
      y_true = tf.cast(y_true, dtype=tf.float32)
      y_pred = tf.cast(y_pred, dtype=tf.float32)
      positive_label_mask = tf.equal(y_true, 1.0)
      cross_entropy = (
          tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
      probs = tf.sigmoid(y_pred)
      probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
      # With small gamma, the implementation could produce NaN during back prop.
      modulator = tf.pow(1.0 - probs_gt, self._gamma)
      loss = modulator * cross_entropy
      weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                               (1.0 - self._alpha) * loss)

    return weighted_loss

  def get_config(self):
    config = {
        'alpha': self._alpha,
        'gamma': self._gamma,
    }
    base_config = super(FocalLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
