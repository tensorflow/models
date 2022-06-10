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

"""Losses used for BASNet models."""
import tensorflow as tf

EPSILON = 1e-5


class BASNetLoss:
  """BASNet hybrid loss."""

  def __init__(self):
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=False)
    self._ssim = tf.image.ssim

  def __call__(self, sigmoids, labels):
    levels = sorted(sigmoids.keys())

    labels_bce = tf.squeeze(labels, axis=-1)
    labels = tf.cast(labels, tf.float32)

    bce_losses = []
    ssim_losses = []
    iou_losses = []

    for level in levels:
      bce_losses.append(
          self._binary_crossentropy(labels_bce, sigmoids[level]))
      ssim_losses.append(
          1 - self._ssim(sigmoids[level], labels, max_val=1.0))
      iou_losses.append(
          self._iou_loss(sigmoids[level], labels))

    total_bce_loss = tf.math.add_n(bce_losses)
    total_ssim_loss = tf.math.add_n(ssim_losses)
    total_iou_loss = tf.math.add_n(iou_losses)

    total_loss = total_bce_loss + total_ssim_loss + total_iou_loss
    total_loss = total_loss / len(levels)

    return total_loss

  def _iou_loss(self, sigmoids, labels):
    total_iou_loss = 0

    intersection = tf.reduce_sum(sigmoids[:, :, :, :] * labels[:, :, :, :])
    union = tf.reduce_sum(sigmoids[:, :, :, :]) + tf.reduce_sum(
        labels[:, :, :, :]) - intersection
    iou = intersection / union
    total_iou_loss += 1-iou

    return total_iou_loss
