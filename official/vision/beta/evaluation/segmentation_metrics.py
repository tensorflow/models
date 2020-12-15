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
"""Metrics for segmentation."""

import tensorflow as tf


class MeanIoU(tf.keras.metrics.MeanIoU):
  """Mean IoU metric for semantic segmentation.

  This class utilizes tf.keras.metrics.MeanIoU to perform batched mean iou when
  both input images and groundtruth masks are resized to the same size
  (rescale_predictions=False). It also computes mean iou on groundtruth original
  sizes, in which case, each prediction is rescaled back to the original image
  size.
  """

  def __init__(
      self, num_classes, rescale_predictions=False, name=None, dtype=None):
    """Constructs Segmentation evaluator class.

    Args:
      num_classes: `int`, number of classes.
      rescale_predictions: `bool`, whether to scale back prediction to original
        image sizes. If True, y_true['image_info'] is used to rescale
        predictions.
      name: `str`, name of the metric instance..
      dtype: data type of the metric result.
    """
    self._rescale_predictions = rescale_predictions
    super(MeanIoU, self).__init__(
        num_classes=num_classes, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred):
    """Updates metic state.

    Args:
      y_true: `dict`, dictionary with the following name, and key values.
        - masks: [batch, width, height, 1], groundtruth masks.
        - valid_masks: [batch, width, height, 1], valid elements in the mask.
        - image_info: [batch, 4, 2], a tensor that holds information about
          original and preprocessed images. Each entry is in the format of
          [[original_height, original_width], [input_height, input_width],
          [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
          desired_width] is the actual scaled image size, and [y_scale, x_scale]
          is the scaling factor, which is the ratio of scaled dimension /
          original dimension.
      y_pred: Tensor [batch, width_p, height_p, num_classes], predicated masks.
    """
    predictions = y_pred
    masks = y_true['masks']
    valid_masks = y_true['valid_masks']
    images_info = y_true['image_info']

    if isinstance(predictions, tuple) or isinstance(predictions, list):
      predictions = tf.concat(predictions, axis=0)
      masks = tf.concat(masks, axis=0)
      valid_masks = tf.concat(valid_masks, axis=0)
      images_info = tf.concat(images_info, axis=0)

    # Ignore mask elements is set to zero for argmax op.
    masks = tf.where(valid_masks, masks, tf.zeros_like(masks))

    if self._rescale_predictions:
      # This part can only run on cpu/gpu due to dynamic image resizing.
      flatten_predictions = []
      flatten_masks = []
      flatten_valid_masks = []
      for mask, valid_mask, predicted_mask, image_info in zip(
          masks, valid_masks, predictions, images_info):

        rescale_size = tf.cast(
            tf.math.ceil(image_info[1, :] / image_info[2, :]), tf.int32)
        image_shape = tf.cast(image_info[0, :], tf.int32)
        offsets = tf.cast(image_info[3, :], tf.int32)

        predicted_mask = tf.image.resize(
            predicted_mask,
            rescale_size,
            method=tf.image.ResizeMethod.BILINEAR)

        predicted_mask = tf.image.crop_to_bounding_box(predicted_mask,
                                                       offsets[0], offsets[1],
                                                       image_shape[0],
                                                       image_shape[1])
        mask = tf.image.crop_to_bounding_box(mask, 0, 0, image_shape[0],
                                             image_shape[1])
        valid_mask = tf.image.crop_to_bounding_box(valid_mask, 0, 0,
                                                   image_shape[0],
                                                   image_shape[1])

        predicted_mask = tf.argmax(predicted_mask, axis=2)
        flatten_predictions.append(tf.reshape(predicted_mask, shape=[1, -1]))
        flatten_masks.append(tf.reshape(mask, shape=[1, -1]))
        flatten_valid_masks.append(tf.reshape(valid_mask, shape=[1, -1]))
      flatten_predictions = tf.concat(flatten_predictions, axis=1)
      flatten_masks = tf.concat(flatten_masks, axis=1)
      flatten_valid_masks = tf.concat(flatten_valid_masks, axis=1)

    else:
      predictions = tf.image.resize(
          predictions,
          tf.shape(masks)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
      predictions = tf.argmax(predictions, axis=3)
      flatten_predictions = tf.reshape(predictions, shape=[-1])
      flatten_masks = tf.reshape(masks, shape=[-1])
      flatten_valid_masks = tf.reshape(valid_masks, shape=[-1])

    super(MeanIoU, self).update_state(
        flatten_masks, flatten_predictions,
        tf.cast(flatten_valid_masks, tf.float32))

