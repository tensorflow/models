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

"""Metrics for segmentation."""

import tensorflow as tf

from official.vision.ops import box_ops
from official.vision.ops import spatial_transform_ops


class MeanIoU(tf.keras.metrics.MeanIoU):
  """Mean IoU metric for semantic segmentation.

  This class utilizes tf.keras.metrics.MeanIoU to perform batched mean iou when
  both input images and groundtruth masks are resized to the same size
  (rescale_predictions=False). It also computes mean iou on groundtruth original
  sizes, in which case, each prediction is rescaled back to the original image
  size.
  """

  def __init__(self,
               num_classes,
               rescale_predictions=False,
               name=None,
               dtype=None):
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
    super().__init__(num_classes=num_classes, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred):
    """Updates metric state.

    Args:
      y_true: `dict`, dictionary with the following name, and key values.
        - masks: [batch, height, width, 1], groundtruth masks.
        - valid_masks: [batch, height, width, 1], valid elements in the mask.
        - image_info: [batch, 4, 2], a tensor that holds information about
          original and preprocessed images. Each entry is in the format of
          [[original_height, original_width], [input_height, input_width],
          [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
          desired_width] is the actual scaled image size, and [y_scale, x_scale]
          is the scaling factor, which is the ratio of scaled dimension /
          original dimension.
      y_pred: Tensor [batch, height_p, width_p, num_classes], predicated masks.
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
    masks_size = tf.shape(masks)[1:3]

    if self._rescale_predictions:
      # Scale back predictions to original image shapes and pad to mask size.
      # Note: instead of cropping the masks to image shape (dynamic), here we
      # pad the rescaled predictions to mask size (fixed). And update the
      # valid_masks to mask out the pixels outside the original image shape.
      predictions, image_shape_masks = _rescale_and_pad_predictions(
          predictions, images_info, output_size=masks_size)
      # Only the area within the original image shape is valid.
      # (batch_size, height, width, 1)
      valid_masks = tf.cast(valid_masks, tf.bool) & tf.expand_dims(
          image_shape_masks, axis=-1)
    else:
      predictions = tf.image.resize(
          predictions, masks_size, method=tf.image.ResizeMethod.BILINEAR)

    predictions = tf.argmax(predictions, axis=3)
    flatten_predictions = tf.reshape(predictions, shape=[-1])
    flatten_masks = tf.reshape(masks, shape=[-1])
    flatten_valid_masks = tf.reshape(valid_masks, shape=[-1])

    super().update_state(
        y_true=flatten_masks,
        y_pred=flatten_predictions,
        sample_weight=tf.cast(flatten_valid_masks, tf.float32))


class PerClassIoU(MeanIoU):
  """Per class IoU metric for semantic segmentation."""

  def result(self):
    """Compute IoU for each class via the confusion matrix."""
    sum_over_row = tf.cast(
        tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = tf.cast(
        tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = tf.cast(
        tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    return tf.math.divide_no_nan(true_positives, denominator)


def _rescale_and_pad_predictions(predictions, images_info, output_size):
  """Scales back predictions to original image shapes and pads to output size.

  Args:
    predictions: A tensor in shape [batch, height, width, num_classes] which
      stores the model predictions.
    images_info: A tensor in shape [batch, 4, 2] that holds information about
      original and preprocessed images. Each entry is in the format of
      [[original_height, original_width], [input_height, input_width], [y_scale,
      x_scale], [y_offset, x_offset]], where [desired_height, desired_width] is
      the actual scaled image size, and [y_scale, x_scale] is the scaling
      factor, which is the ratio of scaled dimension / original dimension.
    output_size: A list/tuple/tensor stores the size of the padded output in
      [output_height, output_width].

  Returns:
    predictions: A tensor in shape [batch, output_height, output_width,
      num_classes] which stores the rescaled and padded predictions.
    image_shape_masks: A bool tensor in shape [batch, output_height,
      output_width] where the pixels inside the original image shape are true,
      otherwise false.
  """
  # (batch_size, 2)
  image_shape = tf.cast(images_info[:, 0, :], tf.int32)
  desired_size = tf.cast(images_info[:, 1, :], tf.float32)
  image_scale = tf.cast(images_info[:, 2, :], tf.float32)
  offset = tf.cast(images_info[:, 3, :], tf.int32)
  rescale_size = tf.cast(tf.math.ceil(desired_size / image_scale), tf.int32)

  # Rescale the predictions, then crop to the original image shape and
  # finally pad zeros to match the mask size.
  predictions = (
      spatial_transform_ops.bilinear_resize_with_crop_and_pad(
          predictions,
          rescale_size,
          crop_offset=offset,
          crop_size=image_shape,
          output_size=output_size))

  # (batch_size, 2)
  y0_x0 = tf.broadcast_to(
      tf.constant([[0, 0]], dtype=image_shape.dtype), tf.shape(image_shape))
  # (batch_size, 4)
  image_shape_bbox = tf.concat([y0_x0, image_shape], axis=1)
  # (batch_size, height, width)
  image_shape_masks = box_ops.bbox2mask(
      bbox=image_shape_bbox,
      image_height=output_size[0],
      image_width=output_size[1],
      dtype=tf.bool)

  return predictions, image_shape_masks
