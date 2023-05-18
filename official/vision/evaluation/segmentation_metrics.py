# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from official.vision.evaluation import iou
from official.vision.ops import box_ops
from official.vision.ops import spatial_transform_ops


class MeanIoU(tf.keras.metrics.MeanIoU):
  """Mean IoU metric for semantic segmentation.

  This class utilizes tf.keras.metrics.MeanIoU to perform batched mean iou when
  both input images and ground-truth masks are resized to the same size
  (rescale_predictions=False). It also computes mean IoU on ground-truth
  original sizes, in which case, each prediction is rescaled back to the
  original image size.
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
        - masks: [batch, height, width, 1], ground-truth masks.
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
    predictions, masks, valid_masks = preprocess_inputs(
        y_true, y_pred, self._rescale_predictions)

    # Ignored mask elements are set to zero for fitting the confusion matrix.
    masks = tf.where(valid_masks, masks, tf.zeros_like(masks))

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


class PerClassIoUV2(iou.PerClassIoUV2):
  """Computes the per-class IoU metric for semantic segmentation.

  This implementation converts predictions and ground truth to binary masks,
  and uses logical AND and OR to compute intersection and union, which is much
  faster than the MeanIoU and PerClassIoU (using confusion matrix) above on TPU,
  but slower on CPU and GPU.
  """

  def __init__(self,
               num_classes: int,
               rescale_predictions: bool = False,
               name: Optional[str] = None,
               dtype: Optional[Union[str, tf.dtypes.DType]] = tf.float32,
               shape: Optional[Sequence[int]] = None,
               axis: int = -1):
    """Constructs Segmentation evaluator class.

    Args:
      num_classes: `int`, number of classes.
      rescale_predictions: `bool`, whether to scale back prediction to original
        image sizes. If True, y_true['image_info'] is used to rescale
        predictions.
      name: `str`, name of the metric instance.
      dtype: data type of the metric result.
      shape: shape of the metrics result.
      axis: (Optional) Defaults to -1. The dimension containing the one-hot
        values.
    """
    super().__init__(
        num_classes=num_classes, name=name, dtype=dtype, shape=shape, axis=axis)
    self._rescale_predictions = rescale_predictions

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """Updates metric state.

    Args:
      y_true: `dict`, dictionary with the following name, and key values.
        - masks: [batch, height, width, num_layers], ground-truth masks. The
          num_layers is 1 by default, while all the operations in this function
          support num_layers > 1.
        - valid_masks: [batch, height, width, num_layers], valid elements in the
          mask.
        - image_info: [batch, 4, 2], a tensor that holds information about
          original and preprocessed images. Each entry is in the format of
          [[original_height, original_width], [input_height, input_width],
          [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
          desired_width] is the actual scaled image size, and [y_scale, x_scale]
          is the scaling factor, which is the ratio of scaled dimension /
          original dimension.
      y_pred: Tensor [batch, height_p, width_p, num_classes], predicated masks.
    """
    logits, gt_masks, valid_masks = preprocess_inputs(y_true, y_pred,
                                                      self._rescale_predictions)
    valid_masks = tf.cast(valid_masks, tf.bool)

    gt_binary_masks = tf.one_hot(
        tf.cast(gt_masks[..., 0], dtype=tf.int32),
        depth=self.num_classes,
        on_value=True,
        off_value=False,
    )
    gt_binary_masks &= valid_masks

    predictions_binary_masks = tf.one_hot(
        tf.argmax(logits, axis=-1, output_type=tf.int32),
        depth=self.num_classes,
        on_value=True,
        off_value=False,
    )
    predictions_binary_masks &= valid_masks

    super().update_state(
        y_true=gt_binary_masks, y_pred=predictions_binary_masks
    )


class MeanIoUV2(PerClassIoUV2):
  """Computes the mean IoU metric for semantic segmentation."""

  def __init__(self,
               target_class_ids: Optional[Tuple[int, ...]] = None,
               **kwargs):
    """Initializes the class.

    Args:
      target_class_ids: computes mean IoU for the target classes. Selects all
        the if empty.
      **kwargs: the other arguments for initializing the base class.
    """
    super().__init__(**kwargs)
    self._target_class_ids = target_class_ids

  def result(self) -> tf.Tensor:
    """Average the IoUs of all the classes."""
    # (num_classes, )
    per_class_ious = super().result()
    if self._target_class_ids:
      # (num_classes, )
      target_class_indicators = tf.reduce_max(
          tf.one_hot(
              self._target_class_ids,
              depth=self.num_classes,
              dtype=per_class_ious.dtype),
          axis=0)
      return tf.math.divide_no_nan(
          tf.reduce_sum(per_class_ious * target_class_indicators),
          tf.reduce_sum(target_class_indicators))
    else:
      return tf.reduce_mean(per_class_ious)


def preprocess_inputs(
    y_true: tf.Tensor, y_pred: tf.Tensor,
    rescale_predictions: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Pre-processes the inputs (predictions and ground-truth) of the metrics.

  Args:
    y_true: `dict`, dictionary with the following name, and key values.
      - masks: [batch, height, width, num_layers], ground-truth masks. The
        num_layers is 1 by default, while all the operations in this function
        support num_layers > 1.
      - valid_masks: [batch, height, width, num_layers], valid elements in the
        mask.
      - image_info: [batch, 4, 2], a tensor that holds information about
        original and preprocessed images. Each entry is in the format of
        [[original_height, original_width], [input_height, input_width],
        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
        desired_width] is the actual scaled image size, and [y_scale, x_scale]
        is the scaling factor, which is the ratio of scaled dimension /
        original dimension.
    y_pred: tensor [batch, height_p, width_p, num_classes], predicated masks.
    rescale_predictions: `bool`, whether to scale back prediction to original
      image sizes. If True, y_true['image_info'] is used to rescale predictions.

  Returns:
    logits: a float tensor in shape [batch, height, width, num_classes], which
      stores the raw output of the model.
    gt_masks: an int tensor in shape [batch, height, width, 1], which stores the
      ground-truth masks.
    valid_masks: a bool tensor in shape [batch, height, width, 1], which
      indicates the valid elements of the masks.
  """
  logits = y_pred
  gt_masks = y_true['masks']
  valid_masks = y_true['valid_masks']
  images_info = y_true['image_info']

  if isinstance(logits, tuple) or isinstance(logits, list):
    logits = tf.concat(logits, axis=0)
    gt_masks = tf.concat(gt_masks, axis=0)
    valid_masks = tf.concat(valid_masks, axis=0)
    images_info = tf.concat(images_info, axis=0)

  # The pixel is valid if any layer of the masks is valid at that pixel.
  # (batch_size, height, width)
  valid_masks = tf.reduce_any(tf.cast(valid_masks, tf.bool), axis=-1)

  gt_masks_size = tf.shape(gt_masks)[1:3]
  if rescale_predictions:
    # Scale back predictions to original image shapes and pad to mask size.
    # Note: instead of cropping the masks to image shape (dynamic), here we
    # pad the rescaled predictions to mask size (fixed). And update the
    # valid_masks to mask out the pixels outside the original image shape.
    logits, image_shape_masks = (
        _rescale_and_pad_predictions(
            logits, images_info, output_size=gt_masks_size))
    # Only the area within the original image shape is valid.
    # (batch_size, height, width)
    valid_masks &= image_shape_masks
  else:
    logits = tf.image.resize(
        logits, gt_masks_size, method=tf.image.ResizeMethod.BILINEAR)

  # (batch_size, height, width, 1)
  valid_masks = valid_masks[..., tf.newaxis]

  return logits, gt_masks, valid_masks


def _rescale_and_pad_predictions(
    predictions: tf.Tensor, images_info: tf.Tensor,
    output_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
