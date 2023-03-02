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

"""Metrics for instance detection & segmentation."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from official.vision.ops import box_ops
from official.vision.ops import mask_ops


class AveragePrecision(tf.keras.layers.Layer):
  """The algorithm which computes average precision from P-R curve."""

  def call(self, precisions, recalls):
    """Computes average precision."""
    raise NotImplementedError


class COCOAveragePrecision(AveragePrecision):
  """Average precision in COCO style.

  In COCO, AP is defined as the mean of interpolated precisions at a set of 101
  equally spaced recall points [0, 0.01, ..., 1]. For each recall point r,
  the precision is interpolated to the maximum precision with corresponding
  recall r' >= r.

  The VOC challenges before 2010 used the similar method, but only 11 recall
  points [0, 0.1, ..., 1].
  """

  def __init__(
      self, num_recall_eval_points: int = 101, recalls_desc: bool = False
  ):
    """Initialization for COCOAveragePrecision.

    Args:
      num_recall_eval_points: the number of equally spaced recall points used
        for interpolating the precisions.
      recalls_desc: If true, the recalls are in descending order.
    """
    super().__init__()
    self._num_recall_eval_points = num_recall_eval_points
    self._recalls_desc = recalls_desc

  def get_config(self) -> Dict[str, Any]:
    return {
        'num_recall_eval_points': self._num_recall_eval_points,
        'recalls_desc': self._recalls_desc,
    }

  def call(self, precisions: tf.Tensor, recalls: tf.Tensor) -> tf.Tensor:
    """Computes average precision.

    Args:
      precisions: a tensor in shape (dim_0, ..., num_confidences) which stores a
        list of precision values at different confidence thresholds with
        arbitrary numbers of leading dimensions.
      recalls: a tensor in shape (dim_0, ..., num_confidences) which stores a
        list of recall values at different confidence threshold with arbitrary
        numbers of leading dimensions.

    Returns:
      A tensor in shape (dim_0, ...), which stores the area under P-R curve.
    """
    p = precisions
    r = recalls

    if not isinstance(p, tf.Tensor):
      p = tf.convert_to_tensor(p)
    if not isinstance(r, tf.Tensor):
      r = tf.convert_to_tensor(r)

    if self._recalls_desc:
      p = tf.reverse(p, axis=[-1])
      r = tf.reverse(r, axis=[-1])

    r_eval_points = tf.linspace(0.0, 1.0, self._num_recall_eval_points)
    # (dim_0, ..., num_recall_eval_points)
    # For each recall eval point, the precision is interpolated to the maximum
    # precision with corresponding recall >= the recall eval point.
    p_max = tf.reduce_max(
        p[..., tf.newaxis, :]
        * tf.cast(
            r[..., tf.newaxis, :] >= r_eval_points[:, tf.newaxis], dtype=p.dtype
        ),
        axis=-1,
    )
    # (dim_0, ...)
    return tf.reduce_mean(p_max, axis=-1)


class VOC2010AveragePrecision(AveragePrecision):
  """Average precision in VOC 2010 style.

  Since VOC 2010, first compute an approximation of the measured P-R curve
  with precision monotonically decreasing, by setting the precision for recall
  r to the maximum precision obtained for any recall r' >= r. Then compute the
  AP as the area under this curve by numerical integration.
  """

  def __init__(self, recalls_desc: bool = False):
    """Initialization for VOC10AveragePrecision.

    Args:
      recalls_desc: If true, the recalls are in descending order.
    """
    super().__init__()
    self._recalls_desc = recalls_desc

  def get_config(self) -> Dict[str, Any]:
    return {
        'recalls_desc': self._recalls_desc,
    }

  def call(self, precisions: tf.Tensor, recalls: tf.Tensor) -> tf.Tensor:
    """Computes average precision.

    Args:
      precisions: a tensor in shape (dim_0, ..., num_confidences) which stores a
        list of precision values at different confidence thresholds with
        arbitrary numbers of leading dimensions.
      recalls: a tensor in shape (dim_0, ..., num_confidences) which stores a
        list of recall values at different confidence threshold with arbitrary
        numbers of leading dimensions.

    Returns:
      A tensor in shape (dim_0, ...), which stores the area under P-R curve.
    """
    p = precisions
    r = recalls

    if not isinstance(p, tf.Tensor):
      p = tf.convert_to_tensor(p)
    if not isinstance(r, tf.Tensor):
      r = tf.convert_to_tensor(r)

    if self._recalls_desc:
      p = tf.reverse(p, axis=[-1])
      r = tf.reverse(r, axis=[-1])

    axis_indices = list(range(len(p.get_shape())))

    # Transpose to (num_confidences, ...), because tf.scan only applies to the
    # first dimension.
    p = tf.transpose(p, np.roll(axis_indices, 1))
    # Compute cumulative maximum in reverse order.
    # For example, the reverse cumulative maximum of [5,6,3,4,2,1] is
    # [6,6,4,4,2,1].
    p = tf.scan(
        tf.maximum, elems=p, initializer=tf.reduce_min(p, axis=0), reverse=True
    )
    # Transpose back to (..., num_confidences)
    p = tf.transpose(p, np.roll(axis_indices, -1))

    # Prepend 0 to r and compute the delta.
    r = tf.concat([tf.zeros_like(r[..., 0:1]), r], axis=-1)
    delta_r = tf.roll(r, shift=-1, axis=-1) - r

    return tf.reduce_sum(p * delta_r[..., :-1], axis=-1)


class MatchingAlgorithm(tf.keras.layers.Layer):
  """The algorithm which matches detections to ground truths."""

  def call(
      self,
      detection_to_gt_ious: tf.Tensor,
      detection_classes: tf.Tensor,
      detection_scores: tf.Tensor,
      gt_classes: tf.Tensor,
  ):
    """Matches detections to ground truths."""
    raise NotImplementedError


class COCOMatchingAlgorithm(MatchingAlgorithm):
  """The detection matching algorithm used in COCO."""

  def __init__(self, iou_thresholds: Tuple[float, ...]):
    """Initialization for COCOMatchingAlgorithm.

    Args:
      iou_thresholds: a list of IoU thresholds.
    """
    super().__init__()
    self._iou_thresholds = iou_thresholds

  def get_config(self) -> Dict[str, Any]:
    return {
        'iou_thresholds': self._iou_thresholds,
    }

  def call(
      self,
      detection_to_gt_ious: tf.Tensor,
      detection_classes: tf.Tensor,
      detection_scores: tf.Tensor,
      gt_classes: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Matches detections to ground truths.

    This is the matching algorithm used in COCO. First, sort all the detections
    based on the scores from high to low. Then for each detection, iterates
    through all ground truth. The unmatched ground truth with the highest IoU
    greater than the threshold is matched to the detection.

    Args:
      detection_to_gt_ious: a tensor in shape of (batch_size, num_detections,
        num_gts) which stores the IoUs for each pair of detection and ground
        truth.
      detection_classes: a tensor in shape of (batch_size, num_detections) which
        stores the classes of the detections.
      detection_scores: a tensor in shape of (batch_size, num_detections) which
        stores the scores of the detections.
      gt_classes: a tensor in shape of (batch_size, num_gts) which stores the
        classes of the ground truth boxes.

    Returns:
      Two bool tensors in shape of (batch_size, num_detections,
      num_iou_thresholds) and (batch_size, num_gts, num_iou_thresholds) which
      indicates whether the detections and ground truths are true positives at
      different IoU thresholds.
    """
    batch_size = tf.shape(detection_classes)[0]
    num_detections = detection_classes.get_shape()[1]
    num_gts = gt_classes.get_shape()[1]
    num_iou_thresholds = len(self._iou_thresholds)

    # (batch_size, num_detections)
    sorted_detection_indices = tf.argsort(
        detection_scores, axis=1, direction='DESCENDING'
    )
    # (batch_size, num_detections)
    sorted_detection_classes = tf.gather(
        detection_classes, sorted_detection_indices, batch_dims=1
    )
    # (batch_size, num_detections, num_gts)
    sorted_detection_to_gt_ious = tf.gather(
        detection_to_gt_ious, sorted_detection_indices, batch_dims=1
    )

    init_loop_vars = (
        0,  # i: the loop counter
        tf.zeros(
            [batch_size, num_detections, num_iou_thresholds], dtype=tf.bool
        ),  # detection_is_tp
        tf.zeros(
            [batch_size, num_gts, num_iou_thresholds], dtype=tf.bool
        ),  # gt_is_tp
    )

    def _match_detection_to_gt_loop_body(
        i: int, detection_is_tp: tf.Tensor, gt_is_tp: tf.Tensor
    ) -> Tuple[int, tf.Tensor, tf.Tensor]:
      """Iterates the sorted detections and matches to the ground truths."""
      # (batch_size, num_gts)
      gt_ious = sorted_detection_to_gt_ious[:, i, :]
      # (batch_size, num_gts, num_iou_thresholds)
      gt_matches_detection = (
          # Ground truth is not matched yet.
          ~gt_is_tp
          # IoU is greater than the threshold.
          & (gt_ious[:, :, tf.newaxis] > self._iou_thresholds)
          # Classes are matched.
          & (
              (sorted_detection_classes[:, i][:, tf.newaxis] == gt_classes)
              & (gt_classes > 0)
          )[:, :, tf.newaxis]
      )
      # Finds the matched ground truth with max IoU.
      # If there is no matched ground truth, the argmax op will return index 0
      # in this step. It's fine because it will be masked out in the next step.
      # (batch_size, num_iou_thresholds)
      matched_gt_with_max_iou = tf.argmax(
          tf.cast(gt_matches_detection, tf.float32) * gt_ious[:, :, tf.newaxis],
          axis=1,
          output_type=tf.int32,
      )
      # (batch_size, num_gts, num_iou_thresholds)
      gt_matches_detection &= tf.one_hot(
          matched_gt_with_max_iou,
          depth=num_gts,
          on_value=True,
          off_value=False,
          axis=1,
      )

      # Updates detection_is_tp
      # Map index back to the unsorted detections.
      # (batch_size, num_detections, num_iou_thresholds)
      detection_is_tp |= (
          tf.reduce_any(gt_matches_detection, axis=1, keepdims=True)
          & tf.one_hot(
              sorted_detection_indices[:, i],
              depth=num_detections,
              on_value=True,
              off_value=False,
              axis=-1,
          )[:, :, tf.newaxis]
      )
      detection_is_tp.set_shape([None, num_detections, num_iou_thresholds])

      # Updates gt_is_tp
      # (batch_size, num_gts, num_iou_thresholds)
      gt_is_tp |= gt_matches_detection
      gt_is_tp.set_shape([None, num_gts, num_iou_thresholds])

      # Returns the updated loop vars.
      return (i + 1, detection_is_tp, gt_is_tp)

    _, detection_is_tp_result, gt_is_tp_result = tf.while_loop(
        cond=lambda i, *_: i < num_detections,
        body=_match_detection_to_gt_loop_body,
        loop_vars=init_loop_vars,
        parallel_iterations=32,
        maximum_iterations=num_detections,
    )
    return detection_is_tp_result, gt_is_tp_result


def _shift_and_rescale_boxes(
    boxes: tf.Tensor,
    output_boundary: Tuple[int, int],
) -> tf.Tensor:
  """Shift and rescale the boxes to fit in the output boundary.

  The output boundary of the boxes can be smaller than the original image size
  for accelerating the downstream calculations (dynamic mask resizing, mask IoU,
  etc.).

  For each image of the batch:
  (1) find the upper boundary (min_ymin) and the left boundary (min_xmin) of all
      the boxes.
  (2) shift all the boxes up min_ymin pixels and left min_xmin pixels.
  (3) find the new lower boundary (max_ymax) and the right boundary (max_xmax)
      of all the boxes.
  (4) if max_ymax > output_height or max_xmax > output_width (some boxes don't
      fit in the output boundary), downsample all the boxes by ratio:
      min(output_height / max_ymax, output_width / max_xmax). The aspect ratio
      is not changed.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. The last dimension is
      the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    output_boundary: two integers that represent the height and width of the
      output.

  Returns:
    The tensor [batch_size, N, 4] of the output boxes.
  """
  boxes = tf.cast(boxes, dtype=tf.float32)

  # (batch_size, num_boxes, 1)
  is_valid_box = tf.reduce_any(
      (boxes[:, :, 2:4] - boxes[:, :, 0:2]) > 0, axis=-1, keepdims=True
  )

  # (batch_size, 2)
  min_ymin_xmin = tf.reduce_min(
      tf.where(is_valid_box, boxes, np.inf)[:, :, 0:2],
      axis=1,
  )
  # (batch_size, num_boxes, 4)
  boxes = tf.where(
      is_valid_box,
      boxes - tf.tile(min_ymin_xmin, [1, 2])[:, tf.newaxis, :],
      0.0,
  )

  # (batch_size,)
  max_ymax = tf.reduce_max(boxes[:, :, 2], axis=1)
  max_xmax = tf.reduce_max(boxes[:, :, 3], axis=1)
  # (batch_size,)
  y_resize_ratio = output_boundary[0] / max_ymax
  x_resize_ratio = output_boundary[1] / max_xmax
  # (batch_size,)
  downsampling_ratio = tf.math.minimum(
      tf.math.minimum(y_resize_ratio, x_resize_ratio), 1.0
  )
  # (batch_size, num_boxes, 4)
  return boxes * downsampling_ratio[:, tf.newaxis, tf.newaxis]


def _count_detection_type(
    detection_type_mask: tf.Tensor,
    detection_classes: tf.Tensor,
    flattened_binned_confidence_one_hot: tf.Tensor,
    num_classes: int,
) -> tf.Tensor:
  """Counts detection type grouped by IoU thresholds, classes and confidence bins.

  Args:
    detection_type_mask: a bool tensor in shape of (batch_size, num_detections,
      num_iou_thresholds), which indicate a certain type of detections (e.g.
      true postives).
    detection_classes: a tensor in shape of (batch_size, num_detections) which
      stores the classes of the detections.
    flattened_binned_confidence_one_hot: a one-hot bool tensor in shape of
      (batch_size * num_detections, num_confidence_bins + 1) which indicates the
      binned confidence score of each detection.
    num_classes: the number of classes.

  Returns:
    A tensor in shape of (num_iou_thresholds, num_classes,
    num_confidence_bins + 1) which stores the count grouped by IoU thresholds,
    classes and confidence bins.
  """
  num_iou_thresholds = detection_type_mask.get_shape()[-1]

  # (batch_size, num_detections, num_iou_thresholds)
  masked_classes = tf.where(
      detection_type_mask, detection_classes[..., tf.newaxis], -1
  )
  # (num_iou_thresholds, batch_size * num_detections)
  flattened_masked_classes = tf.transpose(
      tf.reshape(masked_classes, [-1, num_iou_thresholds])
  )
  # (num_iou_thresholds, num_classes, batch_size * num_detections)
  flattened_masked_classes_one_hot = tf.one_hot(
      flattened_masked_classes, depth=num_classes, axis=1
  )
  # (num_iou_thresholds * num_classes, batch_size * num_detections)
  flattened_masked_classes_one_hot = tf.reshape(
      flattened_masked_classes_one_hot,
      [num_iou_thresholds * num_classes, -1],
  )

  # (num_iou_thresholds * num_classes, num_confidence_bins + 1)
  count = tf.matmul(
      flattened_masked_classes_one_hot,
      tf.cast(flattened_binned_confidence_one_hot, tf.float32),
      a_is_sparse=True,
      b_is_sparse=True,
  )
  # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
  count = tf.reshape(count, [num_iou_thresholds, num_classes, -1])
  # Clears the count of class 0 (background)
  count *= 1.0 - tf.eye(num_classes, 1, dtype=count.dtype)
  return count


class InstanceMetrics(tf.keras.metrics.Metric):
  """Reports the metrics of instance detection & segmentation."""

  def __init__(
      self,
      num_classes: int,
      use_masks: bool = False,
      iou_thresholds: Tuple[float, ...] = (0.5,),
      confidence_thresholds: Tuple[float, ...] = (),
      num_confidence_bins: int = 1000,
      mask_output_boundary: Tuple[int, int] = (640, 640),
      matching_algorithm: Optional[MatchingAlgorithm] = None,
      average_precision_algorithms: Optional[
          Dict[str, AveragePrecision]
      ] = None,
      name: Optional[str] = None,
      dtype: Optional[Union[str, tf.dtypes.DType]] = tf.float32,
      **kwargs
  ):
    """Initialization for AveragePrecision.

    Args:
      num_classes: the number of classes.
      use_masks: if true, use the masks of the instances when calculating the
        metrics, otherwise use the boxes.
      iou_thresholds: a sequence of IoU thresholds over which to calculate the
        instance metrics.
      confidence_thresholds: a sequence of confidence thresholds. If set, also
        report precision and recall at each confidence threshold, otherwise,
        only report average precision.
      num_confidence_bins: the number of confidence bins used for bin sort.
      mask_output_boundary: two integers that represent the height and width of
        the boundary where the resized instance masks are pasted. For each
        example, if any of the detection or ground truth boxes is out of the
        boundary, shift and resize all the detection and ground truth boxes of
        the example to fit them into the boundary. The output boundary of the
        pasted masks can be smaller than the real image size for accelerating
        the calculation.
      matching_algorithm: the algorithm which matches detections to ground
        truths.
      average_precision_algorithms: the algorithms which compute average
        precision from P-R curve. The keys are used in the metrics results.
      name: the name of the metric instance.
      dtype: data type of the metric result.
      **kwargs: Additional keywords arguments.
    """
    super().__init__(name=name, dtype=dtype, **kwargs)
    self._num_classes = num_classes
    self._use_masks = use_masks
    self._iou_thresholds = iou_thresholds
    self._confidence_thresholds = confidence_thresholds
    self._num_iou_thresholds = len(iou_thresholds)
    self._num_confidence_bins = num_confidence_bins
    self._mask_output_boundary = mask_output_boundary
    if not matching_algorithm:
      self._matching_algorithm = COCOMatchingAlgorithm(iou_thresholds)
    else:
      self._matching_algorithm = matching_algorithm
    if not average_precision_algorithms:
      self._average_precision_algorithms = {'ap': COCOAveragePrecision()}
    else:
      self._average_precision_algorithms = average_precision_algorithms

    # Variables
    self.tp_count = self.add_weight(
        'tp_count',
        shape=[
            self._num_iou_thresholds,
            self._num_classes,
            self._num_confidence_bins + 1,
        ],
        initializer='zeros',
        dtype=tf.float32,
    )
    self.fp_count = self.add_weight(
        'fp_count',
        shape=[
            self._num_iou_thresholds,
            self._num_classes,
            self._num_confidence_bins + 1,
        ],
        initializer='zeros',
        dtype=tf.float32,
    )
    self.gt_count = self.add_weight(
        'gt_count',
        shape=[self._num_classes],
        initializer='zeros',
        dtype=tf.float32,
    )

  def get_config(self) -> Dict[str, Any]:
    """Returns the serializable config of the metric."""
    return {
        'num_classes': self._num_classes,
        'use_masks': self._use_masks,
        'iou_thresholds': self._iou_thresholds,
        'confidence_thresholds': self._confidence_thresholds,
        'num_confidence_bins': self._num_confidence_bins,
        'mask_output_boundary': self._mask_output_boundary,
        'matching_algorithm': self._matching_algorithm,
        'average_precision_algorithms': self._average_precision_algorithms,
        'name': self.name,
        'dtype': self.dtype,
    }

  def reset_state(self):
    """Resets all of the metric state variables."""
    for v in self.variables:
      tf.keras.backend.set_value(v, np.zeros(v.shape))

  def update_state(
      self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]
  ):
    # (batch_size, num_detections, 4) in absolute coordinates.
    detection_boxes = tf.cast(y_pred['detection_boxes'], tf.float32)
    # (batch_size, num_detections)
    detection_classes = tf.cast(y_pred['detection_classes'], tf.int32)
    # (batch_size, num_detections)
    detection_scores = tf.cast(y_pred['detection_scores'], tf.float32)
    # (batch_size, num_gts, 4) in absolute coordinates.
    gt_boxes = tf.cast(y_true['boxes'], tf.float32)
    # (batch_size, num_gts)
    gt_classes = tf.cast(y_true['classes'], tf.int32)
    # (batch_size, num_gts)
    if 'is_crowds' in y_true:
      gt_is_crowd = tf.cast(y_true['is_crowds'], tf.bool)
    else:
      gt_is_crowd = tf.zeros_like(gt_classes, dtype=tf.bool)

    image_scale = tf.tile(y_true['image_info'][:, 2:3, :], multiples=[1, 1, 2])
    detection_boxes = detection_boxes / tf.cast(
        image_scale, dtype=detection_boxes.dtype
    )

    # Step 1: Computes IoUs between the detections and the non-crowd ground
    # truths and IoAs between the detections and the crowd ground truths.
    if not self._use_masks:
      # (batch_size, num_detections, num_gts)
      detection_to_gt_ious = box_ops.bbox_overlap(detection_boxes, gt_boxes)
      detection_to_gt_ioas = box_ops.bbox_intersection_over_area(
          detection_boxes, gt_boxes
      )
    else:
      # Use outer boxes to generate the masks if available.
      if 'detection_outer_boxes' in y_pred:
        detection_boxes = y_pred['detection_outer_boxes']

      # (batch_size, num_detections, mask_height, mask_width)
      detection_masks = tf.cast(y_pred['detection_masks'], tf.float32)
      # (batch_size, num_gts, gt_mask_height, gt_mask_width)
      gt_masks = tf.cast(y_true['masks'], tf.float32)

      num_detections = detection_boxes.get_shape()[1]
      # (batch_size, num_detections + num_gts, 4)
      all_boxes = _shift_and_rescale_boxes(
          tf.concat([detection_boxes, gt_boxes], axis=1),
          self._mask_output_boundary,
      )
      detection_boxes = all_boxes[:, :num_detections, :]
      gt_boxes = all_boxes[:, num_detections:, :]
      # (batch_size, num_detections, num_gts)
      detection_to_gt_ious, detection_to_gt_ioas = (
          mask_ops.instance_masks_overlap(
              detection_boxes,
              detection_masks,
              gt_boxes,
              gt_masks,
              output_size=self._mask_output_boundary,
          )
      )
    # (batch_size, num_detections, num_gts)
    detection_to_gt_ious = tf.where(
        gt_is_crowd[:, tf.newaxis, :], 0.0, detection_to_gt_ious
    )
    detection_to_crowd_ioas = tf.where(
        gt_is_crowd[:, tf.newaxis, :], detection_to_gt_ioas, 0.0
    )

    # Step 2: counts true positives grouped by IoU thresholds, classes and
    # confidence bins.

    # (batch_size, num_detections, num_iou_thresholds)
    detection_is_tp, _ = self._matching_algorithm(
        detection_to_gt_ious, detection_classes, detection_scores, gt_classes
    )
    # (batch_size * num_detections,)
    flattened_binned_confidence = tf.reshape(
        tf.cast(detection_scores * self._num_confidence_bins, tf.int32), [-1]
    )
    # (batch_size * num_detections, num_confidence_bins + 1)
    flattened_binned_confidence_one_hot = tf.one_hot(
        flattened_binned_confidence, self._num_confidence_bins + 1, axis=1
    )
    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    tp_count = _count_detection_type(
        detection_is_tp,
        detection_classes,
        flattened_binned_confidence_one_hot,
        self._num_classes,
    )

    # Step 3: Counts false positives grouped by IoU thresholds, classes and
    # confidence bins.
    # False positive: detection is not true positive (see above) and not part of
    # the crowd ground truth with the same class.

    # (batch_size, num_detections, num_gts, num_iou_thresholds)
    detection_matches_crowd = (
        (detection_to_crowd_ioas[..., tf.newaxis] > self._iou_thresholds)
        & (
            detection_classes[:, :, tf.newaxis, tf.newaxis]
            == gt_classes[:, tf.newaxis, :, tf.newaxis]
        )
        & (detection_classes[:, :, tf.newaxis, tf.newaxis] > 0)
    )
    # (batch_size, num_detections, num_iou_thresholds)
    detection_matches_any_crowd = tf.reduce_any(
        detection_matches_crowd & ~detection_is_tp[:, :, tf.newaxis, :], axis=2
    )
    detection_is_fp = ~detection_is_tp & ~detection_matches_any_crowd
    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    fp_count = _count_detection_type(
        detection_is_fp,
        detection_classes,
        flattened_binned_confidence_one_hot,
        self._num_classes,
    )

    # Step 4: Counts non-crowd groundtruths grouped by classes.
    # (num_classes, )
    gt_count = tf.reduce_sum(
        tf.one_hot(
            tf.where(gt_is_crowd, -1, gt_classes), self._num_classes, axis=-1
        ),
        axis=[0, 1],
    )
    # Clears the count of class 0 (background).
    gt_count *= 1.0 - tf.eye(1, self._num_classes, dtype=gt_count.dtype)[0]

    # Accumulates the variables.
    self.fp_count.assign_add(tf.cast(fp_count, self.fp_count.dtype))
    self.tp_count.assign_add(tf.cast(tp_count, self.tp_count.dtype))
    self.gt_count.assign_add(tf.cast(gt_count, self.gt_count.dtype))

  def result(self) -> Dict[str, tf.Tensor]:
    """Returns the metrics values as a dict.

    Returns:
      A `dict` containing:
        'ap': a float tensor in shape (num_iou_thresholds, num_classes) which
        stores the average precision of each class at different IoU thresholds.
        'precision': a float tensor in shape (num_confidence_thresholds,
        num_iou_thresholds, num_classes) which stores the precision of each
        class at different confidence thresholds & IoU thresholds.
        'recall': a float tensor in shape (num_confidence_thresholds,
        num_iou_thresholds, num_classes) which stores the recall of each
        class at different confidence thresholds & IoU thresholds.
        'valid_classes': a bool tensor in shape (num_classes,). If False, there
        is no instance of the class in the ground truth.
    """
    result = {
        # (num_classes,)
        'valid_classes': self.gt_count != 0,
    }

    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    tp_count_cum_by_confidence = tf.math.cumsum(
        self.tp_count, axis=-1, reverse=True
    )
    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    fp_count_cum_by_confidence = tf.math.cumsum(
        self.fp_count, axis=-1, reverse=True
    )

    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    precisions = tf.math.divide_no_nan(
        tp_count_cum_by_confidence,
        tp_count_cum_by_confidence + fp_count_cum_by_confidence,
    )
    # (num_iou_thresholds, num_classes, num_confidence_bins + 1)
    recalls = tf.math.divide_no_nan(
        tp_count_cum_by_confidence, self.gt_count[..., tf.newaxis]
    )

    if self._confidence_thresholds:
      # If confidence_thresholds is set, reports precision and recall at each
      # confidence threshold.
      confidence_thresholds = tf.cast(
          tf.constant(self._confidence_thresholds, dtype=tf.float32)
          * self._num_confidence_bins,
          dtype=tf.int32,
      )
      # (num_confidence_thresholds, num_iou_thresholds, num_classes)
      result['precisions'] = tf.gather(
          tf.transpose(precisions, [2, 0, 1]), confidence_thresholds
      )
      result['recalls'] = tf.gather(
          tf.transpose(recalls, [2, 0, 1]), confidence_thresholds
      )

    precisions = tf.reverse(precisions, axis=[-1])
    recalls = tf.reverse(recalls, axis=[-1])
    result.update(
        {
            # (num_iou_thresholds, num_classes)
            key: ap_algorithm(precisions, recalls)
            for key, ap_algorithm in self._average_precision_algorithms.items()
        }
    )
    return result

  def get_average_precision_metrics_keys(self):
    """Gets the keys of the average precision metrics in the results."""
    return self._average_precision_algorithms.keys()
