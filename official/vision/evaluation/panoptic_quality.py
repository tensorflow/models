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

"""Implementation of the Panoptic Quality metric.

Panoptic Quality is an instance-based metric for evaluating the task of
image parsing, aka panoptic segmentation.

Please see the paper for details:
"Panoptic Segmentation", Alexander Kirillov, Kaiming He, Ross Girshick,
Carsten Rother and Piotr Dollar. arXiv:1801.00868, 2018.

Note that this metric class is branched from
https://github.com/tensorflow/models/blob/master/research/deeplab/evaluation/panoptic_quality.py
"""

import collections
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf, tf_keras

from official.vision.ops import box_ops

_EPSILON = 1e-10


def realdiv_maybe_zero(x, y):
  """Element-wise x / y where y may contain zeros, for those returns 0 too."""
  return np.where(
      np.less(np.abs(y), _EPSILON), np.zeros_like(x), np.divide(x, y))


def _ids_to_counts(id_array):
  """Given a numpy array, a mapping from each unique entry to its count."""
  ids, counts = np.unique(id_array, return_counts=True)
  return dict(zip(ids, counts))


class PanopticQuality:
  """Metric class for Panoptic Quality.

  "Panoptic Segmentation" by Alexander Kirillov, Kaiming He, Ross Girshick,
  Carsten Rother, Piotr Dollar.
  https://arxiv.org/abs/1801.00868
  """

  def __init__(self, num_categories, ignored_label, max_instances_per_category,
               offset):
    """Initialization for PanopticQualityMetric.

    Args:
      num_categories: The number of segmentation categories (or "classes" in the
        dataset).
      ignored_label: A category id that is ignored in evaluation, e.g. the void
        label as defined in COCO panoptic segmentation dataset.
      max_instances_per_category: The maximum number of instances for each
        category. Used in ensuring unique instance labels.
      offset: The maximum number of unique labels. This is used, by multiplying
        the ground-truth labels, to generate unique ids for individual regions
        of overlap between ground-truth and predicted segments.
    """
    self.num_categories = num_categories
    self.ignored_label = ignored_label
    self.max_instances_per_category = max_instances_per_category
    self.offset = offset
    self.reset()

  def _naively_combine_labels(self, category_mask, instance_mask):
    """Naively creates a combined label array from categories and instances."""
    return (category_mask.astype(np.uint32) * self.max_instances_per_category +
            instance_mask.astype(np.uint32))

  def compare_and_accumulate(self, groundtruths, predictions):
    """Compares predictions with ground-truths, and accumulates the metrics.

    It is not assumed that instance ids are unique across different categories.
    See for example combine_semantic_and_instance_predictions.py in official
    PanopticAPI evaluation code for issues to consider when fusing category
    and instance labels.

    Instances ids of the ignored category have the meaning that id 0 is "void"
    and remaining ones are crowd instances.

    Args:
      groundtruths: A dictionary contains ground-truth labels. It should contain
        the following fields.
        - category_mask: A 2D numpy uint16 array of ground-truth per-pixel
          category labels.
        - instance_mask: A 2D numpy uint16 array of ground-truth per-pixel
          instance labels.
      predictions: A dictionary contains the model outputs. It should contain
        the following fields.
        - category_array: A 2D numpy uint16 array of predicted per-pixel
          category labels.
        - instance_array: A 2D numpy uint16 array of predicted instance labels.
    """
    groundtruth_category_mask = groundtruths['category_mask']
    groundtruth_instance_mask = groundtruths['instance_mask']
    predicted_category_mask = predictions['category_mask']
    predicted_instance_mask = predictions['instance_mask']

    # First, combine the category and instance labels so that every unique
    # value for (category, instance) is assigned a unique integer label.
    pred_segment_id = self._naively_combine_labels(predicted_category_mask,
                                                   predicted_instance_mask)
    gt_segment_id = self._naively_combine_labels(groundtruth_category_mask,
                                                 groundtruth_instance_mask)

    # Pre-calculate areas for all ground-truth and predicted segments.
    gt_segment_areas = _ids_to_counts(gt_segment_id)
    pred_segment_areas = _ids_to_counts(pred_segment_id)

    # We assume there is only one void segment and it has instance id = 0.
    void_segment_id = self.ignored_label * self.max_instances_per_category

    # There may be other ignored ground-truth segments with instance id > 0,
    # find those ids using the unique segment ids extracted with the area
    # computation above.
    ignored_segment_ids = {
        gt_segment_id for gt_segment_id in gt_segment_areas
        if (gt_segment_id //
            self.max_instances_per_category) == self.ignored_label
    }

    # Next, combine the ground-truth and predicted labels. Divide up the pixels
    # based on which ground-truth segment and predicted segment they belong to,
    # this will assign a different 32-bit integer label to each choice of
    # (ground-truth segment, predicted segment), encoded as
    #   gt_segment_id * offset + pred_segment_id.
    intersection_id_array = (
        gt_segment_id.astype(np.uint64) * self.offset +
        pred_segment_id.astype(np.uint64))

    # For every combination of (ground-truth segment, predicted segment) with a
    # non-empty intersection, this counts the number of pixels in that
    # intersection.
    intersection_areas = _ids_to_counts(intersection_id_array)

    # Helper function that computes the area of the overlap between a predicted
    # segment and the ground-truth void/ignored segment.
    def prediction_void_overlap(pred_segment_id):
      void_intersection_id = void_segment_id * self.offset + pred_segment_id
      return intersection_areas.get(void_intersection_id, 0)

    # Compute overall ignored overlap.
    def prediction_ignored_overlap(pred_segment_id):
      total_ignored_overlap = 0
      for ignored_segment_id in ignored_segment_ids:
        intersection_id = ignored_segment_id * self.offset + pred_segment_id
        total_ignored_overlap += intersection_areas.get(intersection_id, 0)
      return total_ignored_overlap

    # Sets that are populated with segments which ground-truth/predicted
    # segments have been matched with overlapping predicted/ground-truth
    # segments respectively.
    gt_matched = set()
    pred_matched = set()

    # Calculate IoU per pair of intersecting segments of the same category.
    for intersection_id, intersection_area in intersection_areas.items():
      gt_segment_id = int(intersection_id // self.offset)
      pred_segment_id = int(intersection_id % self.offset)

      gt_category = int(gt_segment_id // self.max_instances_per_category)
      pred_category = int(pred_segment_id // self.max_instances_per_category)
      if gt_category != pred_category:
        continue

      # Union between the ground-truth and predicted segments being compared
      # does not include the portion of the predicted segment that consists of
      # ground-truth "void" pixels.
      union = (
          gt_segment_areas[gt_segment_id] +
          pred_segment_areas[pred_segment_id] - intersection_area -
          prediction_void_overlap(pred_segment_id))
      iou = intersection_area / union
      if iou > 0.5:
        self.tp_per_class[gt_category] += 1
        self.iou_per_class[gt_category] += iou
        gt_matched.add(gt_segment_id)
        pred_matched.add(pred_segment_id)

    # Count false negatives for each category.
    for gt_segment_id in gt_segment_areas:
      if gt_segment_id in gt_matched:
        continue
      category = gt_segment_id // self.max_instances_per_category
      # Failing to detect a void segment is not a false negative.
      if category == self.ignored_label:
        continue
      self.fn_per_class[category] += 1

    # Count false positives for each category.
    for pred_segment_id in pred_segment_areas:
      if pred_segment_id in pred_matched:
        continue
      # A false positive is not penalized if is mostly ignored in the
      # ground-truth.
      if (prediction_ignored_overlap(pred_segment_id) /
          pred_segment_areas[pred_segment_id]) > 0.5:
        continue
      category = pred_segment_id // self.max_instances_per_category
      self.fp_per_class[category] += 1

  def _valid_categories(self):
    """Categories with a "valid" value for the metric, have > 0 instances.

    We will ignore the `ignore_label` class and other classes which have
    `tp + fn + fp = 0`.

    Returns:
      Boolean array of shape `[num_categories]`.
    """
    valid_categories = np.not_equal(
        self.tp_per_class + self.fn_per_class + self.fp_per_class, 0)
    if self.ignored_label >= 0 and self.ignored_label < self.num_categories:
      valid_categories[self.ignored_label] = False
    return valid_categories

  def result_per_category(self):
    """For supported metrics, return individual per-category metric values.

    Returns:
      A dictionary contains all per-class metrics, each metrics is a numpy array
      of shape `[self.num_categories]`, where index `i` is the metrics value
      over only that category.
    """
    sq_per_class = realdiv_maybe_zero(self.iou_per_class, self.tp_per_class)
    rq_per_class = realdiv_maybe_zero(
        self.tp_per_class,
        self.tp_per_class + 0.5 * self.fn_per_class + 0.5 * self.fp_per_class)
    return {
        'sq_per_class': sq_per_class,
        'rq_per_class': rq_per_class,
        'pq_per_class': np.multiply(sq_per_class, rq_per_class)
    }

  def result(self, is_thing=None):
    """Computes and returns the detailed metric results over all comparisons.

    Args:
      is_thing: A boolean array of length `num_categories`. The entry
        `is_thing[category_id]` is True iff that category is a "thing" category
        instead of "stuff."

    Returns:
      A dictionary with a breakdown of metrics and/or metric factors by things,
      stuff, and all categories.
    """
    results = self.result_per_category()
    valid_categories = self._valid_categories()
    # If known, break down which categories are valid _and_ things/stuff.
    category_sets = collections.OrderedDict()
    category_sets['All'] = valid_categories
    if is_thing is not None:
      category_sets['Things'] = np.logical_and(valid_categories, is_thing)
      category_sets['Stuff'] = np.logical_and(valid_categories,
                                              np.logical_not(is_thing))

    for category_set_name, in_category_set in category_sets.items():
      if np.any(in_category_set):
        results.update({
            f'{category_set_name}_pq':
                np.mean(results['pq_per_class'][in_category_set]),
            f'{category_set_name}_sq':
                np.mean(results['sq_per_class'][in_category_set]),
            f'{category_set_name}_rq':
                np.mean(results['rq_per_class'][in_category_set]),
            # The number of categories in this subset.
            f'{category_set_name}_num_categories':
                np.sum(in_category_set.astype(np.int32)),
        })
      else:
        results.update({
            f'{category_set_name}_pq': 0.,
            f'{category_set_name}_sq': 0.,
            f'{category_set_name}_rq': 0.,
            f'{category_set_name}_num_categories': 0
        })

    return results

  def reset(self):
    """Resets the accumulation to the metric class's state at initialization."""
    self.iou_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.tp_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fn_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fp_per_class = np.zeros(self.num_categories, dtype=np.float64)


def _get_instance_class_ids(
    category_mask: tf.Tensor,
    instance_mask: tf.Tensor,
    max_num_instances: int,
    ignored_label: int,
) -> tf.Tensor:
  """Get the class id of each instance (index starts from 1)."""
  # (batch_size, height, width)
  instance_mask = tf.where(
      (instance_mask == 0) | (category_mask == ignored_label), -1, instance_mask
  )
  # (batch_size, height, width, max_num_instances + 1)
  instance_binary_mask = tf.one_hot(
      instance_mask, max_num_instances + 1, dtype=tf.int32
  )
  # (batch_size, max_num_instances + 1)
  result = tf.reduce_max(
      instance_binary_mask * category_mask[..., tf.newaxis], axis=[1, 2]
  )
  # If not an instance, sets the class id to -1.
  return tf.where(result == 0, -1, result)


class PanopticQualityV2(tf_keras.metrics.Metric):
  """Panoptic quality metrics with vectorized implementation.

  This implementation is supported on TPU.

  "Panoptic Segmentation" by Alexander Kirillov, Kaiming He, Ross Girshick,
  Carsten Rother, Piotr Dollar.
  https://arxiv.org/abs/1801.00868
  """

  def __init__(
      self,
      num_categories: int,
      is_thing: Optional[Tuple[bool, ...]] = None,
      max_num_instances: int = 255,
      ignored_label: int = 255,
      rescale_predictions: bool = False,
      name: Optional[str] = None,
      dtype: Optional[Union[str, tf.dtypes.DType]] = tf.float32,
  ):
    """Initialization for PanopticQualityV2.

    Args:
      num_categories: the number of categories.
      is_thing: a boolean array of length `num_categories`. The entry
        `is_thing[category_id]` is True iff that category is a "thing" category
        instead of "stuff". Default to `None`, and it means categories are not
        classified into these two categories.
      max_num_instances: the maximum number of instances in an image.
      ignored_label: a category id that is ignored in evaluation, e.g. the void
        label as defined in COCO panoptic segmentation dataset.
      rescale_predictions: whether to scale back prediction to original image
        sizes. If True, the image_info of the groundtruth is used to rescale
        predictions.
      name: string name of the metric instance.
      dtype: data type of the metric result.
    """
    super().__init__(name=name, dtype=dtype)

    self._num_categories = num_categories
    if is_thing is not None:
      self._is_thing = is_thing
    else:
      self._is_thing = [True] * self._num_categories
    self._max_num_instances = max_num_instances
    self._ignored_label = ignored_label
    self._rescale_predictions = rescale_predictions

    # Variables
    self.tp_count = self.add_weight(
        'tp_count',
        shape=[self._num_categories],
        initializer='zeros',
        dtype=tf.float32,
    )
    self.fp_count = self.add_weight(
        'fp_count',
        shape=[self._num_categories],
        initializer='zeros',
        dtype=tf.float32,
    )
    self.fn_count = self.add_weight(
        'fn_count',
        shape=[self._num_categories],
        initializer='zeros',
        dtype=tf.float32,
    )
    self.tp_iou_sum = self.add_weight(
        'tp_iou_sum',
        shape=[self._num_categories],
        initializer='zeros',
        dtype=tf.float32,
    )

  def get_config(self) -> Dict[str, Any]:
    """Returns the serializable config of the metric."""
    return {
        'num_categories': self._num_categories,
        'is_thing': self._is_thing,
        'max_num_instances': self._max_num_instances,
        'ignored_label': self._ignored_label,
        'rescale_predictions': self._rescale_predictions,
        'name': self.name,
        'dtype': self.dtype,
    }

  def reset_state(self):
    """Resets all of the metric state variables."""
    self.tp_count.assign(tf.zeros_like(self.tp_count))
    self.fp_count.assign(tf.zeros_like(self.fp_count))
    self.fn_count.assign(tf.zeros_like(self.fn_count))
    self.tp_iou_sum.assign(tf.zeros_like(self.tp_iou_sum))

  def update_state(
      self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]
  ):
    category_mask = tf.convert_to_tensor(y_pred['category_mask'], tf.int32)
    instance_mask = tf.convert_to_tensor(y_pred['instance_mask'], tf.int32)
    gt_category_mask = tf.convert_to_tensor(y_true['category_mask'], tf.int32)
    gt_instance_mask = tf.convert_to_tensor(y_true['instance_mask'], tf.int32)

    if self._rescale_predictions:
      _, height, width = gt_category_mask.get_shape().as_list()
      # Instead of cropping the masks to the original image shape (dynamic),
      # here we keep the mask shape (fixed) and ignore the pixels outside the
      # original image shape.
      image_shape = tf.cast(y_true['image_info'][:, 0, :], tf.int32)
      # (batch_size, 2)
      y0_x0 = tf.broadcast_to(
          tf.constant([[0, 0]], dtype=tf.int32), tf.shape(image_shape)
      )
      # (batch_size, 4)
      image_shape_bbox = tf.concat([y0_x0, image_shape], axis=1)
      # (batch_size, height, width)
      image_shape_masks = box_ops.bbox2mask(
          bbox=image_shape_bbox,
          image_height=height,
          image_width=width,
          dtype=tf.bool,
      )
      # (batch_size, height, width)
      category_mask = tf.where(
          image_shape_masks, category_mask, self._ignored_label
      )
      instance_mask = tf.where(image_shape_masks, instance_mask, 0)
      gt_category_mask = tf.where(
          image_shape_masks, gt_category_mask, self._ignored_label
      )
      gt_instance_mask = tf.where(image_shape_masks, gt_instance_mask, 0)

    self._update_thing_classes(
        category_mask, instance_mask, gt_category_mask, gt_instance_mask
    )
    self._update_stuff_classes(category_mask, gt_category_mask)

  def _update_thing_classes(
      self,
      category_mask: tf.Tensor,
      instance_mask: tf.Tensor,
      gt_category_mask: tf.Tensor,
      gt_instance_mask: tf.Tensor,
  ):
    _, height, width = category_mask.get_shape().as_list()

    # (batch_size, num_detections + 1)
    instance_class_ids = _get_instance_class_ids(
        category_mask,
        instance_mask,
        self._max_num_instances,
        self._ignored_label,
    )
    # (batch_size, num_gts + 1)
    gt_instance_class_ids = _get_instance_class_ids(
        gt_category_mask,
        gt_instance_mask,
        self._max_num_instances,
        self._ignored_label,
    )

    # (batch_size, height, width)
    valid_mask = gt_category_mask != self._ignored_label

    # (batch_size, height, width, num_detections + 1)
    instance_binary_masks = tf.one_hot(
        tf.where(instance_mask > 0, instance_mask, -1),
        self._max_num_instances + 1,
        on_value=True,
        off_value=False,
    )
    # (batch_size, height, width, num_gts + 1)
    gt_instance_binary_masks = tf.one_hot(
        tf.where(gt_instance_mask > 0, gt_instance_mask, -1),
        self._max_num_instances + 1,
        on_value=True,
        off_value=False,
    )

    # (batch_size, height * width, num_detections + 1)
    flattened_binary_masks = tf.reshape(
        instance_binary_masks & valid_mask[..., tf.newaxis],
        [-1, height * width, self._max_num_instances + 1],
    )
    # (batch_size, height * width, num_gts + 1)
    flattened_gt_binary_masks = tf.reshape(
        gt_instance_binary_masks & valid_mask[..., tf.newaxis],
        [-1, height * width, self._max_num_instances + 1],
    )
    # (batch_size, num_detections + 1, height * width)
    flattened_binary_masks = tf.transpose(flattened_binary_masks, [0, 2, 1])
    # (batch_size, num_detections + 1, num_gts + 1)
    intersection = tf.matmul(
        tf.cast(flattened_binary_masks, tf.float32),
        tf.cast(flattened_gt_binary_masks, tf.float32),
    )
    union = (
        tf.math.count_nonzero(
            flattened_binary_masks, axis=-1, keepdims=True, dtype=tf.float32
        )
        + tf.math.count_nonzero(
            flattened_gt_binary_masks, axis=-2, keepdims=True, dtype=tf.float32
        )
        - intersection
    )
    # (batch_size, num_detections + 1, num_gts + 1)
    detection_to_gt_ious = tf.math.divide_no_nan(intersection, union)
    detection_matches_gt = (
        (detection_to_gt_ious > 0.5)
        & (
            instance_class_ids[:, :, tf.newaxis]
            == gt_instance_class_ids[:, tf.newaxis, :]
        )
        & (gt_instance_class_ids[:, tf.newaxis, :] > 0)
    )

    # (batch_size, num_gts + 1)
    is_tp = tf.reduce_any(detection_matches_gt, axis=1)
    # (batch_size, num_gts + 1)
    tp_iou = tf.reduce_max(
        tf.where(detection_matches_gt, detection_to_gt_ious, 0), axis=1
    )

    # (batch_size, num_detections + 1)
    is_fp = tf.reduce_any(instance_binary_masks, axis=[1, 2]) & ~tf.reduce_any(
        detection_matches_gt, axis=2
    )
    # (batch_size, height, width, num_detections + 1)
    fp_binary_mask = is_fp[:, tf.newaxis, tf.newaxis, :] & instance_binary_masks
    # (batch_size, num_detections + 1)
    fp_area = tf.math.count_nonzero(
        fp_binary_mask, axis=[1, 2], dtype=tf.float32
    )
    # (batch_size, num_detections + 1)
    fp_crowd_or_ignored_area = tf.math.count_nonzero(
        fp_binary_mask
        & (
            (
                # An instance detection matches a crowd ground truth instance if
                # the instance class of the detection matches the class of the
                # ground truth and the instance id of the ground truth is 0 (the
                # instance is crowd).
                (instance_mask > 0)
                & (category_mask > 0)
                & (gt_category_mask == category_mask)
                & (gt_instance_mask == 0)
            )
            | (gt_category_mask == self._ignored_label)
        )[..., tf.newaxis],
        axis=[1, 2],
        dtype=tf.float32,
    )
    # Don't count the detection as false positive if over 50% pixels of the
    # instance detection are crowd of the matching class or ignored pixels in
    # ground truth.
    # (batch_size, num_detections + 1)
    is_fp &= tf.math.divide_no_nan(fp_crowd_or_ignored_area, fp_area) <= 0.5

    # (batch_size, num_detections + 1, num_categories)
    detection_by_class = tf.one_hot(
        instance_class_ids, self._num_categories, on_value=True, off_value=False
    )
    # (batch_size, num_gts + 1, num_categories)
    gt_by_class = tf.one_hot(
        gt_instance_class_ids,
        self._num_categories,
        on_value=True,
        off_value=False,
    )

    # (num_categories,)
    gt_count = tf.math.count_nonzero(gt_by_class, axis=[0, 1], dtype=tf.float32)
    tp_count = tf.math.count_nonzero(
        is_tp[..., tf.newaxis] & gt_by_class, axis=[0, 1], dtype=tf.float32
    )
    fn_count = gt_count - tp_count
    fp_count = tf.math.count_nonzero(
        is_fp[..., tf.newaxis] & detection_by_class,
        axis=[0, 1],
        dtype=tf.float32,
    )
    tp_iou_sum = tf.reduce_sum(
        tf.cast(gt_by_class, tf.float32) * tp_iou[..., tf.newaxis], axis=[0, 1]
    )

    self.tp_count.assign_add(tp_count)
    self.fn_count.assign_add(fn_count)
    self.fp_count.assign_add(fp_count)
    self.tp_iou_sum.assign_add(tp_iou_sum)

  def _update_stuff_classes(
      self, category_mask: tf.Tensor, gt_category_mask: tf.Tensor
  ):
    # (batch_size, height, width, num_categories)
    category_binary_mask = tf.one_hot(
        category_mask, self._num_categories, on_value=True, off_value=False
    )
    gt_category_binary_mask = tf.one_hot(
        gt_category_mask, self._num_categories, on_value=True, off_value=False
    )

    # (batch_size, height, width)
    valid_mask = gt_category_mask != self._ignored_label

    # (batch_size, num_categories)
    intersection = tf.math.count_nonzero(
        category_binary_mask
        & gt_category_binary_mask
        & valid_mask[..., tf.newaxis],
        axis=[1, 2],
        dtype=tf.float32,
    )
    union = tf.math.count_nonzero(
        (category_binary_mask | gt_category_binary_mask)
        & valid_mask[..., tf.newaxis],
        axis=[1, 2],
        dtype=tf.float32,
    )
    iou = tf.math.divide_no_nan(intersection, union)

    is_thing = tf.constant(self._is_thing, dtype=tf.bool)
    # (batch_size, num_categories)
    is_tp = (iou > 0.5) & ~is_thing
    is_fn = (
        tf.reduce_any(gt_category_binary_mask, axis=[1, 2]) & ~is_thing & ~is_tp
    )
    is_fp = (
        tf.reduce_any(category_binary_mask, axis=[1, 2]) & ~is_thing & ~is_tp
    )

    # (batch_size, height, width, num_categories)
    fp_binary_mask = is_fp[:, tf.newaxis, tf.newaxis, :] & category_binary_mask
    # (batch_size, num_categories)
    fp_area = tf.math.count_nonzero(
        fp_binary_mask, axis=[1, 2], dtype=tf.float32
    )
    fp_ignored_area = tf.math.count_nonzero(
        fp_binary_mask
        & (gt_category_mask == self._ignored_label)[..., tf.newaxis],
        axis=[1, 2],
        dtype=tf.float32,
    )
    # Don't count the detection as false positive if over 50% pixels of the
    # stuff detection are ignored pixels in ground truth.
    is_fp &= tf.math.divide_no_nan(fp_ignored_area, fp_area) <= 0.5

    # (num_categories,)
    tp_count = tf.math.count_nonzero(is_tp, axis=0, dtype=tf.float32)
    fn_count = tf.math.count_nonzero(is_fn, axis=0, dtype=tf.float32)
    fp_count = tf.math.count_nonzero(is_fp, axis=0, dtype=tf.float32)
    tp_iou_sum = tf.reduce_sum(tf.cast(is_tp, tf.float32) * iou, axis=0)

    self.tp_count.assign_add(tp_count)
    self.fn_count.assign_add(fn_count)
    self.fp_count.assign_add(fp_count)
    self.tp_iou_sum.assign_add(tp_iou_sum)

  def result(self) -> Dict[str, tf.Tensor]:
    """Returns the metrics values as a dict."""
    # (num_categories,)
    tp_fn_fp_count = self.tp_count + self.fn_count + self.fp_count
    is_ignore_label = tf.one_hot(
        self._ignored_label,
        self._num_categories,
        on_value=True,
        off_value=False,
    )

    sq_per_class = tf.math.divide_no_nan(
        self.tp_iou_sum, self.tp_count
    ) * tf.cast(~is_ignore_label, tf.float32)
    rq_per_class = tf.math.divide_no_nan(
        self.tp_count, self.tp_count + 0.5 * self.fp_count + 0.5 * self.fn_count
    ) * tf.cast(~is_ignore_label, tf.float32)
    pq_per_class = sq_per_class * rq_per_class
    is_thing = tf.constant(self._is_thing, dtype=tf.bool)

    result = {
        # (num_categories,)
        'valid_thing_classes': (
            (tp_fn_fp_count > 0) & is_thing & ~is_ignore_label
        ),
        # (num_categories,)
        'valid_stuff_classes': (
            (tp_fn_fp_count > 0) & ~is_thing & ~is_ignore_label
        ),
        # (num_categories,)
        'sq_per_class': sq_per_class,
        # (num_categories,)
        'rq_per_class': rq_per_class,
        # (num_categories,)
        'pq_per_class': pq_per_class,
    }
    return result
