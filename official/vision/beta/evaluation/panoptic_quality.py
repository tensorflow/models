# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np

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
        dataset.
      ignored_label: A category id that is ignored in evaluation, e.g. the void
        label as defined in COCO panoptic segmentation dataset.
      max_instances_per_category: The maximum number of instances for each
        category. Used in ensuring unique instance labels.
      offset: The maximum number of unique labels. This is used, by multiplying
        the ground-truth labels, to generate unique ids for individual regions
        of overlap between groundtruth and predicted segments.
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
    """Compares predicted segmentation with groundtruth, accumulates its metric.

    It is not assumed that instance ids are unique across different categories.
    See for example combine_semantic_and_instance_predictions.py in official
    PanopticAPI evaluation code for issues to consider when fusing category
    and instance labels.

    Instances ids of the ignored category have the meaning that id 0 is "void"
    and remaining ones are crowd instances.

    Args:
      groundtruths: A dictionary contains groundtruth labels. It should contain
        the following fields.
        - category_mask: A 2D numpy uint16 array of groundtruth per-pixel
          category labels.
        - instance_mask: A 2D numpy uint16 array of groundtruth instance labels.
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

    # Pre-calculate areas for all groundtruth and predicted segments.
    gt_segment_areas = _ids_to_counts(gt_segment_id)
    pred_segment_areas = _ids_to_counts(pred_segment_id)

    # We assume there is only one void segment and it has instance id = 0.
    void_segment_id = self.ignored_label * self.max_instances_per_category

    # There may be other ignored groundtruth segments with instance id > 0, find
    # those ids using the unique segment ids extracted with the area computation
    # above.
    ignored_segment_ids = {
        gt_segment_id for gt_segment_id in gt_segment_areas
        if (gt_segment_id //
            self.max_instances_per_category) == self.ignored_label
    }

    # Next, combine the groundtruth and predicted labels. Dividing up the pixels
    # based on which groundtruth segment and which predicted segment they belong
    # to, this will assign a different 32-bit integer label to each choice
    # of (groundtruth segment, predicted segment), encoded as
    #   gt_segment_id * offset + pred_segment_id.
    intersection_id_array = (
        gt_segment_id.astype(np.uint64) * self.offset +
        pred_segment_id.astype(np.uint64))

    # For every combination of (groundtruth segment, predicted segment) with a
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

    # Sets that are populated with which segments groundtruth/predicted segments
    # have been matched with overlapping predicted/groundtruth segments
    # respectively.
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

      # Union between the groundtruth and predicted segments being compared does
      # not include the portion of the predicted segment that consists of
      # groundtruth "void" pixels.
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
      # groundtruth.
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
        results[category_set_name] = {
            f'{category_set_name}_pq': 0.,
            f'{category_set_name}_sq': 0.,
            f'{category_set_name}_rq': 0.,
            f'{category_set_name}_num_categories': 0
        }

    return results

  def reset(self):
    """Resets the accumulation to the metric class's state at initialization."""
    self.iou_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.tp_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fn_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fp_per_class = np.zeros(self.num_categories, dtype=np.float64)
