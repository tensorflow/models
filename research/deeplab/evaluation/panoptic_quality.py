# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Implementation of the Panoptic Quality metric.

Panoptic Quality is an instance-based metric for evaluating the task of
image parsing, aka panoptic segmentation.

Please see the paper for details:
"Panoptic Segmentation", Alexander Kirillov, Kaiming He, Ross Girshick,
Carsten Rother and Piotr Dollar. arXiv:1801.00868, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import prettytable
import six

from deeplab.evaluation import base_metric


def _ids_to_counts(id_array):
  """Given a numpy array, a mapping from each unique entry to its count."""
  ids, counts = np.unique(id_array, return_counts=True)
  return dict(six.moves.zip(ids, counts))


class PanopticQuality(base_metric.SegmentationMetric):
  """Metric class for Panoptic Quality.

  "Panoptic Segmentation" by Alexander Kirillov, Kaiming He, Ross Girshick,
  Carsten Rother, Piotr Dollar.
  https://arxiv.org/abs/1801.00868
  """

  def compare_and_accumulate(
      self, groundtruth_category_array, groundtruth_instance_array,
      predicted_category_array, predicted_instance_array):
    """See base class."""
    # First, combine the category and instance labels so that every unique
    # value for (category, instance) is assigned a unique integer label.
    pred_segment_id = self._naively_combine_labels(predicted_category_array,
                                                   predicted_instance_array)
    gt_segment_id = self._naively_combine_labels(groundtruth_category_array,
                                                 groundtruth_instance_array)

    # Pre-calculate areas for all groundtruth and predicted segments.
    gt_segment_areas = _ids_to_counts(gt_segment_id)
    pred_segment_areas = _ids_to_counts(pred_segment_id)

    # We assume there is only one void segment and it has instance id = 0.
    void_segment_id = self.ignored_label * self.max_instances_per_category

    # There may be other ignored groundtruth segments with instance id > 0, find
    # those ids using the unique segment ids extracted with the area computation
    # above.
    ignored_segment_ids = {
        gt_segment_id for gt_segment_id in six.iterkeys(gt_segment_areas)
        if (gt_segment_id //
            self.max_instances_per_category) == self.ignored_label
    }

    # Next, combine the groundtruth and predicted labels. Dividing up the pixels
    # based on which groundtruth segment and which predicted segment they belong
    # to, this will assign a different 32-bit integer label to each choice
    # of (groundtruth segment, predicted segment), encoded as
    #   gt_segment_id * offset + pred_segment_id.
    intersection_id_array = (
        gt_segment_id.astype(np.uint32) * self.offset +
        pred_segment_id.astype(np.uint32))

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
    for intersection_id, intersection_area in six.iteritems(intersection_areas):
      gt_segment_id = intersection_id // self.offset
      pred_segment_id = intersection_id % self.offset

      gt_category = gt_segment_id // self.max_instances_per_category
      pred_category = pred_segment_id // self.max_instances_per_category
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
    for gt_segment_id in six.iterkeys(gt_segment_areas):
      if gt_segment_id in gt_matched:
        continue
      category = gt_segment_id // self.max_instances_per_category
      # Failing to detect a void segment is not a false negative.
      if category == self.ignored_label:
        continue
      self.fn_per_class[category] += 1

    # Count false positives for each category.
    for pred_segment_id in six.iterkeys(pred_segment_areas):
      if pred_segment_id in pred_matched:
        continue
      # A false positive is not penalized if is mostly ignored in the
      # groundtruth.
      if (prediction_ignored_overlap(pred_segment_id) /
          pred_segment_areas[pred_segment_id]) > 0.5:
        continue
      category = pred_segment_id // self.max_instances_per_category
      self.fp_per_class[category] += 1

    return self.result()

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

  def detailed_results(self, is_thing=None):
    """See base class."""
    valid_categories = self._valid_categories()

    # If known, break down which categories are valid _and_ things/stuff.
    category_sets = collections.OrderedDict()
    category_sets['All'] = valid_categories
    if is_thing is not None:
      category_sets['Things'] = np.logical_and(valid_categories, is_thing)
      category_sets['Stuff'] = np.logical_and(valid_categories,
                                              np.logical_not(is_thing))

    # Compute individual per-class metrics that constitute factors of PQ.
    sq = base_metric.realdiv_maybe_zero(self.iou_per_class, self.tp_per_class)
    rq = base_metric.realdiv_maybe_zero(
        self.tp_per_class,
        self.tp_per_class + 0.5 * self.fn_per_class + 0.5 * self.fp_per_class)
    pq = np.multiply(sq, rq)

    # Assemble detailed results dictionary.
    results = {}
    for category_set_name, in_category_set in six.iteritems(category_sets):
      if np.any(in_category_set):
        results[category_set_name] = {
            'pq': np.mean(pq[in_category_set]),
            'sq': np.mean(sq[in_category_set]),
            'rq': np.mean(rq[in_category_set]),
            # The number of categories in this subset.
            'n': np.sum(in_category_set.astype(np.int32)),
        }
      else:
        results[category_set_name] = {'pq': 0, 'sq': 0, 'rq': 0, 'n': 0}

    return results

  def result_per_category(self):
    """See base class."""
    sq = base_metric.realdiv_maybe_zero(self.iou_per_class, self.tp_per_class)
    rq = base_metric.realdiv_maybe_zero(
        self.tp_per_class,
        self.tp_per_class + 0.5 * self.fn_per_class + 0.5 * self.fp_per_class)
    return np.multiply(sq, rq)

  def print_detailed_results(self, is_thing=None, print_digits=3):
    """See base class."""
    results = self.detailed_results(is_thing=is_thing)

    tab = prettytable.PrettyTable()

    tab.add_column('', [], align='l')
    for fieldname in ['PQ', 'SQ', 'RQ', 'N']:
      tab.add_column(fieldname, [], align='r')

    for category_set, subset_results in six.iteritems(results):
      data_cols = [
          round(subset_results[col_key], print_digits) * 100
          for col_key in ['pq', 'sq', 'rq']
      ]
      data_cols += [subset_results['n']]
      tab.add_row([category_set] + data_cols)

    print(tab)

  def result(self):
    """See base class."""
    pq_per_class = self.result_per_category()
    valid_categories = self._valid_categories()
    if not np.any(valid_categories):
      return 0.
    return np.mean(pq_per_class[valid_categories])

  def merge(self, other_instance):
    """See base class."""
    self.iou_per_class += other_instance.iou_per_class
    self.tp_per_class += other_instance.tp_per_class
    self.fn_per_class += other_instance.fn_per_class
    self.fp_per_class += other_instance.fp_per_class

  def reset(self):
    """See base class."""
    self.iou_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.tp_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fn_per_class = np.zeros(self.num_categories, dtype=np.float64)
    self.fp_per_class = np.zeros(self.num_categories, dtype=np.float64)
