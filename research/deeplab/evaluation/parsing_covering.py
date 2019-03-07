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
"""Implementation of the Parsing Covering metric.

Parsing Covering is a region-based metric for evaluating the task of
image parsing, aka panoptic segmentation.

Please see the paper for details:
"DeeperLab: Single-Shot Image Parser", Tien-Ju Yang, Maxwell D. Collins,
Yukun Zhu, Jyh-Jing Hwang, Ting Liu, Xiao Zhang, Vivienne Sze,
George Papandreou, Liang-Chieh Chen. arXiv: 1902.05093, 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import prettytable
import six

from deeplab.evaluation import base_metric


class ParsingCovering(base_metric.SegmentationMetric):
  r"""Metric class for Parsing Covering.

  Computes segmentation covering metric introduced in (Arbelaez, et al., 2010)
  with extension to handle multi-class semantic labels (a.k.a. parsing
  covering). Specifically, segmentation covering (SC) is defined in Eq. (8) in
  (Arbelaez et al., 2010) as:

  SC(c) = \sum_{R\in S}(|R| * \max_{R'\in S'}O(R,R')) / \sum_{R\in S}|R|,

  where S are the groundtruth instance regions and S' are the predicted
  instance regions. The parsing covering is simply:

  PC = \sum_{c=1}^{C}SC(c) / C,

  where C is the number of classes.
  """

  def __init__(self,
               num_categories,
               ignored_label,
               max_instances_per_category,
               offset,
               normalize_by_image_size=True):
    """Initialization for ParsingCovering.

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
      normalize_by_image_size: Whether to normalize groundtruth instance region
        areas by image size. If True, groundtruth instance areas and weighted
        IoUs will be divided by the size of the corresponding image before
        accumulated across the dataset.
    """
    super(ParsingCovering, self).__init__(num_categories, ignored_label,
                                          max_instances_per_category, offset)
    self.normalize_by_image_size = normalize_by_image_size

  def compare_and_accumulate(
      self, groundtruth_category_array, groundtruth_instance_array,
      predicted_category_array, predicted_instance_array):
    """See base class."""
    # Allocate intermediate data structures.
    max_ious = np.zeros([self.num_categories, self.max_instances_per_category],
                        dtype=np.float64)
    gt_areas = np.zeros([self.num_categories, self.max_instances_per_category],
                        dtype=np.float64)
    pred_areas = np.zeros(
        [self.num_categories, self.max_instances_per_category],
        dtype=np.float64)
    # This is a dictionary in the format:
    #   {(category, gt_instance): [(pred_instance, intersection_area)]}.
    intersections = collections.defaultdict(list)

    # First, combine the category and instance labels so that every unique
    # value for (category, instance) is assigned a unique integer label.
    pred_segment_id = self._naively_combine_labels(predicted_category_array,
                                                   predicted_instance_array)
    gt_segment_id = self._naively_combine_labels(groundtruth_category_array,
                                                 groundtruth_instance_array)

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
    intersection_ids, intersection_areas = np.unique(
        intersection_id_array, return_counts=True)

    # Find areas of all groundtruth and predicted instances, as well as of their
    # intersections.
    for intersection_id, intersection_area in six.moves.zip(
        intersection_ids, intersection_areas):
      gt_segment_id = intersection_id // self.offset
      gt_category = gt_segment_id // self.max_instances_per_category
      if gt_category == self.ignored_label:
        continue
      gt_instance = gt_segment_id % self.max_instances_per_category
      gt_areas[gt_category, gt_instance] += intersection_area

      pred_segment_id = intersection_id % self.offset
      pred_category = pred_segment_id // self.max_instances_per_category
      pred_instance = pred_segment_id % self.max_instances_per_category
      pred_areas[pred_category, pred_instance] += intersection_area
      if pred_category != gt_category:
        continue

      intersections[gt_category, gt_instance].append((pred_instance,
                                                      intersection_area))

    # Find maximum IoU for every groundtruth instance.
    for gt_label, instance_intersections in six.iteritems(intersections):
      category, gt_instance = gt_label
      gt_area = gt_areas[category, gt_instance]
      ious = []
      for pred_instance, intersection_area in instance_intersections:
        pred_area = pred_areas[category, pred_instance]
        union = gt_area + pred_area - intersection_area
        ious.append(intersection_area / union)
      max_ious[category, gt_instance] = max(ious)

    # Normalize groundtruth instance areas by image size if necessary.
    if self.normalize_by_image_size:
      gt_areas /= groundtruth_category_array.size

    # Compute per-class weighted IoUs and areas summed over all groundtruth
    # instances.
    self.weighted_iou_per_class += np.sum(max_ious * gt_areas, axis=-1)
    self.gt_area_per_class += np.sum(gt_areas, axis=-1)

    return self.result()

  def result_per_category(self):
    """See base class."""
    return base_metric.realdiv_maybe_zero(self.weighted_iou_per_class,
                                          self.gt_area_per_class)

  def _valid_categories(self):
    """Categories with a "valid" value for the metric, have > 0 instances.

    We will ignore the `ignore_label` class and other classes which have
    groundtruth area of 0.

    Returns:
      Boolean array of shape `[num_categories]`.
    """
    valid_categories = np.not_equal(self.gt_area_per_class, 0)
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

    covering_per_class = self.result_per_category()
    results = {}
    for category_set_name, in_category_set in six.iteritems(category_sets):
      if np.any(in_category_set):
        results[category_set_name] = {
            'pc': np.mean(covering_per_class[in_category_set]),
            # The number of valid categories in this subset.
            'n': np.sum(in_category_set.astype(np.int32)),
        }
      else:
        results[category_set_name] = {'pc': 0, 'n': 0}

    return results

  def print_detailed_results(self, is_thing=None, print_digits=3):
    """See base class."""
    results = self.detailed_results(is_thing=is_thing)

    tab = prettytable.PrettyTable()

    tab.add_column('', [], align='l')
    for fieldname in ['PC', 'N']:
      tab.add_column(fieldname, [], align='r')

    for category_set, subset_results in six.iteritems(results):
      data_cols = [
          round(subset_results['pc'], print_digits) * 100, subset_results['n']
      ]
      tab.add_row([category_set] + data_cols)

    print(tab)

  def result(self):
    """See base class."""
    covering_per_class = self.result_per_category()
    valid_categories = self._valid_categories()
    if not np.any(valid_categories):
      return 0.
    return np.mean(covering_per_class[valid_categories])

  def merge(self, other_instance):
    """See base class."""
    self.weighted_iou_per_class += other_instance.weighted_iou_per_class
    self.gt_area_per_class += other_instance.gt_area_per_class

  def reset(self):
    """See base class."""
    self.weighted_iou_per_class = np.zeros(
        self.num_categories, dtype=np.float64)
    self.gt_area_per_class = np.zeros(self.num_categories, dtype=np.float64)
