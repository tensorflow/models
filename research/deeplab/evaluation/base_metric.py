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
"""Defines the top-level interface for evaluating segmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six


_EPSILON = 1e-10


def realdiv_maybe_zero(x, y):
  """Element-wise x / y where y may contain zeros, for those returns 0 too."""
  return np.where(
      np.less(np.abs(y), _EPSILON), np.zeros_like(x), np.divide(x, y))


@six.add_metaclass(abc.ABCMeta)
class SegmentationMetric(object):
  """Abstract base class for computers of segmentation metrics.

  Subclasses will implement both:
  1. Comparing the predicted segmentation for an image with the groundtruth.
  2. Computing the final metric over a set of images.
  These are often done as separate steps, due to the need to accumulate
  intermediate values other than the metric itself across images, computing the
  actual metric value only on these accumulations after all the images have been
  compared.

  A simple usage would be:

    metric = MetricImplementation(...)
    for <image>, <groundtruth> in evaluation_set:
      <prediction> = run_segmentation(<image>)
      metric.compare_and_accumulate(<prediction>, <groundtruth>)
    print(metric.result())

  """

  def __init__(self, num_categories, ignored_label, max_instances_per_category,
               offset):
    """Base initialization for SegmentationMetric.

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

  def _naively_combine_labels(self, category_array, instance_array):
    """Naively creates a combined label array from categories and instances."""
    return (category_array.astype(np.uint32) * self.max_instances_per_category +
            instance_array.astype(np.uint32))

  @abc.abstractmethod
  def compare_and_accumulate(
      self, groundtruth_category_array, groundtruth_instance_array,
      predicted_category_array, predicted_instance_array):
    """Compares predicted segmentation with groundtruth, accumulates its metric.

    It is not assumed that instance ids are unique across different categories.
    See for example combine_semantic_and_instance_predictions.py in official
    PanopticAPI evaluation code for issues to consider when fusing category
    and instance labels.

    Instances ids of the ignored category have the meaning that id 0 is "void"
    and remaining ones are crowd instances.

    Args:
      groundtruth_category_array: A 2D numpy uint16 array of groundtruth
        per-pixel category labels.
      groundtruth_instance_array: A 2D numpy uint16 array of groundtruth
        instance labels.
      predicted_category_array: A 2D numpy uint16 array of predicted per-pixel
        category labels.
      predicted_instance_array: A 2D numpy uint16 array of predicted instance
        labels.

    Returns:
      The value of the metric over all comparisons done so far, including this
      one, as a float scalar.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def result(self):
    """Computes the metric over all comparisons done so far."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def detailed_results(self, is_thing=None):
    """Computes and returns the detailed final metric results.

    Args:
      is_thing: A boolean array of length `num_categories`. The entry
        `is_thing[category_id]` is True iff that category is a "thing" category
        instead of "stuff."

    Returns:
      A dictionary with a breakdown of metrics and/or metric factors by things,
      stuff, and all categories.
    """
    raise NotImplementedError('Not implemented in subclasses.')

  @abc.abstractmethod
  def result_per_category(self):
    """For supported metrics, return individual per-category metric values.

    Returns:
      A numpy array of shape `[self.num_categories]`, where index `i` is the
      metrics value over only that category.
    """
    raise NotImplementedError('Not implemented in subclass.')

  def print_detailed_results(self, is_thing=None, print_digits=3):
    """Prints out a detailed breakdown of metric results.

    Args:
      is_thing: A boolean array of length num_categories.
        `is_thing[category_id]` will say whether that category is a "thing"
        rather than "stuff."
      print_digits: Number of significant digits to print in computed metrics.
    """
    raise NotImplementedError('Not implemented in subclass.')

  @abc.abstractmethod
  def merge(self, other_instance):
    """Combines the accumulated results of another instance into self.

    The following two cases should put `metric_a` into an equivalent state.

    Case 1 (with merge):

      metric_a = MetricsSubclass(...)
      metric_a.compare_and_accumulate(<comparison 1>)
      metric_a.compare_and_accumulate(<comparison 2>)

      metric_b = MetricsSubclass(...)
      metric_b.compare_and_accumulate(<comparison 3>)
      metric_b.compare_and_accumulate(<comparison 4>)

      metric_a.merge(metric_b)

    Case 2 (without merge):

      metric_a = MetricsSubclass(...)
      metric_a.compare_and_accumulate(<comparison 1>)
      metric_a.compare_and_accumulate(<comparison 2>)
      metric_a.compare_and_accumulate(<comparison 3>)
      metric_a.compare_and_accumulate(<comparison 4>)

    Args:
      other_instance: Another compatible instance of the same metric subclass.
    """
    raise NotImplementedError('Not implemented in subclass.')

  @abc.abstractmethod
  def reset(self):
    """Resets the accumulation to the metric class's state at initialization.

    Note that this function will be called in SegmentationMetric.__init__.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
