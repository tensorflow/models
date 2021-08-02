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

"""The panoptic quality evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = PanopticQualityEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update_state(groundtruths, predictions)
    evaluator.result()  # finish one full eval and reset states.

See also: https://github.com/cocodataset/cocoapi/
"""

import numpy as np
import tensorflow as tf

from official.vision.beta.evaluation import panoptic_quality


class PanopticQualityEvaluator:
  """Panoptic Quality metric class."""

  def __init__(self, num_categories, ignored_label, max_instances_per_category,
               offset, is_thing=None):
    """Constructs Panoptic Quality evaluation class.

    The class provides the interface to Panoptic Quality metrics_fn.

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
      is_thing: A boolean array of length `num_categories`. The entry
        `is_thing[category_id]` is True iff that category is a "thing" category
        instead of "stuff." Default to `None`, and it means categories are not
        classified into these two categories.
    """
    self._pq_metric_module = panoptic_quality.PanopticQuality(
        num_categories, ignored_label, max_instances_per_category, offset)
    self._is_thing = is_thing
    self._required_prediction_fields = ['category_mask', 'instance_mask']
    self._required_groundtruth_fields = ['category_mask', 'instance_mask']
    self.reset_states()

  @property
  def name(self):
    return 'panoptic_quality'

  def reset_states(self):
    """Resets internal states for a fresh run."""
    self._pq_metric_module.reset()

  def result(self):
    """Evaluates detection results, and reset_states."""
    results = self._pq_metric_module.result(self._is_thing)
    self.reset_states()
    return results

  def _convert_to_numpy(self, groundtruths, predictions):
    """Converts tesnors to numpy arrays."""
    if groundtruths:
      labels = tf.nest.map_structure(lambda x: x.numpy(), groundtruths)
      numpy_groundtruths = {}
      for key, val in labels.items():
        if isinstance(val, tuple):
          val = np.concatenate(val)
        numpy_groundtruths[key] = val
    else:
      numpy_groundtruths = groundtruths

    if predictions:
      outputs = tf.nest.map_structure(lambda x: x.numpy(), predictions)
      numpy_predictions = {}
      for key, val in outputs.items():
        if isinstance(val, tuple):
          val = np.concatenate(val)
        numpy_predictions[key] = val
    else:
      numpy_predictions = predictions

    return numpy_groundtruths, numpy_predictions

  def update_state(self, groundtruths, predictions):
    """Update and aggregate detection results and groundtruth data.

    Args:
      groundtruths: a dictionary of Tensors including the fields below. See also
        different parsers under `../dataloader` for more details.
        Required fields:
          - category_mask: a numpy array of uint16 of shape [batch_size, H, W].
          - instance_mask: a numpy array of uint16 of shape [batch_size, H, W].
      predictions: a dictionary of tensors including the fields below. See
        different parsers under `../dataloader` for more details.
        Required fields:
          - category_mask: a numpy array of uint16 of shape [batch_size, H, W].
          - instance_mask: a numpy array of uint16 of shape [batch_size, H, W].

    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    groundtruths, predictions = self._convert_to_numpy(groundtruths,
                                                       predictions)
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError(
            'Missing the required key `{}` in predictions!'.format(k))

    for k in self._required_groundtruth_fields:
      if k not in groundtruths:
        raise ValueError(
            'Missing the required key `{}` in groundtruths!'.format(k))

    self._pq_metric_module.compare_and_accumulate(groundtruths, predictions)
