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


def rescale_masks(groundtruth, prediction, image_info):
  rescale_size = tf.cast(
      tf.math.ceil(image_info[1, :] / image_info[2, :]), tf.int32)
  image_shape = tf.cast(image_info[0, :], tf.int32)
  offsets = tf.cast(image_info[3, :], tf.int32)

  prediction = tf.image.resize(
      tf.expand_dims(prediction, axis=-1),
      rescale_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  prediction = tf.image.crop_to_bounding_box(
      prediction,
      offsets[0], offsets[1],
      image_shape[0],
      image_shape[1])
  groundtruth = tf.image.crop_to_bounding_box(
      tf.expand_dims(groundtruth, axis=-1), 0, 0,
      image_shape[0], image_shape[1])

  return (
    tf.expand_dims(groundtruth[:, :, 0], axis=0),
    tf.expand_dims(prediction[:, :, 0], axis=0))

class PanopticQualityEvaluator:
  """Panoptic Quality metric class."""

  def __init__(self, num_categories, ignored_label, max_instances_per_category,
               offset, is_thing=None, rescale_predictions=False):
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
      rescale_predictions: `bool`, whether to scale back prediction to original
        image sizes. If True, y_true['image_info'] is used to rescale
        predictions.        
    """
    self._pq_metric_module = panoptic_quality.PanopticQuality(
        num_categories, ignored_label, max_instances_per_category, offset)
    self._is_thing = is_thing
    self._rescale_predictions = rescale_predictions
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
          - image_info: [batch, 4, 2], a tensor that holds information about
          original and preprocessed images. Each entry is in the format of
          [[original_height, original_width], [input_height, input_width],
          [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
          desired_width] is the actual scaled image size, and [y_scale, x_scale]
          is the scaling factor, which is the ratio of scaled dimension /
          original dimension.
      predictions: a dictionary of tensors including the fields below. See
        different parsers under `../dataloader` for more details.
        Required fields:
          - category_mask: a numpy array of uint16 of shape [batch_size, H, W].
          - instance_mask: a numpy array of uint16 of shape [batch_size, H, W].

    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError(
            'Missing the required key `{}` in predictions!'.format(k))

    for k in self._required_groundtruth_fields:
      if k not in groundtruths:
        raise ValueError(
            'Missing the required key `{}` in groundtruths!'.format(k))

    if self._rescale_predictions:
      for idx in range(len(groundtruths['category_mask'])):
        image_info = groundtruths['image_info'][idx]
        groundtruth_category_mask, prediction_category_mask = rescale_masks(
            groundtruths['category_mask'][idx],
            predictions['category_mask'][idx],
            image_info)
        groundtruth_instance_mask, prediction_instance_mask = rescale_masks(
            groundtruths['instance_mask'][idx],
            predictions['instance_mask'][idx],
            image_info)
        _groundtruths = {
            'category_mask': groundtruth_category_mask,
            'instance_mask': groundtruth_instance_mask
            }
        _predictions = {
            'category_mask': prediction_category_mask,
            'instance_mask': prediction_instance_mask
            }
        _groundtruths, _predictions = self._convert_to_numpy(
            _groundtruths, _predictions)

        self._pq_metric_module.compare_and_accumulate(
            _groundtruths, _predictions)
    else:
      groundtruths, predictions = self._convert_to_numpy(
          groundtruths, predictions)
      self._pq_metric_module.compare_and_accumulate(groundtruths, predictions)
