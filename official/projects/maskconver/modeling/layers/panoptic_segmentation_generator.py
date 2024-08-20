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

"""Postprocessing layer to generate panoptic segmentations."""

from typing import Any, Dict, List, Mapping, Optional
import tensorflow as tf, tf_keras
from official.modeling import activations
from official.vision.ops import spatial_transform_ops


class MaskConverProposalGenerator(tf_keras.layers.Layer):
  """MaskConverProposalGenerator."""

  def __init__(self,
               max_proposals: int = 100,
               peak_error: float = 1e-6,
               peak_extract_kernel_size: int = 3,
               **kwargs):
    """Initialize MaskConverProposalGenerator.

    Args:
      max_proposals: An `int` specifying the maximum number of max_proposals.
      peak_error: A `float` for determining non-valid heatmap locations to mask.
      peak_extract_kernel_size: An `int` indicating the kernel size used when
        performing max-pool over the heatmaps to detect valid center locations
        from its neighbors. From the paper, set this to 3 to detect valid.
        locations that have responses greater than its 8-connected neighbors
      **kwargs: Additional keyword arguments to be passed.
    """
    super(MaskConverProposalGenerator, self).__init__(**kwargs)

    # Object center selection parameters
    self._max_proposals = max_proposals
    self._peak_error = peak_error
    self._peak_extract_kernel_size = peak_extract_kernel_size

  def process_heatmap(self, feature_map: tf.Tensor,
                      kernel_size: int) -> tf.Tensor:
    """Processes the heatmap into peaks for box selection.

    Given a heatmap, this function first masks out nearby heatmap locations of
    the same class using max-pooling such that, ideally, only one center for the
    object remains. Then, center locations are masked according to their scores
    in comparison to a threshold. NOTE: Repurposed from Google OD API.

    Args:
      feature_map: A Tensor with shape [batch_size, height, width, num_classes]
        which is the center heatmap predictions.
      kernel_size: An integer value for max-pool kernel size.

    Returns:
      A Tensor with the same shape as the input but with non-valid center
        prediction locations masked out.
    """

    feature_map = tf.math.sigmoid(feature_map)
    if not kernel_size or kernel_size == 1:
      feature_map_peaks = feature_map
    else:
      feature_map_max_pool = tf.nn.max_pool(
          feature_map,
          ksize=kernel_size,
          strides=1,
          padding='SAME')

      feature_map_peak_mask = tf.math.abs(
          feature_map - feature_map_max_pool) < self._peak_error

      # Zero out everything that is not a peak.
      feature_map_peaks = (
          feature_map * tf.cast(feature_map_peak_mask, feature_map.dtype))

    return feature_map_peaks

  def get_top_k_peaks(self,
                      feature_map_peaks: tf.Tensor,
                      batch_size: int,
                      width: int,
                      num_classes: int,
                      k: int = 100):
    """Gets the scores and indices of the top-k peaks from the feature map.

    This function flattens the feature map in order to retrieve the top-k
    peaks, then computes the x, y, and class indices for those scores.
    NOTE: Repurposed from Google OD API.

    Args:
      feature_map_peaks: A `Tensor` with shape [batch_size, height,
        width, num_classes] which is the processed center heatmap peaks.
      batch_size: An `int` that indicates the batch size of the input.
      width: An `int` that indicates the width (and also height) of the input.
      num_classes: An `int` for the number of possible classes. This is also
        the channel depth of the input.
      k: `int` that controls how many peaks to select.

    Returns:
      top_scores: A Tensor with shape [batch_size, k] containing the top-k
        scores.
      y_indices: A Tensor with shape [batch_size, k] containing the top-k
        y-indices corresponding to top_scores.
      x_indices: A Tensor with shape [batch_size, k] containing the top-k
        x-indices corresponding to top_scores.
      channel_indices: A Tensor with shape [batch_size, k] containing the top-k
        channel indices corresponding to top_scores.
    """
    # Flatten the entire prediction per batch
    feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])

    # top_scores and top_indices have shape [batch_size, k]
    top_scores, top_indices = tf.math.top_k(feature_map_peaks_flat, k=k)

    # Get x, y and channel indices corresponding to the top indices in the flat
    # array.
    y_indices = (top_indices // num_classes) // width
    x_indices = (top_indices // num_classes) - y_indices * width
    channel_indices_temp = top_indices // num_classes
    channel_indices = top_indices - channel_indices_temp * num_classes

    embedding_indices = tf.stack([y_indices, x_indices], axis=2)

    return top_scores, embedding_indices, channel_indices

  def __call__(self, ct_heatmaps: tf.Tensor):
    # Get heatmaps from decoded outputs via final hourglass stack output
    shape = tf.shape(ct_heatmaps)

    _, width = shape[1], shape[2]
    batch_size, num_channels = shape[0], shape[3]

    # Process heatmaps using 3x3 max pool and applying sigmoid
    peaks = self.process_heatmap(
        feature_map=ct_heatmaps,
        kernel_size=self._peak_extract_kernel_size)

    # Get top scores along with their x, y, and class
    # Each has size [batch_size, k]
    scores, embedding_indices, channel_indices = self.get_top_k_peaks(
        feature_map_peaks=peaks,
        batch_size=batch_size,
        width=width,
        num_classes=num_channels,
        k=self._max_proposals)

    num_proposals = tf.reduce_sum(tf.cast(scores > 0, dtype=tf.int32), axis=1)
    return {
        'classes': channel_indices,
        'confidence': scores,
        'embedding_indices': embedding_indices,
        'num_proposals': num_proposals
    }

  def get_config(self) -> Mapping[str, Any]:
    config = {
        'max_proposals': self._max_proposals,
        'peak_error': self._peak_error,
        'peak_extract_kernel_size': self._peak_extract_kernel_size,
    }

    base_config = super(MaskConverProposalGenerator, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class MaskConverPanopticGenerator(tf_keras.layers.Layer):
  """MaskConver panoptic generator."""

  def __init__(self,
               output_size: List[int],
               num_classes: int,
               is_thing: List[bool],
               num_instances: int = 100,
               object_mask_threshold: float = 0.,
               small_area_threshold: int = 0,
               overlap_threshold: float = 0.,
               rescale_predictions: bool = False,
               use_hardware_optimization: bool = False,
               **kwargs):
    super(MaskConverPanopticGenerator, self).__init__(**kwargs)

    self._output_size = output_size
    assert num_classes == len(is_thing)
    self._num_classes = num_classes
    self._is_thing = tf.constant(is_thing, dtype=tf.bool)
    self._num_instances = num_instances
    self._object_mask_threshold = object_mask_threshold
    self._small_area_threshold = small_area_threshold
    self._overlap_threshold = overlap_threshold
    self._rescale_predictions = rescale_predictions
    self._use_hardware_optimization = use_hardware_optimization

    self.config_dict = {
        'output_size': output_size,
        'num_classes': num_classes,
        'is_thing': is_thing,
        'num_instances': num_instances,
        'object_mask_threshold': object_mask_threshold,
        'small_area_threshold': small_area_threshold,
        'overlap_threshold': overlap_threshold,
        'rescale_predictions': rescale_predictions,
        'use_hardware_optimization': use_hardware_optimization,
    }

  def _resize_and_pad_masks(self, masks, images_info):
    """Resizes masks to match the original image shape and pads to`output_size`.

    Args:
      masks: a padded binary mask tensor, shape [b, h, w, 1].
      images_info: a tensor that holds information about original and
        preprocessed images.

    Returns:
      resized and padded masks: tf.Tensor.
    """
    rescale_size = tf.cast(
        tf.math.ceil(images_info[:, 1, :] / images_info[:, 2, :]), tf.int32)
    image_shape = tf.cast(images_info[:, 0, :], tf.int32)
    offsets = tf.cast(images_info[:, 3, :], tf.int32)

    return spatial_transform_ops.bilinear_resize_with_crop_and_pad(
        masks,
        rescale_size,
        crop_offset=offsets,
        crop_size=image_shape,
        output_size=self._output_size)

  def _generate_panoptic_masks(self, scores: tf.Tensor, classes: tf.Tensor,
                               masks: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Compute category and instance masks.

    Args:
      scores: Class confidences, shape [num_proposals].
      classes: Class IDs, shape [num_proposals]
      masks: Predicted binary mask logits, shape [num_proposals, height, width].

    Returns:
      Category and instance masks, both have shape [num_proposals, height,
      width].
    """
    # Collect valid proposals.
    valid_proposals = scores > self._object_mask_threshold

    cur_scores = tf.cast(valid_proposals, tf.float32) * scores  # n
    cur_classes = tf.cast(valid_proposals, tf.int32) * classes  # n
    cur_masks = tf.cast(  # h x w x n
        valid_proposals, tf.float32)[None, None, :] * masks

    cur_scores, cur_indices = tf.math.top_k(cur_scores, k=self._num_instances)
    cur_classes = tf.gather(cur_classes, cur_indices)
    cur_masks = tf.gather(cur_masks, cur_indices, axis=2)

    num_proposals = self._num_instances

    # Find the proposal ID for each pixel.
    cur_mask_ids = tf.argmax(  # h x w
        cur_masks * cur_scores[None, None, :],
        axis=2,
        output_type=tf.int32)
    # Compute original areas from binary mask.
    original_areas = tf.reduce_sum(
        tf.cast(cur_masks > 0.5, tf.float32), axis=[0, 1])  # n
    # Compute mask areas from the proposal ID mask.
    proposal_masks = tf.range(  # h x w x n
        num_proposals, dtype=tf.int32)[None, None, :] == cur_mask_ids[..., None]
    mask_areas = tf.reduce_sum(  # n
        tf.cast(proposal_masks, tf.float32), axis=[0, 1])
    # Compute valid masks to filter results.
    valid_masks = tf.logical_and(mask_areas > self._small_area_threshold,  # n
                                 original_areas > 0)
    valid_masks = tf.logical_and(  # n
        valid_masks, mask_areas > self._overlap_threshold * original_areas)
    # Compute category mask and instance mask.
    category_mask = tf.gather(cur_classes * tf.cast(valid_masks, tf.int32),
                              cur_mask_ids)
    is_thing_mask = tf.gather(self._is_thing,
                              cur_classes * tf.cast(valid_masks, tf.int32))
    instance_mask = tf.gather(is_thing_mask, cur_mask_ids)
    instance_mask = tf.where(instance_mask, cur_mask_ids + 1,
                             tf.zeros_like(cur_mask_ids))
    return {'category_mask': category_mask, 'instance_mask': instance_mask}

  def __call__(self,
               inputs: Dict[str, tf.Tensor],
               images_info: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    batched_scores = tf.cast(inputs['confidence'], dtype=tf.float32)
    batched_classes = tf.cast(inputs['classes'], dtype=tf.int32)

    batched_masks = tf.cast(inputs['mask_proposal_logits'], dtype=tf.float32)
    # For on-device run, we use the following codes to speed up.
    if self._use_hardware_optimization:
      batched_masks = activations.hard_sigmoid(batched_masks)
      # Note that we assume batch size is always 1.
      panoptic_masks = self._generate_panoptic_masks(batched_scores[0],
                                                     batched_classes[0],
                                                     batched_masks[0])
      for k, v in panoptic_masks.items():
        panoptic_masks[k] = v[None]
        panoptic_masks[k].set_shape((1, *self._output_size))
      return panoptic_masks

    if self._rescale_predictions and images_info is not None:
      batched_masks = self._resize_and_pad_masks(batched_masks, images_info)
    else:
      batched_masks = tf.image.resize(batched_masks, self._output_size,
                                      'bilinear')
    batched_masks = tf.nn.sigmoid(batched_masks)
    panoptic_masks = tf.map_fn(
        fn=lambda x: self._generate_panoptic_masks(x[0], x[1], x[2]),
        elems=(batched_scores, batched_classes, batched_masks),
        fn_output_signature={
            'category_mask': tf.int32,
            'instance_mask': tf.int32
        },
        parallel_iterations=32)

    return panoptic_masks

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
