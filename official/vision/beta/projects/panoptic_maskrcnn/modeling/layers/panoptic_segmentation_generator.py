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

"""Contains definition for postprocessing layer to genrate panoptic segmentations."""

from typing import List, Optional

import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.modeling.layers import paste_masks


class PanopticSegmentationGenerator(tf.keras.layers.Layer):
  """Panoptic segmentation generator layer."""

  def __init__(
      self,
      output_size: List[int],
      max_num_detections: int,
      stuff_classes_offset: int,
      mask_binarize_threshold: float = 0.5,
      score_threshold: float = 0.5,
      things_overlap_threshold: float = 0.5,
      stuff_area_threshold: float = 4096,
      things_class_label: int = 1,
      void_class_label: int = 0,
      void_instance_id: int = -1,
      rescale_predictions: bool = False,
      **kwargs):
    """Generates panoptic segmentation masks.

    Args:
      output_size: A `List` of integers that represent the height and width of
        the output mask.
      max_num_detections: `int` for maximum number of detections.
      stuff_classes_offset: An `int` that is added to the output of the
        semantic segmentation mask to make sure that the stuff class ids do not
        ovelap with the thing class ids of the MaskRCNN outputs.
      mask_binarize_threshold: A `float`
      score_threshold: A `float` representing the threshold for deciding
      when to remove objects based on score.
      things_overlap_threshold: A `float` representing a threshold for deciding
        to ignore a thing if overlap is above the threshold.
      stuff_area_threshold: A `float` representing a threshold for deciding to
        to ignore a stuff class if area is below certain threshold.
      things_class_label: An `int` that represents a single merged category of
        all thing classes in the semantic segmentation output.
      void_class_label: An `int` that is used to represent empty or unlabelled
        regions of the mask
      void_instance_id: An `int` that is used to denote regions that are not
        assigned to any thing class. That is, void_instance_id are assigned to
        both stuff regions and empty regions.
      rescale_predictions: `bool`, whether to scale back prediction to original
        image sizes. If True, image_info is used to rescale predictions.
      **kwargs: additional kewargs arguments.
    """
    self._output_size = output_size
    self._max_num_detections = max_num_detections
    self._stuff_classes_offset = stuff_classes_offset
    self._mask_binarize_threshold = mask_binarize_threshold
    self._score_threshold = score_threshold
    self._things_overlap_threshold = things_overlap_threshold
    self._stuff_area_threshold = stuff_area_threshold
    self._things_class_label = things_class_label
    self._void_class_label = void_class_label
    self._void_instance_id = void_instance_id
    self._rescale_predictions = rescale_predictions

    self._config_dict = {
        'output_size': output_size,
        'max_num_detections': max_num_detections,
        'stuff_classes_offset': stuff_classes_offset,
        'mask_binarize_threshold': mask_binarize_threshold,
        'score_threshold': score_threshold,
        'things_class_label': things_class_label,
        'void_class_label': void_class_label,
        'void_instance_id': void_instance_id,
        'rescale_predictions': rescale_predictions
    }
    super(PanopticSegmentationGenerator, self).__init__(**kwargs)

  def build(self, input_shape):
    grid_sampler = paste_masks.BilinearGridSampler(align_corners=False)
    self._paste_masks_fn = paste_masks.PasteMasks(
        output_size=self._output_size, grid_sampler=grid_sampler)

  def _generate_panoptic_masks(self, boxes, scores, classes, detections_masks,
                               segmentation_mask):
    """Generates panoptic masks for a single image.

    This function implements the following steps to merge instance and semantic
      segmentation masks described in https://arxiv.org/pdf/1901.02446.pdf
    Steps:
      1. resolving overlaps between different instances based on their
          confidence scores
      2. resolving overlaps between instance and semantic segmentation
          outputs in favor of instances
      3. removing any stuff regions labeled other or under a given area
          threshold.
    Args:
      boxes: A `tf.Tensor` of shape [num_rois, 4], representing the bounding
        boxes for detected objects.
      scores: A `tf.Tensor` of shape [num_rois], representing the
        confidence scores for each object.
      classes: A `tf.Tensor` of shape [num_rois], representing the class
        for each object.
      detections_masks: A `tf.Tensor` of shape
        [num_rois, mask_height, mask_width, 1], representing the cropped mask
        for each object.
      segmentation_mask: A `tf.Tensor` of shape [height, width], representing
        the semantic segmentation output.
    Returns:
      Dict with the following keys:
        - category_mask: A `tf.Tensor` for category masks.
        - instance_mask: A `tf.Tensor for instance masks.
    """

    # Offset stuff class predictions
    segmentation_mask = tf.where(
        tf.logical_or(
            tf.equal(segmentation_mask, self._things_class_label),
            tf.equal(segmentation_mask, self._void_class_label)),
        segmentation_mask,
        segmentation_mask + self._stuff_classes_offset
    )
    # sort instances by their scores
    sorted_indices = tf.argsort(scores, direction='DESCENDING')

    mask_shape = self._output_size + [1]
    category_mask = tf.ones(mask_shape,
                            dtype=tf.float32) * self._void_class_label
    instance_mask = tf.ones(
        mask_shape, dtype=tf.float32) * self._void_instance_id

    # filter instances with low confidence
    sorted_scores = tf.sort(scores, direction='DESCENDING')

    valid_indices = tf.where(sorted_scores > self._score_threshold)

    # if no instance has sufficient confidence score, skip merging
    # instance segmentation masks
    if tf.shape(valid_indices)[0] > 0:
      loop_end_idx = valid_indices[-1, 0] + 1
      loop_end_idx = tf.minimum(
          tf.cast(loop_end_idx, dtype=tf.int32),
          self._max_num_detections)
      pasted_masks = self._paste_masks_fn((
          detections_masks[:loop_end_idx],
          boxes[:loop_end_idx]))

      # add things segmentation to panoptic masks
      for i in range(loop_end_idx):
        # we process instances in decending order, which will make sure
        # the overlaps are resolved based on confidence score
        instance_idx = sorted_indices[i]

        pasted_mask = pasted_masks[instance_idx]

        class_id = tf.cast(classes[instance_idx], dtype=tf.float32)

        # convert sigmoid scores to binary values
        binary_mask = tf.greater(
            pasted_mask, self._mask_binarize_threshold)

        # filter empty instance masks
        if not tf.reduce_sum(tf.cast(binary_mask, tf.float32)) > 0:
          continue

        overlap = tf.logical_and(
            binary_mask,
            tf.not_equal(category_mask, self._void_class_label))
        binary_mask_area = tf.reduce_sum(
            tf.cast(binary_mask, dtype=tf.float32))
        overlap_area = tf.reduce_sum(
            tf.cast(overlap, dtype=tf.float32))

        # skip instance that have a big enough overlap with instances with
        # higer scores
        if overlap_area / binary_mask_area > self._things_overlap_threshold:
          continue

        # fill empty regions in category_mask represented by
        # void_class_label with class_id of the instance.
        category_mask = tf.where(
            tf.logical_and(
                binary_mask, tf.equal(category_mask, self._void_class_label)),
            tf.ones_like(category_mask) * class_id, category_mask)

        # fill empty regions in the instance_mask represented by
        # void_instance_id with the id of the instance, starting from 1
        instance_mask = tf.where(
            tf.logical_and(
                binary_mask,
                tf.equal(instance_mask, self._void_instance_id)),
            tf.ones_like(instance_mask) *
            tf.cast(instance_idx + 1, tf.float32), instance_mask)

    stuff_class_ids = tf.unique(tf.reshape(segmentation_mask, [-1])).y
    for stuff_class_id in stuff_class_ids:
      if stuff_class_id == self._things_class_label:
        continue

      stuff_mask = tf.logical_and(
          tf.equal(segmentation_mask, stuff_class_id),
          tf.equal(category_mask, self._void_class_label))

      stuff_mask_area = tf.reduce_sum(
          tf.cast(stuff_mask, dtype=tf.float32))

      if stuff_mask_area < self._stuff_area_threshold:
        continue

      category_mask = tf.where(
          stuff_mask,
          tf.ones_like(category_mask) * stuff_class_id,
          category_mask)

    results = {
        'category_mask': category_mask[:, :, 0],
        'instance_mask': instance_mask[:, :, 0]
    }
    return results

  def _resize_and_pad_masks(self, mask, image_info):
    """Resizes masks to match the original image shape and pads to`output_size`.

    Args:
      mask: a padded mask tensor.
      image_info: a tensor that holds information about original and
        preprocessed images.
    Returns:
      resized and padded masks: tf.Tensor.
    """
    rescale_size = tf.cast(
        tf.math.ceil(image_info[1, :] / image_info[2, :]), tf.int32)
    image_shape = tf.cast(image_info[0, :], tf.int32)
    offsets = tf.cast(image_info[3, :], tf.int32)

    mask = tf.image.resize(
        mask,
        rescale_size,
        method='bilinear')
    mask = tf.image.crop_to_bounding_box(
        mask,
        offsets[0], offsets[1],
        image_shape[0],
        image_shape[1])
    mask = tf.image.pad_to_bounding_box(
        mask, 0, 0, self._output_size[0], self._output_size[1])
    return mask

  def call(self, inputs: tf.Tensor, image_info: Optional[tf.Tensor] = None):
    detections = inputs

    batched_scores = detections['detection_scores']
    batched_classes = detections['detection_classes']
    batched_detections_masks = tf.expand_dims(
        detections['detection_masks'], axis=-1)
    batched_boxes = detections['detection_boxes']
    batched_segmentation_masks = tf.cast(
        detections['segmentation_outputs'], dtype=tf.float32)

    if self._rescale_predictions:
      scale = tf.tile(
          tf.cast(image_info[:, 2:3, :], dtype=batched_boxes.dtype),
          multiples=[1, 1, 2])
      batched_boxes /= scale

      batched_segmentation_masks = tf.map_fn(
          fn=lambda x: self._resize_and_pad_masks(x[0], x[1]),
          elems=(
              batched_segmentation_masks,
              image_info),
          fn_output_signature=tf.float32,
          parallel_iterations=32)
    else:
      batched_segmentation_masks = tf.image.resize(
          batched_segmentation_masks,
          size=self._output_size,
          method='bilinear')

    batched_segmentation_masks = tf.expand_dims(tf.cast(
        tf.argmax(batched_segmentation_masks, axis=-1),
        dtype=tf.float32), axis=-1)

    panoptic_masks = tf.map_fn(
        fn=lambda x: self._generate_panoptic_masks(  # pylint:disable=g-long-lambda
            x[0], x[1], x[2], x[3], x[4]),
        elems=(
            batched_boxes,
            batched_scores,
            batched_classes,
            batched_detections_masks,
            batched_segmentation_masks),
        fn_output_signature={
            'category_mask': tf.float32,
            'instance_mask': tf.float32
        }, parallel_iterations=32)

    for k, v in panoptic_masks.items():
      panoptic_masks[k] = tf.cast(v, dtype=tf.int32)

    return panoptic_masks

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
