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

"""Contains definition for postprocessing layer to genrate panoptic segmentations."""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from official.projects.panoptic.modeling.layers import paste_masks
from official.vision.ops import spatial_transform_ops


def _batch_count_ones(masks: tf.Tensor,
                      dtype: tf.dtypes.DType = tf.int32) -> tf.Tensor:
  """Counts the ones/trues for each mask in the batch.

  Args:
    masks: A tensor in shape (..., height, width) with arbitrary numbers of
      batch dimensions.
    dtype: DType of the resulting tensor. Default is tf.int32.

  Returns:
    A tensor which contains the count of non-zero elements for each mask in the
    batch. The rank of the resulting tensor is equal to rank(masks) - 2.
  """
  masks_shape = masks.get_shape().as_list()
  if len(masks_shape) < 2:
    raise ValueError(
        'Expected the input masks (..., height, width) has rank >= 2, was: %s' %
        masks_shape)
  return tf.reduce_sum(tf.cast(masks, dtype), axis=[-2, -1])


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
    super().__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape):
    grid_sampler = paste_masks.BilinearGridSampler(align_corners=False)
    self._paste_masks_fn = paste_masks.PasteMasks(
        output_size=self._output_size, grid_sampler=grid_sampler)
    super().build(input_shape)

  def _generate_panoptic_masks(
      self, boxes: tf.Tensor, scores: tf.Tensor, classes: tf.Tensor,
      detections_masks: tf.Tensor,
      segmentation_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
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

  def call(self,
           inputs: tf.Tensor,
           image_info: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
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

  def get_config(self) -> Dict[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Dict[str,
                                    Any]) -> 'PanopticSegmentationGenerator':
    return cls(**config)


class PanopticSegmentationGeneratorV2(tf.keras.layers.Layer):
  """Panoptic segmentation generator layer V2."""

  def __init__(self,
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
      stuff_classes_offset: An `int` that is added to the output of the semantic
        segmentation mask to make sure that the stuff class ids do not ovelap
        with the thing class ids of the MaskRCNN outputs.
      mask_binarize_threshold: A `float`
      score_threshold: A `float` representing the threshold for deciding when to
        remove objects based on score.
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
    super().__init__(**kwargs)

  def call(self,
           inputs: tf.Tensor,
           image_info: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    """Generates panoptic segmentation masks."""
    # (batch_size, num_rois, 4) in absolute coordinates.
    detection_boxes = tf.cast(inputs['detection_boxes'], tf.float32)
    # (batch_size, num_rois)
    detection_classes = tf.cast(inputs['detection_classes'], tf.int32)
    # (batch_size, num_rois)
    detection_scores = inputs['detection_scores']
    # (batch_size, num_rois, mask_height, mask_width)
    detections_masks = inputs['detection_masks']
    # (batch_size, height, width, num_semantic_classes)
    segmentation_outputs = inputs['segmentation_outputs']

    if self._rescale_predictions:
      # (batch_size, 2)
      original_size = tf.cast(image_info[:, 0, :], tf.float32)
      desired_size = tf.cast(image_info[:, 1, :], tf.float32)
      image_scale = tf.cast(image_info[:, 2, :], tf.float32)
      offset = tf.cast(image_info[:, 3, :], tf.float32)
      rescale_size = tf.math.ceil(desired_size / image_scale)
      # (batch_size, output_height, output_width, num_semantic_classes)
      segmentation_outputs = (
          spatial_transform_ops.bilinear_resize_with_crop_and_pad(
              segmentation_outputs,
              rescale_size,
              crop_offset=offset,
              crop_size=original_size,
              output_size=self._output_size))
      # (batch_size, 1, 4)
      image_scale = tf.tile(image_scale, multiples=[1, 2])[:, tf.newaxis]
      detection_boxes /= image_scale
    else:
      # (batch_size, output_height, output_width, num_semantic_classes)
      segmentation_outputs = tf.image.resize(
          segmentation_outputs, size=self._output_size, method='bilinear')

    # (batch_size, output_height, output_width)
    instance_mask, instance_category_mask = self._generate_instances(
        detection_boxes, detection_classes, detection_scores, detections_masks)

    # (batch_size, output_height, output_width)
    stuff_category_mask = self._generate_stuffs(segmentation_outputs)

    # (batch_size, output_height, output_width)
    category_mask = tf.where((stuff_category_mask != self._void_class_label) &
                             (instance_category_mask == self._void_class_label),
                             stuff_category_mask + self._stuff_classes_offset,
                             instance_category_mask)

    return {'instance_mask': instance_mask, 'category_mask': category_mask}

  def _generate_instances(
      self, detection_boxes: tf.Tensor, detection_classes: tf.Tensor,
      detection_scores: tf.Tensor,
      detections_masks: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generates instance & category masks from instance segmentation outputs."""
    batch_size = tf.shape(detections_masks)[0]
    num_rois = tf.shape(detections_masks)[1]
    mask_height = tf.shape(detections_masks)[2]
    mask_width = tf.shape(detections_masks)[3]
    output_height = self._output_size[0]
    output_width = self._output_size[1]

    # (batch_size, num_rois, mask_height, mask_width)
    detections_masks = detections_masks * (
        tf.cast((detection_scores > self._score_threshold) &
                (detection_classes != self._void_class_label),
                detections_masks.dtype)[:, :, tf.newaxis, tf.newaxis])

    # Resizes and copies the detections_masks to the bounding boxes in the
    # output canvas.
    # (batch_size, num_rois, output_height, output_width)
    pasted_detection_masks = tf.reshape(
        spatial_transform_ops.bilinear_resize_to_bbox(
            tf.reshape(detections_masks, [-1, mask_height, mask_width]),
            tf.reshape(detection_boxes, [-1, 4]), self._output_size),
        shape=[-1, num_rois, output_height, output_width])

    # (batch_size, num_rois, output_height, output_width)
    instance_binary_masks = (
        pasted_detection_masks > self._mask_binarize_threshold)

    # Sorts detection related tensors by scores.
    # (batch_size, num_rois)
    sorted_detection_indices = tf.argsort(
        detection_scores, axis=1, direction='DESCENDING')
    # (batch_size, num_rois)
    sorted_detection_classes = tf.gather(
        detection_classes, sorted_detection_indices, batch_dims=1)
    # (batch_size, num_rois, output_height, output_width)
    sorted_instance_binary_masks = tf.gather(
        instance_binary_masks, sorted_detection_indices, batch_dims=1)
    # (batch_size, num_rois)
    instance_areas = _batch_count_ones(
        sorted_instance_binary_masks, dtype=tf.float32)

    init_loop_vars = (
        0,  # i: the loop counter
        tf.ones([batch_size, output_height, output_width], dtype=tf.int32) *
        self._void_instance_id,  # combined_instance_mask
        tf.ones([batch_size, output_height, output_width], dtype=tf.int32) *
        self._void_class_label  # combined_category_mask
    )

    def _copy_instances_loop_body(
        i: int, combined_instance_mask: tf.Tensor,
        combined_category_mask: tf.Tensor) -> Tuple[int, tf.Tensor, tf.Tensor]:
      """Iterates the sorted detections and copies the instances."""
      # (batch_size, output_height, output_width)
      instance_binary_mask = sorted_instance_binary_masks[:, i]

      # Masks out the instances that have a big enough overlap with the other
      # instances with higher scores.
      # (batch_size, )
      overlap_areas = _batch_count_ones(
          (combined_instance_mask != self._void_instance_id)
          & instance_binary_mask,
          dtype=tf.float32)
      # (batch_size, )
      instance_overlap_threshold_mask = tf.math.divide_no_nan(
          overlap_areas, instance_areas[:, i]) < self._things_overlap_threshold
      # (batch_size, output_height, output_width)
      instance_binary_mask &= (
          instance_overlap_threshold_mask[:, tf.newaxis, tf.newaxis]
          & (combined_instance_mask == self._void_instance_id))

      # Updates combined_instance_mask.
      # (batch_size, )
      instance_id = tf.cast(
          sorted_detection_indices[:, i] + 1,  # starting from 1
          dtype=combined_instance_mask.dtype)
      # (batch_size, output_height, output_width)
      combined_instance_mask = tf.where(instance_binary_mask,
                                        instance_id[:, tf.newaxis, tf.newaxis],
                                        combined_instance_mask)

      # Updates combined_category_mask.
      # (batch_size, )
      class_id = tf.cast(
          sorted_detection_classes[:, i], dtype=combined_category_mask.dtype)
      # (batch_size, output_height, output_width)
      combined_category_mask = tf.where(instance_binary_mask,
                                        class_id[:, tf.newaxis, tf.newaxis],
                                        combined_category_mask)

      # Returns the updated loop vars.
      return (
          i + 1,  # Increment the loop counter i
          combined_instance_mask,
          combined_category_mask)

    # (batch_size, output_height, output_width)
    _, instance_mask, category_mask = tf.while_loop(
        cond=lambda i, *_: i < num_rois,
        body=_copy_instances_loop_body,
        loop_vars=init_loop_vars,
        parallel_iterations=32,
        maximum_iterations=num_rois)
    return instance_mask, category_mask

  def _generate_stuffs(self, segmentation_outputs: tf.Tensor) -> tf.Tensor:
    """Generates category mask from semantic segmentation outputs."""
    num_semantic_classes = tf.shape(segmentation_outputs)[3]

    # (batch_size, output_height, output_width)
    segmentation_masks = tf.argmax(
        segmentation_outputs, axis=-1, output_type=tf.int32)
    stuff_binary_masks = (segmentation_masks != self._things_class_label) & (
        segmentation_masks != self._void_class_label)
    # (batch_size, num_semantic_classes, output_height, output_width)
    stuff_class_binary_masks = ((tf.one_hot(
        segmentation_masks, num_semantic_classes, axis=1, dtype=tf.int32) == 1)
                                & tf.expand_dims(stuff_binary_masks, axis=1))

    # Masks out the stuff class whose area is below the given threshold.
    # (batch_size, num_semantic_classes)
    stuff_class_areas = _batch_count_ones(
        stuff_class_binary_masks, dtype=tf.float32)
    # (batch_size, num_semantic_classes, output_height, output_width)
    stuff_class_binary_masks &= tf.greater(
        stuff_class_areas, self._stuff_area_threshold)[:, :, tf.newaxis,
                                                       tf.newaxis]
    # (batch_size, output_height, output_width)
    stuff_binary_masks = tf.reduce_any(stuff_class_binary_masks, axis=1)

    # (batch_size, output_height, output_width)
    return tf.where(stuff_binary_masks, segmentation_masks,
                    tf.ones_like(segmentation_masks) * self._void_class_label)

  def get_config(self) -> Dict[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Dict[str,
                                    Any]) -> 'PanopticSegmentationGeneratorV2':
    return cls(**config)
