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

"""Data parser and processing for Mask R-CNN."""

import tensorflow as tf

from official.legacy.detection.dataloader import anchor
from official.legacy.detection.dataloader.maskrcnn_parser import Parser as MaskrcnnParser
from official.legacy.detection.utils import box_utils
from official.legacy.detection.utils import class_utils
from official.legacy.detection.utils import input_utils


class Parser(MaskrcnnParser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               rpn_match_threshold=0.7,
               rpn_unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               max_num_instances=100,
               include_mask=False,
               mask_crop_size=112,
               use_bfloat16=True,
               mode=None,
               # for centerness learning.
               has_centerness=False,
               rpn_center_match_iou_threshold=0.3,
               rpn_center_unmatched_iou_threshold=0.1,
               rpn_num_center_samples_per_im=256,
               # for class manipulation.
               class_agnostic=False,
               train_class='all',
               ):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold:
      rpn_unmatched_threshold:
      rpn_batch_size_per_im:
      rpn_fg_fraction:
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      include_mask: a bool to indicate whether parse mask groundtruth.
      mask_crop_size: the size which groundtruth mask is cropped to.
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction
        or prediction with groundtruths in the outputs.
      has_centerness: whether to create centerness targets
      rpn_center_match_iou_threshold: iou threshold for valid centerness samples
        ,set to 0.3 by default.
      rpn_center_unmatched_iou_threshold: iou threshold for invalid centerness
        samples, set to 0.1 by default.
      rpn_num_center_samples_per_im: number of centerness samples per image,
        256 by default.
      class_agnostic: whether to merge class ids into one foreground(=1) class,
        False by default.
      train_class: 'all' or 'voc' or 'nonvoc', 'all' by default.
    """
    super(Parser, self).__init__(
        output_size=output_size,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size,
        rpn_match_threshold=rpn_match_threshold,
        rpn_unmatched_threshold=rpn_unmatched_threshold,
        rpn_batch_size_per_im=rpn_batch_size_per_im,
        rpn_fg_fraction=rpn_fg_fraction,
        aug_rand_hflip=aug_rand_hflip,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        skip_crowd_during_training=skip_crowd_during_training,
        max_num_instances=max_num_instances,
        include_mask=include_mask,
        mask_crop_size=mask_crop_size,
        use_bfloat16=use_bfloat16,
        mode=mode,)

    # Centerness target assigning.
    self._has_centerness = has_centerness
    self._rpn_center_match_iou_threshold = rpn_center_match_iou_threshold
    self._rpn_center_unmatched_iou_threshold = (
        rpn_center_unmatched_iou_threshold)
    self._rpn_num_center_samples_per_im = rpn_num_center_samples_per_im

    # Class manipulation.
    self._class_agnostic = class_agnostic
    self._train_class = train_class

  def _parse_train_data(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    if self._include_mask:
      masks = data['groundtruth_instance_masks']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training and self._is_training:
      num_groundtruths = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtruths, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtruths), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)
      if self._include_mask:
        masks = tf.gather(masks, indices)

    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes = input_utils.random_horizontal_flip(
            image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = box_utils.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset)

    # Filters out ground truth boxes that are all zeros.
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    if self._include_mask:
      masks = tf.gather(masks, indices)
      # Transfer boxes to the original image space and do normalization.
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

    # Class manipulation.
    # Filter out novel split classes from training.
    if self._train_class != 'all':
      valid_classes = tf.cast(
          class_utils.coco_split_class_ids(self._train_class),
          dtype=classes.dtype)
      match = tf.reduce_any(tf.equal(
          tf.expand_dims(valid_classes, 1),
          tf.expand_dims(classes, 0)), 0)
      # kill novel split classes and boxes.
      boxes = tf.gather(boxes, tf.where(match)[:, 0])
      classes = tf.gather(classes, tf.where(match)[:, 0])
      if self._include_mask:
        masks = tf.gather(masks, tf.where(match)[:, 0])

    # Assigns anchor targets.
    # Note that after the target assignment, box targets are absolute pixel
    # offsets w.r.t. the scaled image.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))
    anchor_labeler = anchor.OlnAnchorLabeler(
        input_anchor,
        self._rpn_match_threshold,
        self._rpn_unmatched_threshold,
        self._rpn_batch_size_per_im,
        self._rpn_fg_fraction,
        # for centerness target.
        self._has_centerness,
        self._rpn_center_match_iou_threshold,
        self._rpn_center_unmatched_iou_threshold,
        self._rpn_num_center_samples_per_im,)

    if self._has_centerness:
      rpn_score_targets, _, rpn_lrtb_targets, rpn_center_targets = (
          anchor_labeler.label_anchors_lrtb(
              gt_boxes=boxes,
              gt_labels=tf.cast(
                  tf.expand_dims(classes, axis=-1), dtype=tf.float32)))
    else:
      rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
          boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))
      # For base rpn, dummy placeholder for centerness target.
      rpn_center_targets = rpn_score_targets.copy()

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    inputs = {
        'image': image,
        'image_info': image_info,
    }
    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'rpn_score_targets': rpn_score_targets,
        'rpn_box_targets': (rpn_lrtb_targets if self._has_centerness
                            else rpn_box_targets),
        'rpn_center_targets': rpn_center_targets,
    }
    # If class_agnostic, convert to binary classes.
    if self._class_agnostic:
      classes = tf.where(tf.greater(classes, 0),
                         tf.ones_like(classes),
                         tf.zeros_like(classes))

    inputs['gt_boxes'] = input_utils.pad_to_fixed_size(boxes,
                                                       self._max_num_instances,
                                                       -1)
    inputs['gt_classes'] = input_utils.pad_to_fixed_size(
        classes, self._max_num_instances, -1)
    if self._include_mask:
      inputs['gt_masks'] = input_utils.pad_to_fixed_size(
          masks, self._max_num_instances, -1)

    return inputs, labels
