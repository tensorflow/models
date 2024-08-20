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

"""Data parser and processing for Panoptic Mask R-CNN."""
from typing import Optional

import tensorflow as tf, tf_keras
from tensorflow_addons import image as tfa_image

from official.projects.centernet.ops import target_assigner
from official.vision.configs import common
from official.vision.dataloaders import maskrcnn_input
from official.vision.dataloaders import tf_example_decoder
from official.vision.ops import augment
from official.vision.ops import preprocess_ops


class TfExampleDecoder(tf_example_decoder.TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(
      self,
      regenerate_source_id: bool,
      mask_binarize_threshold: float,
      include_panoptic_masks: bool,
      panoptic_category_mask_key: str = 'image/panoptic/category_mask',
      panoptic_instance_mask_key: str = 'image/panoptic/instance_mask'):
    super(TfExampleDecoder, self).__init__(
        include_mask=True,
        regenerate_source_id=regenerate_source_id,
        mask_binarize_threshold=None)

    self._include_panoptic_masks = include_panoptic_masks
    self._panoptic_category_mask_key = panoptic_category_mask_key
    self._panoptic_instance_mask_key = panoptic_instance_mask_key
    keys_to_features = {
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value='')}

    if include_panoptic_masks:
      keys_to_features.update({
          panoptic_category_mask_key:
              tf.io.FixedLenFeature((), tf.string, default_value=''),
          panoptic_instance_mask_key:
              tf.io.FixedLenFeature((), tf.string, default_value='')})
    self._segmentation_keys_to_features = keys_to_features

  def decode(self, serialized_example):
    decoded_tensors = super(TfExampleDecoder, self).decode(serialized_example)
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, self._segmentation_keys_to_features)
    segmentation_mask = tf.io.decode_image(
        parsed_tensors['image/segmentation/class/encoded'],
        channels=1)
    segmentation_mask.set_shape([None, None, 1])
    decoded_tensors.update({'groundtruth_segmentation_mask': segmentation_mask})

    if self._include_panoptic_masks:
      category_mask = tf.io.decode_image(
          parsed_tensors[self._panoptic_category_mask_key],
          channels=1)
      instance_mask = tf.io.decode_image(
          parsed_tensors[self._panoptic_instance_mask_key],
          channels=1)
      category_mask.set_shape([None, None, 1])
      instance_mask.set_shape([None, None, 1])

      decoded_tensors.update({
          'groundtruth_panoptic_category_mask':
              category_mask,
          'groundtruth_panoptic_instance_mask':
              instance_mask})
    return decoded_tensors


class Parser(maskrcnn_input.Parser):
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
               mask_crop_size=112,
               segmentation_resize_eval_groundtruth=True,
               segmentation_groundtruth_padded_size=None,
               segmentation_ignore_label=255,
               panoptic_ignore_label=0,
               include_panoptic_masks=True,
               level=3,
               num_panoptic_categories=201,
               num_thing_categories=91,
               gaussian_iou=0.7,
               aug_type: Optional[common.Augmentation] = None,
               max_num_stuff_centers=1,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instance, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instance, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold: `float`, match threshold for anchors in RPN.
      rpn_unmatched_threshold: `float`, unmatched threshold for anchors in RPN.
      rpn_batch_size_per_im: `int` for batch size per image in RPN.
      rpn_fg_fraction: `float` for forground fraction per batch in RPN.
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
      mask_crop_size: the size which groundtruth mask is cropped to.
      segmentation_resize_eval_groundtruth: `bool`, if True, eval groundtruth
        masks are resized to output_size.
      segmentation_groundtruth_padded_size: `Tensor` or `list` for [height,
        width]. When resize_eval_groundtruth is set to False, the groundtruth
        masks are padded to this size.
      segmentation_ignore_label: `int` the pixels with ignore label will not be
        used for training and evaluation.
      panoptic_ignore_label: `int` the pixels with ignore label will not be used
        by the PQ evaluator.
      include_panoptic_masks: `bool`, if True, category_mask and instance_mask
        will be parsed. Set this to true if PQ evaluator is enabled.
      level: `int`, output level, used to generate target assignments.
      num_panoptic_categories: `int`, number of panoptic categories.
      num_thing_categories: `int1, number of thing categories.
      gaussian_iou: `float`, used for generating the center heatmaps.
      aug_type: An optional Augmentation object with params for AutoAugment.
      max_num_stuff_centers: `int`, max number of stuff centers.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
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
        aug_rand_hflip=False,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        skip_crowd_during_training=skip_crowd_during_training,
        max_num_instances=max_num_instances,
        include_mask=True,
        mask_crop_size=mask_crop_size,
        dtype=dtype)

    self.aug_rand_hflip = aug_rand_hflip
    self._segmentation_resize_eval_groundtruth = segmentation_resize_eval_groundtruth
    if (not segmentation_resize_eval_groundtruth) and (
        segmentation_groundtruth_padded_size is None):
      raise ValueError(
          'segmentation_groundtruth_padded_size ([height, width]) needs to be'
          'specified when segmentation_resize_eval_groundtruth is False.')
    self._segmentation_groundtruth_padded_size = segmentation_groundtruth_padded_size
    self._segmentation_ignore_label = segmentation_ignore_label
    self._panoptic_ignore_label = panoptic_ignore_label
    self._include_panoptic_masks = include_panoptic_masks
    self.level = level
    self._num_panoptic_categories = num_panoptic_categories
    self._num_thing_categories = num_thing_categories
    self._gaussian_iou = gaussian_iou
    self._max_num_stuff_centers = max_num_stuff_centers

    if aug_type and aug_type.type:
      if aug_type.type == 'autoaug':
        self._augmenter = augment.AutoAugment(
            augmentation_name=aug_type.autoaug.augmentation_name,
            cutout_const=aug_type.autoaug.cutout_const,
            translate_const=aug_type.autoaug.translate_const)
      else:
        raise ValueError('Augmentation policy {} not supported.'.format(
            aug_type.type))
    else:
      self._augmenter = None

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
          [[original_height, original_width], [scaled_height, scaled_width]],
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
        gt_masks: Groundtruth masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
        gt_segmentation_mask: Groundtruth mask for segmentation head, this is
          resized to a fixed size determined by output_size.
        gt_segmentation_valid_mask: Binary mask that marks the pixels that
          are supposed to be used in computing the segmentation loss while
          training.
    """
    segmentation_mask = data['groundtruth_segmentation_mask']

    # Flips image randomly during training.
    if self.aug_rand_hflip:
      masks = data['groundtruth_instance_masks']
      instance_mask = data['groundtruth_panoptic_instance_mask']
      category_mask = data['groundtruth_panoptic_category_mask']
      image_mask = tf.concat(
          [data['image'], segmentation_mask, category_mask, instance_mask],
          axis=2)

      image_mask, boxes, masks = preprocess_ops.random_horizontal_flip(
          image_mask, data['groundtruth_boxes'], masks)

      segmentation_mask = image_mask[:, :, -3:-2]
      category_mask = image_mask[:, :, -2:-1]
      instance_mask = image_mask[:, :, -1:]
      image = image_mask[:, :, :-3]

      data['image'] = image
      if self._augmenter is not None:
        data['image'] = self._augmenter.distort(image)

      data['groundtruth_boxes'] = boxes
      data['groundtruth_instance_masks'] = masks
      data['groundtruth_panoptic_instance_mask'] = instance_mask
      data['groundtruth_panoptic_category_mask'] = category_mask

    image, labels = super(Parser, self)._parse_train_data(data)

    image_info = labels['image_info']

    def _process_mask(mask, ignore_label, image_info):
      mask = tf.cast(mask, dtype=tf.float32)
      mask = tf.reshape(mask, shape=[1, data['height'], data['width'], 1])
      mask += 1

      image_scale = image_info[2, :]
      offset = image_info[3, :]
      mask = preprocess_ops.resize_and_crop_masks(
          mask, image_scale, self._output_size, offset)

      mask -= 1
      # Assign ignore label to the padded region.
      mask = tf.where(
          tf.equal(mask, -1),
          ignore_label * tf.ones_like(mask),
          mask)
      mask = tf.squeeze(mask, axis=0)
      return mask

    if self._include_panoptic_masks:
      panoptic_category_mask = _process_mask(
          data['groundtruth_panoptic_category_mask'],
          -1, image_info)
      panoptic_instance_mask = _process_mask(
          data['groundtruth_panoptic_instance_mask'],
          self._panoptic_ignore_label, image_info)

      panoptic_category_mask = panoptic_category_mask[:, :, 0]
      panoptic_instance_mask = panoptic_instance_mask[:, :, 0]

      padding_mask = tf.cast(panoptic_category_mask == -1, tf.float32)
      labels.update({
          'gt_panoptic_category_mask':
              tf.cast(panoptic_category_mask, dtype=tf.int32),
          'gt_panoptic_instance_mask':
              tf.cast(panoptic_instance_mask, dtype=tf.int32)})

      def get_bbox(mask):
        rows = tf.math.count_nonzero(mask, axis=0, keepdims=None, dtype=tf.bool)
        columns = tf.math.count_nonzero(
            mask, axis=1, keepdims=None, dtype=tf.bool)
        indices = tf.where(tf.equal(columns, True))
        y_min = indices[0]
        y_max = indices[-1]
        indices = tf.where(tf.equal(rows, True))
        x_min = indices[0]
        x_max = indices[-1]
        return tf.stack([y_min, x_min, y_max, x_max])

      def get_thing_class_label(instance_id):
        mask = tf.cast(labels['gt_panoptic_instance_mask'] == instance_id,
                       tf.int32)
        bbox = tf.cast(tf.reshape(get_bbox(mask), [-1]), tf.float32)
        inds = tf.where(labels['gt_panoptic_instance_mask'] == instance_id)
        thing_classes = tf.gather_nd(labels['gt_panoptic_category_mask'], inds)
        thing_classes, _, counts = tf.unique_with_counts(thing_classes)
        max_index = tf.argmax(counts)
        thing_class = tf.cast(thing_classes[max_index], tf.int64)
        return tf.cast(mask, tf.float32), bbox, thing_class

      instance_ids, _ = tf.unique(
          tf.reshape(labels['gt_panoptic_instance_mask'], [-1]))
      instance_ids = tf.boolean_mask(instance_ids, instance_ids > 0)
      things_masks, things_bboxes, things_classes = tf.map_fn(
          get_thing_class_label,
          instance_ids,
          fn_output_signature=(tf.TensorSpec(self._output_size), tf.TensorSpec([
              4,
          ]), tf.int64))

      stuff_classes, _ = tf.unique(
          tf.reshape(labels['gt_panoptic_category_mask'], [-1]))
      stuff_classes = tf.boolean_mask(
          stuff_classes, stuff_classes > self._num_thing_categories - 1)

      # Compute classes and bboxes for stuff classes
      def get_stuff_class_label(stuff_class, k=self._max_num_stuff_centers):
        mask = tf.cast(
            labels['gt_panoptic_category_mask'] == stuff_class, tf.int32)
        smoothed_mask = tf.nn.max_pool(mask[:, :, None], 21, 1, padding='SAME')
        smoothed_mask = tf.nn.max_pool(smoothed_mask, 21, 1, padding='SAME')
        smoothed_mask = tf.nn.max_pool(smoothed_mask, 21, 1, padding='SAME')

        smoothed_mask = tf.cast(tf.squeeze(smoothed_mask, axis=2), tf.int32)
        connected_components = tfa_image.connected_components(
            images=smoothed_mask)
        counts = tf.math.bincount(connected_components)
        ids = tf.argsort(counts[1:], axis=-1, direction='DESCENDING') + 1

        masks = tf.cast(tf.repeat(mask[None, :, :], k, axis=0), tf.float32)
        stuff_classes = tf.cast(stuff_class, tf.int64) * tf.ones([k], tf.int64)

        # Pad or clip ids
        ids_length = tf.shape(ids)[0]
        ids_length = tf.clip_by_value(ids_length, 0, k)
        # batch_weights = 1 / tf.cast(ids_length, tf.float32) * tf.ones([1, k])
        batch_weights = 1.0 * tf.ones([1, k], tf.float32)
        ids = ids[:ids_length]
        padding_length = tf.maximum(0, k - ids_length)
        paddings = tf.cast(-1 * tf.ones([padding_length]), ids.dtype)
        ids = tf.concat([ids, paddings], axis=0)

        def get_bbox_and_batch_mask(island_id):
          if island_id == -1:
            bbox = tf.zeros([4], tf.float32)
            batch_mask = -1 * tf.ones([1], tf.int32)
            smask = tf.cast(tf.zeros_like(connected_components), tf.float32)
          else:
            smask = tf.cast(connected_components, tf.int32) == island_id
            smask = tf.cast(
                tf.logical_and(tf.cast(smask, tf.bool), tf.cast(mask, tf.bool)),
                tf.float32)
            if tf.reduce_sum(smask) < 1024:
              batch_mask = -1 * tf.ones([1], tf.int32)
            else:
              batch_mask = tf.ones([1], tf.int32)
            bbox = tf.cast(tf.reshape(get_bbox(smask), [-1]), tf.float32)
          return smask, bbox, batch_mask

        stuff_center_masks, bboxes, batch_mask = tf.map_fn(
            get_bbox_and_batch_mask,
            ids,
            fn_output_signature=((tf.TensorSpec(self._output_size, tf.float32),
                                  tf.TensorSpec([4,], tf.float32),
                                  tf.TensorSpec([1], tf.int32))))

        stuff_classes = tf.reshape(
            stuff_classes, [1, k])
        batch_mask = tf.reshape(batch_mask, [1, k])
        stuff_center_masks = tf.reshape(
            stuff_center_masks, [1, k] + self._output_size)
        return masks[None, :, :, :], bboxes[
            None, :, :], stuff_classes, batch_mask, batch_weights

      centers = self._max_num_stuff_centers
      stuff_masks, stuff_bboxes, stuff_classes, stuff_batch_mask, stuff_batch_weigths = tf.map_fn(
          get_stuff_class_label,
          stuff_classes,
          fn_output_signature=(tf.TensorSpec([1, centers] + self._output_size),
                               tf.TensorSpec([1, centers, 4,]),
                               tf.TensorSpec([1, centers], tf.int64),
                               tf.TensorSpec([1, centers], tf.int32),
                               tf.TensorSpec([1, centers], tf.float32)))
      stuff_masks = tf.reshape(stuff_masks, [-1] + self._output_size)
      stuff_bboxes = tf.reshape(stuff_bboxes, [-1, 4])
      stuff_classes = tf.reshape(stuff_classes, [-1])
      stuff_batch_mask = tf.reshape(stuff_batch_mask, [-1])
      stuff_batch_weigths = tf.reshape(stuff_batch_weigths, [-1])
      stuff_masks = stuff_masks[stuff_batch_mask == 1]
      stuff_bboxes = stuff_bboxes[stuff_batch_mask == 1]
      stuff_classes = stuff_classes[stuff_batch_mask == 1]

      # Start to generate panoptic heatmap.
      bboxes = tf.concat([things_bboxes, stuff_bboxes], axis=0)
      classes = tf.concat([things_classes, stuff_classes], axis=0)
      panoptic_masks = tf.concat([things_masks, stuff_masks], axis=0)

      width_ratio = 1 / float(2 ** self.level)
      height_ratio = 1 / float(2 ** self.level)

      # Original box coordinates
      # [max_num_instances, ]
      ytl, ybr = bboxes[..., 0], bboxes[..., 2]
      xtl, xbr = bboxes[..., 1], bboxes[..., 3]
      yct = (ytl + ybr) / 2
      xct = (xtl + xbr) / 2

      # Scaled box coordinates (could be floating point)
      # [max_num_instances, ]
      scale_xct = xct * width_ratio
      scale_yct = yct * height_ratio

      # Floor the scaled box coordinates to be placed on heatmaps
      # [max_num_instances, ]
      scale_xct_floor = tf.math.floor(scale_xct)
      scale_yct_floor = tf.math.floor(scale_yct)

      # Get the scaled box dimensions for computing the gaussian radius
      # [max_num_instances, ]
      box_widths = bboxes[..., 3] - bboxes[..., 1]
      box_heights = bboxes[..., 2] - bboxes[..., 0]

      box_widths = box_widths * width_ratio
      box_heights = box_heights * height_ratio

      ct_heatmap = target_assigner.assign_center_targets(
          out_height=int(self._output_size[0] * height_ratio),
          out_width=int(self._output_size[1]  * width_ratio),
          y_center=scale_yct,
          x_center=scale_xct,
          boxes_height=box_heights,
          boxes_width=box_widths,
          channel_onehot=tf.one_hot(
              tf.cast(classes, tf.int32),
              self._num_panoptic_categories, off_value=0.),
          gaussian_iou=self._gaussian_iou)
      box_indices = tf.cast(
          tf.stack([scale_yct_floor, scale_xct_floor], axis=-1), dtype=tf.int32)

      panoptic_classes = preprocess_ops.clip_or_pad_to_fixed_size(
          classes, self._max_num_instances, -1)
      panoptic_masks = preprocess_ops.clip_or_pad_to_fixed_size(
          panoptic_masks, self._max_num_instances, -1.0)
      panoptic_masks = tf.transpose(panoptic_masks, [1, 2, 0])
      panoptic_boxes = preprocess_ops.clip_or_pad_to_fixed_size(
          bboxes, self._max_num_instances, -1)

      box_indices = preprocess_ops.clip_or_pad_to_fixed_size(
          box_indices, self._max_num_instances, 0)

      labels = {}
      labels.update({
          'panoptic_heatmaps': ct_heatmap,
          'panoptic_classes': panoptic_classes,
          'panoptic_masks': panoptic_masks,
          'panoptic_mask_weights': tf.cast(panoptic_classes >= 0, tf.float32),
          'panoptic_padding_mask': padding_mask,
          'panoptic_boxes': panoptic_boxes,
          'panoptic_box_indices': box_indices,
          'num_instances': tf.reduce_sum(
              tf.cast(panoptic_classes >= 0, tf.float32))
      })
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      A dictionary of {'images': image, 'labels': labels} where
        image: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following
          describes {key: value} pairs in the dictionary.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
          image_info: a 2D `Tensor` that encodes the information of the image
            and the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width]],
          anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each
            level.
    """
    def _process_mask(mask, ignore_label, image_info):
      mask = tf.cast(mask, dtype=tf.float32)
      mask = tf.reshape(mask, shape=[1, data['height'], data['width'], 1])
      mask += 1

      if self._segmentation_resize_eval_groundtruth:
        # Resizes eval masks to match input image sizes. In that case, mean IoU
        # is computed on output_size not the original size of the images.
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        mask = preprocess_ops.resize_and_crop_masks(
            mask, image_scale, self._output_size, offset)
      else:
        mask = tf.image.pad_to_bounding_box(
            mask, 0, 0,
            self._segmentation_groundtruth_padded_size[0],
            self._segmentation_groundtruth_padded_size[1])
      mask -= 1
      # Assign ignore label to the padded region.
      mask = tf.where(
          tf.equal(mask, -1),
          ignore_label * tf.ones_like(mask),
          mask)
      mask = tf.squeeze(mask, axis=0)
      return mask

    image, labels = super(Parser, self)._parse_eval_data(data)
    image_info = labels['image_info']

    segmentation_mask = _process_mask(
        data['groundtruth_segmentation_mask'],
        self._segmentation_ignore_label, image_info)
    segmentation_valid_mask = tf.not_equal(
        segmentation_mask, self._segmentation_ignore_label)
    labels['groundtruths'].update({
        'gt_segmentation_mask': segmentation_mask,
        'gt_segmentation_valid_mask': segmentation_valid_mask})

    if self._include_panoptic_masks:
      panoptic_category_mask = _process_mask(
          data['groundtruth_panoptic_category_mask'],
          self._panoptic_ignore_label, image_info)
      panoptic_instance_mask = _process_mask(
          data['groundtruth_panoptic_instance_mask'],
          0, image_info)

      panoptic_category_mask = panoptic_category_mask[:, :, 0]
      panoptic_instance_mask = panoptic_instance_mask[:, :, 0]

      labels['groundtruths'].update({
          'gt_panoptic_category_mask':
              tf.cast(panoptic_category_mask, dtype=tf.int32),
          'gt_panoptic_instance_mask':
              tf.cast(panoptic_instance_mask, dtype=tf.int32)})

    return image, labels
