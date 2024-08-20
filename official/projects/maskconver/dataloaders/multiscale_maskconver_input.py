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

"""Data parser and processing for Panoptic MaskConver."""
from typing import Optional

import tensorflow as tf, tf_keras

from official.projects.centernet.ops import target_assigner
from official.vision.configs import common
from official.vision.dataloaders import parser
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import utils
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

    super().__init__(
        include_mask=True, regenerate_source_id=regenerate_source_id,
        mask_binarize_threshold=None)

    self._include_panoptic_masks = include_panoptic_masks
    self._panoptic_category_mask_key = panoptic_category_mask_key
    self._panoptic_instance_mask_key = panoptic_instance_mask_key

    if include_panoptic_masks:
      self._segmentation_keys_to_features = {
          panoptic_category_mask_key:
              tf.io.FixedLenFeature((), tf.string, default_value=''),
          panoptic_instance_mask_key:
              tf.io.FixedLenFeature((), tf.string, default_value='')}

  def decode(self, serialized_example):
    decoded_tensors = super().decode(serialized_example)

    if self._include_panoptic_masks:
      parsed_tensors = tf.io.parse_single_example(
          serialized_example, self._segmentation_keys_to_features)
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


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               fpn_low_range,
               fpn_high_range,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               max_num_instances=100,
               segmentation_resize_eval_groundtruth=True,
               segmentation_groundtruth_padded_size=None,
               segmentation_ignore_label=255,
               panoptic_ignore_label=0,
               level=3,
               mask_target_level=1,
               num_panoptic_categories=201,
               num_thing_categories=91,
               gaussian_iou=0.7,
               aug_type: Optional[common.Augmentation] = None,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      fpn_low_range: List of `int`.
      fpn_high_range: List of `int`.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      segmentation_resize_eval_groundtruth: `bool`, for whether or not to
        resize the eval groundtruth masks.
      segmentation_groundtruth_padded_size: `Tensor` or `list` for [height,
        width]. When resize_eval_groundtruth is set to False, the groundtruth
        masks are padded to this size.
      segmentation_ignore_label: `int` the pixels with ignore label will not be
        used for training and evaluation.
      panoptic_ignore_label: `int` the pixels with ignore label will not be used
        by the PQ evaluator.
      level: `int`, output level, used to generate target assignments.
      mask_target_level: `int`, target level for panoptic masks, default is 1.
      num_panoptic_categories: `int`, number of panoptic categories.
      num_thing_categories: `int1, number of thing categories.
      gaussian_iou: `float`, used for generating the center heatmaps.
      aug_type: An optional Augmentation object with params for AutoAugment.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    super().__init__()

    self.aug_rand_hflip = aug_rand_hflip
    self._segmentation_resize_eval_groundtruth = (
        segmentation_resize_eval_groundtruth
    )
    if (not segmentation_resize_eval_groundtruth) and (
        segmentation_groundtruth_padded_size is None
    ):
      raise ValueError(
          'segmentation_groundtruth_padded_size ([height, width]) needs to be'
          'specified when segmentation_resize_eval_groundtruth is False.'
      )
    self._segmentation_groundtruth_padded_size = (
        segmentation_groundtruth_padded_size
    )
    self._dtype = dtype
    self._max_num_instances = max_num_instances
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._segmentation_ignore_label = segmentation_ignore_label
    self._panoptic_ignore_label = panoptic_ignore_label
    self.level = level
    self._mask_target_level = mask_target_level
    self._num_panoptic_categories = num_panoptic_categories
    self._num_thing_categories = num_thing_categories
    self._gaussian_iou = gaussian_iou
    self.fpn_low_range = fpn_low_range
    self.fpn_high_range = fpn_high_range

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
    """
    # Flips image randomly during training.
    if self.aug_rand_hflip:
      instance_mask = data['groundtruth_panoptic_instance_mask']
      category_mask = data['groundtruth_panoptic_category_mask']
      image_mask = tf.concat([
          tf.cast(data['image'], category_mask.dtype), category_mask,
          instance_mask
      ], axis=2)

      image_mask, _, _ = preprocess_ops.random_horizontal_flip(image_mask)

      instance_mask = image_mask[:, :, -1:]
      category_mask = image_mask[:, :, -2:-1]
      image = tf.cast(image_mask[:, :, :-2], tf.uint8)

      if self._augmenter is not None:
        image = self._augmenter.distort(image)

    image = preprocess_ops.normalize_image(image)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(self._output_size,
                                                       2**self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    def _process_mask(mask, ignore_label, image_info):
      mask = tf.cast(mask, dtype=tf.float32)
      mask = tf.reshape(mask, shape=[1, data['height'], data['width'], 1])
      mask += 1

      image_scale = image_info[2, :]
      offset = image_info[3, :]
      mask = preprocess_ops.resize_and_crop_masks(
          mask, image_scale, self._output_size, offset)

      mask -= 1
      # Assigns ignore label to the padded region.
      mask = tf.where(
          tf.equal(mask, -1),
          ignore_label * tf.ones_like(mask),
          mask)
      mask = tf.squeeze(mask, axis=0)
      return mask

    panoptic_category_mask = _process_mask(
        category_mask,
        -1, image_info)
    panoptic_instance_mask = _process_mask(
        instance_mask,
        self._panoptic_ignore_label, image_info)

    mask_level = self._mask_target_level
    mask_size = [
        int(self._output_size[0] / 2**mask_level),
        int(self._output_size[1] / 2**mask_level)]
    panoptic_category_mask = tf.image.resize(
        panoptic_category_mask, mask_size, method='nearest')
    panoptic_instance_mask = tf.image.resize(
        panoptic_instance_mask, mask_size, method='nearest')

    panoptic_category_mask = tf.cast(panoptic_category_mask[..., 0], tf.int32)
    panoptic_instance_mask = tf.cast(panoptic_instance_mask[..., 0], tf.int32)

    padding_mask = tf.cast(panoptic_category_mask == -1, tf.float32)

    def get_bbox(mask):
      """Gets the bounding box of a mask."""
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
      """Gets bounding box, center, and labels for a mask."""
      mask = tf.cast(panoptic_instance_mask == instance_id, tf.int32)
      bbox = tf.cast(tf.reshape(get_bbox(mask), [-1]), tf.float32)
      # Computing the mask centers.
      center = tf.reduce_mean(tf.cast(tf.where(mask > 0), tf.float32), axis=0)
      inds = tf.where(panoptic_instance_mask == instance_id)
      thing_classes = tf.gather_nd(panoptic_category_mask, inds)
      thing_classes, _, counts = tf.unique_with_counts(thing_classes)
      max_index = tf.argmax(counts)
      thing_class = tf.cast(thing_classes[max_index], tf.int64)
      return tf.cast(mask, tf.float32), bbox, center, thing_class

    instance_ids, _ = tf.unique(tf.reshape(panoptic_instance_mask, [-1]))
    instance_ids = tf.boolean_mask(instance_ids, instance_ids > 0)
    things_masks, things_bboxes, things_centers, things_classes = tf.map_fn(
        get_thing_class_label,
        instance_ids,
        fn_output_signature=(
            tf.TensorSpec(mask_size),
            tf.TensorSpec([4]),
            tf.TensorSpec([2]),
            tf.int64,
        ),
    )

    stuff_classes, _ = tf.unique(tf.reshape(panoptic_category_mask, [-1]))
    stuff_classes = tf.boolean_mask(
        stuff_classes, stuff_classes > self._num_thing_categories - 1
    )

    def get_stuff_class_label(stuff_class):
      """Gets bounding box, center, and labels for a mask."""
      mask = tf.cast(panoptic_category_mask == stuff_class, tf.int32)
      bbox = tf.cast(tf.reshape(get_bbox(mask), [-1]), tf.float32)
      center = tf.cast(tf.reduce_mean(tf.where(mask > 0), axis=0), tf.float32)
      stuff_class = tf.cast(stuff_class, tf.int64)
      return tf.cast(mask, tf.float32), bbox, center, stuff_class

    stuff_masks, stuff_bboxes, stuff_centers, stuff_classes = tf.map_fn(
        get_stuff_class_label,
        stuff_classes,
        fn_output_signature=(
            tf.TensorSpec(mask_size),
            tf.TensorSpec([4]),
            tf.TensorSpec([2]),
            tf.int64,
        ),
    )

    # Start to generate panoptic heatmap.
    bboxes = tf.concat([things_bboxes, stuff_bboxes], axis=0)
    centers = tf.concat([things_centers, stuff_centers], axis=0)
    classes = tf.concat([things_classes, stuff_classes], axis=0)
    panoptic_masks = tf.concat([things_masks, stuff_masks], axis=0)
    groundtruth_ids = tf.range(tf.shape(panoptic_masks)[0], dtype=tf.int32)

    box_widths = bboxes[..., 3] - bboxes[..., 1]
    box_heights = bboxes[..., 2] - bboxes[..., 0]
    diag_lengths = (tf.sqrt(box_widths**2 + box_heights**2) / 2)[None]

    # Compute according levels for each bounding box.
    fpn_low_range = tf.constant(self.fpn_low_range, tf.float32)[:, None] / (
        2**mask_level
    )
    fpn_high_range = tf.constant(self.fpn_high_range, tf.float32)[:, None] / (
        2**mask_level
    )
    levels = tf.logical_and(
        diag_lengths >= fpn_low_range, diag_lengths <= fpn_high_range
    )

    ct_heatmaps, box_indices = [], []
    offset = 0
    level_ids = []
    level_bboxes, level_classes = [], []
    for level in range(self._min_level, self._max_level + 1):
      level_idx = level - self._min_level
      level_ids.append(tf.where(levels[level_idx]))
      level_bboxes.append(tf.boolean_mask(bboxes, levels[level_idx]))
      level_classes.append(tf.boolean_mask(classes, levels[level_idx]))

      level_center = tf.boolean_mask(centers, levels[level_idx])

      width_ratio = 1 / float(2 ** (level - mask_level))
      height_ratio = 1 / float(2 ** (level - mask_level))

      out_height = int(self._output_size[0] / float(2 ** (level)))
      out_width = int(self._output_size[1] / float(2 ** (level)))

      # Original box coordinates
      # [max_num_instances, ]
      ytl, ybr = level_bboxes[-1][..., 0], level_bboxes[-1][..., 2]
      xtl, xbr = level_bboxes[-1][..., 1], level_bboxes[-1][..., 3]

      # Scaled centers (could be floating point)
      # [max_num_instances, ]
      scale_xct = level_center[:, 1] * width_ratio
      scale_yct = level_center[:, 0] * height_ratio

      # Floor the scaled box coordinates to be placed on heatmaps
      # [max_num_instances, ]
      scale_xct_floor = tf.math.floor(scale_xct)
      scale_yct_floor = tf.math.floor(scale_yct)
      scale_indices_floor = scale_yct_floor * out_width + scale_xct_floor

      # Get the scaled box dimensions for computing the gaussian radius
      # [max_num_instances, ]
      box_widths = (xbr - xtl) * width_ratio
      box_heights = (ybr - ytl) * height_ratio

      ct_heatmap = target_assigner.assign_center_targets(
          out_height=out_height,
          out_width=out_width,
          y_center=scale_yct,
          x_center=scale_xct,
          boxes_height=box_heights,
          boxes_width=box_widths,
          channel_onehot=tf.one_hot(
              tf.cast(level_classes[-1], tf.int32),
              self._num_panoptic_categories, off_value=0.),
          gaussian_iou=self._gaussian_iou)
      ct_heatmaps.append(tf.reshape(
          ct_heatmap, [-1, self._num_panoptic_categories]))
      box_indices.append(
          tf.cast(scale_indices_floor + offset, dtype=tf.int32))
      offset += out_width * out_height

    ct_heatmaps = tf.concat(ct_heatmaps, axis=0)
    box_indices = tf.concat(box_indices, axis=0)
    ids = tf.concat(level_ids, axis=0)[:, 0]
    bboxes = tf.concat(level_bboxes, axis=0)
    classes = tf.concat(level_classes, axis=0)
    panoptic_masks = tf.gather(panoptic_masks, ids)
    groundtruth_ids = tf.gather(groundtruth_ids, ids)

    panoptic_mask_weights = tf.cast(classes >= 0, tf.float32)
    panoptic_mask_weights = preprocess_ops.clip_or_pad_to_fixed_size(
        panoptic_mask_weights, self._max_num_instances, 0
    )
    panoptic_classes = preprocess_ops.clip_or_pad_to_fixed_size(
        classes, self._max_num_instances, -1
    )
    panoptic_masks = preprocess_ops.clip_or_pad_to_fixed_size(
        panoptic_masks, self._max_num_instances, -1.0
    )
    panoptic_masks = tf.transpose(panoptic_masks, [1, 2, 0])
    panoptic_boxes = preprocess_ops.clip_or_pad_to_fixed_size(
        bboxes, self._max_num_instances, -1
    )
    box_indices = preprocess_ops.clip_or_pad_to_fixed_size(
        box_indices, self._max_num_instances, 0
    )
    groundtruth_ids = preprocess_ops.clip_or_pad_to_fixed_size(
        groundtruth_ids, self._max_num_instances, -1
    )

    labels = {}
    labels.update({
        'panoptic_heatmaps': ct_heatmaps,
        'panoptic_classes': panoptic_classes,
        'panoptic_masks': panoptic_masks,
        'panoptic_mask_weights': tf.cast(panoptic_classes >= 0, tf.float32),
        'panoptic_padding_mask': padding_mask,
        'panoptic_boxes': panoptic_boxes,
        'panoptic_box_indices': box_indices,
        'num_instances': tf.reduce_sum(
            tf.cast(panoptic_classes >= 0, tf.float32)
        ),
        'groundtruth_ids': groundtruth_ids,
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
        labels: a dictionary of tensors used for training.
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

    image = data['image']
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)

    # Casts input image to self._dtype
    image = tf.cast(image, dtype=self._dtype)
    groundtruths = {
        'source_id': utils.process_source_id(data['source_id']),
        'height': data['height'],
        'width': data['width'],
    }
    labels = {'image_info': image_info, 'groundtruths': groundtruths}

    panoptic_category_mask = _process_mask(
        data['groundtruth_panoptic_category_mask'],
        self._panoptic_ignore_label,
        image_info,
    )
    panoptic_instance_mask = _process_mask(
        data['groundtruth_panoptic_instance_mask'], 0, image_info
    )

    panoptic_category_mask = panoptic_category_mask[:, :, 0]
    panoptic_instance_mask = panoptic_instance_mask[:, :, 0]

    labels['groundtruths'].update({
        'gt_panoptic_category_mask': tf.cast(
            panoptic_category_mask, dtype=tf.int32
        ),
        'gt_panoptic_instance_mask': tf.cast(
            panoptic_instance_mask, dtype=tf.int32
        ),
    })
    return image, labels
