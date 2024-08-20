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

"""Data parser and processing for maskconver segmentation dataloader."""
from typing import Optional

import tensorflow as tf, tf_keras
from tensorflow_addons import image as tfa_image

from official.projects.centernet.ops import target_assigner
from official.vision.configs import common
from official.vision.dataloaders import segmentation_input
from official.vision.ops import augment
from official.vision.ops import preprocess_ops


class Parser(segmentation_input.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               num_classes,
               crop_size=None,
               level=3,
               resize_eval_groundtruth=True,
               groundtruth_padded_size=None,
               ignore_label=255,
               aug_rand_hflip=False,
               preserve_aspect_ratio=True,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               gaussian_iou=0.7,
               aug_type: Optional[common.Augmentation] = None,
               max_num_instances: int = 100,
               max_num_stuff_centers: int = 3,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      num_classes: `int` for number of classes.
      crop_size: `Tensor` or `list` for [height, width] of the crop. If
        specified a training crop of size crop_size is returned. This is useful
        for cropping original images during training while evaluating on
        original image sizes.
      level: level for output masks.
      resize_eval_groundtruth: `bool`, if True, eval groundtruth masks are
        resized to output_size.
      groundtruth_padded_size: `Tensor` or `list` for [height, width]. When
        resize_eval_groundtruth is set to False, the groundtruth masks are
        padded to this size.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      preserve_aspect_ratio: `bool`, if True, the aspect ratio is preserved,
        otherwise, the image is resized to output_size.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      gaussian_iou: `float`, used for generating the center heatmaps.
      aug_type: Optional[common.Augmentation].
      max_num_instances: max number of instances.
      max_num_stuff_centers: max number of stuff centers.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    super(Parser, self).__init__(
        output_size=output_size,
        crop_size=crop_size,
        resize_eval_groundtruth=resize_eval_groundtruth,
        groundtruth_padded_size=groundtruth_padded_size,
        ignore_label=ignore_label,
        aug_rand_hflip=aug_rand_hflip,
        preserve_aspect_ratio=preserve_aspect_ratio,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max)

    self._num_classes = num_classes
    self.level = level
    self._gaussian_iou = gaussian_iou
    self._max_num_stuff_centers = max_num_stuff_centers
    self._max_num_instances = max_num_instances

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

  def _prepare_image_and_label(self, data, use_augment=False):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(data['image/segmentation/class/encoded'],
                               channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))

    label = tf.reshape(label, (1, height, width))
    label = tf.cast(label, tf.float32)

    if use_augment and self._augmenter is not None:
      image = self._augmenter.distort(image)
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    if not self._preserve_aspect_ratio:
      label = tf.reshape(label, [data['image/height'], data['image/width'], 1])
      image = tf.image.resize(image, self._output_size, method='bilinear')
      label = tf.image.resize(label, self._output_size, method='nearest')
      label = tf.reshape(label[:, :, -1], [1] + self._output_size)

    return image, label

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data, use_augment=True)

    if self._crop_size:
      label = tf.reshape(label, [data['image/height'], data['image/width'], 1])
      # If output_size is specified, resize image, and label to desired
      # output_size.
      if self._output_size:
        image = tf.image.resize(image, self._output_size, method='bilinear')
        label = tf.image.resize(label, self._output_size, method='nearest')

      image_mask = tf.concat([image, label], axis=2)
      image_mask_crop = tf.image.random_crop(image_mask,
                                             self._crop_size + [4])
      image = image_mask_crop[:, :, :-1]
      label = tf.reshape(image_mask_crop[:, :, -1], [1] + self._crop_size)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, _, label = preprocess_ops.random_horizontal_flip(
          image, masks=label)

    train_image_size = self._crop_size if self._crop_size else self._output_size
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        train_image_size,
        train_image_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)
    label = preprocess_ops.resize_and_crop_masks(
        label, image_scale, train_image_size, offset)
    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)
    label = tf.squeeze(label, axis=-1)

    valid_mask = tf.not_equal(label, self._ignore_label)

    stuff_classes, _ = tf.unique(tf.reshape(label, [-1]))
    stuff_classes = tf.boolean_mask(
        stuff_classes, stuff_classes != self._ignore_label)

    # Compute bboxes for stuff classes
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

    def get_stuff_class_label(stuff_class, k=self._max_num_stuff_centers):
      mask = tf.cast(
          label == stuff_class, tf.int32)
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
          fn_output_signature=((tf.TensorSpec(train_image_size, tf.float32),
                                tf.TensorSpec([4,], tf.float32),
                                tf.TensorSpec([1], tf.int32))))

      stuff_classes = tf.reshape(
          stuff_classes, [1, k])
      batch_mask = tf.reshape(batch_mask, [1, k])
      stuff_center_masks = tf.reshape(
          stuff_center_masks, [1, k] + train_image_size)
      return masks[None, :, :, :], bboxes[
          None, :, :], stuff_classes, batch_mask, batch_weights

    centers = self._max_num_stuff_centers
    stuff_masks, stuff_bboxes, stuff_classes, stuff_batch_mask, stuff_batch_weigths = tf.map_fn(
        get_stuff_class_label,
        stuff_classes,
        fn_output_signature=(tf.TensorSpec([1, centers] + train_image_size),
                             tf.TensorSpec([1, centers, 4,]),
                             tf.TensorSpec([1, centers], tf.int64),
                             tf.TensorSpec([1, centers], tf.int32),
                             tf.TensorSpec([1, centers], tf.float32)))
    stuff_masks = tf.reshape(stuff_masks, [-1] + train_image_size)
    stuff_bboxes = tf.reshape(stuff_bboxes, [-1, 4])
    stuff_classes = tf.reshape(stuff_classes, [-1])
    stuff_batch_mask = tf.reshape(stuff_batch_mask, [-1])
    stuff_batch_weigths = tf.reshape(stuff_batch_weigths, [-1])
    masks = stuff_masks[stuff_batch_mask == 1]
    bboxes = stuff_bboxes[stuff_batch_mask == 1]
    classes = stuff_classes[stuff_batch_mask == 1]

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
        out_height=int(train_image_size[0] * height_ratio),
        out_width=int(train_image_size[1]  * width_ratio),
        y_center=scale_yct,
        x_center=scale_xct,
        boxes_height=box_heights,
        boxes_width=box_widths,
        channel_onehot=tf.one_hot(
            tf.cast(classes, tf.int32),
            self._num_classes, off_value=0.),
        gaussian_iou=self._gaussian_iou)
    box_indices = tf.cast(
        tf.stack([scale_yct_floor, scale_xct_floor], axis=-1), dtype=tf.int32)

    seg_classes = preprocess_ops.clip_or_pad_to_fixed_size(
        classes, self._max_num_instances, -1)
    seg_masks = preprocess_ops.clip_or_pad_to_fixed_size(
        masks, self._max_num_instances, -1.0)
    seg_masks = tf.transpose(seg_masks, [1, 2, 0])
    seg_boxes = preprocess_ops.clip_or_pad_to_fixed_size(
        bboxes, self._max_num_instances, -1)

    box_indices = preprocess_ops.clip_or_pad_to_fixed_size(
        box_indices, self._max_num_instances, 0)

    labels = {
        'image': image,
        'seg_ct_heatmaps': ct_heatmap,
        'seg_classes': seg_classes,
        'seg_masks': seg_masks,
        'seg_mask_weights': tf.cast(seg_classes >= 0, tf.float32),
        'seg_valid_mask': tf.cast(valid_mask, tf.float32),
        'seg_boxes': seg_boxes,
        'seg_box_indices': box_indices,
        'num_instances': tf.reduce_sum(
            tf.cast(seg_classes >= 0, tf.float32)),
        'image_info': image_info,
    }
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image, self._output_size, self._output_size)

    if self._resize_eval_groundtruth:
      # Resizes eval masks to match input image sizes. In that case, mean IoU
      # is computed on output_size not the original size of the images.
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      label = preprocess_ops.resize_and_crop_masks(label, image_scale,
                                                   self._output_size, offset)
    else:
      label = tf.image.pad_to_bounding_box(
          label, 0, 0, self._groundtruth_padded_size[0],
          self._groundtruth_padded_size[1])

    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)

    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask,
        'image_info': image_info
    }

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels
