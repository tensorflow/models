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

"""Data parser and processing for Panoptic Deeplab."""

import numpy as np
import tensorflow as tf

from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.ops import preprocess_ops


def _compute_gaussian_from_std(sigma):
  """Computes the Gaussian and its size from a given standard deviation."""
  size = int(6 * sigma + 3)
  x = np.arange(size, dtype=np.float)
  y = x[:, np.newaxis]
  x0, y0 = 3 * sigma + 1, 3 * sigma + 1
  gaussian = tf.constant(
      np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)),
      dtype=tf.float32)
  return gaussian, size


class TfExampleDecoder(tf_example_decoder.TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self, regenerate_source_id):
    super(TfExampleDecoder,
          self).__init__(
              include_mask=True,
              regenerate_source_id=regenerate_source_id)

    self._segmentation_keys_to_features = {
        'image/panoptic/category_mask':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/panoptic/instance_mask':
            tf.io.FixedLenFeature((), tf.string, default_value='')
    }

  def decode(self, serialized_example):
    decoded_tensors = super(TfExampleDecoder,
                            self).decode(serialized_example)
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, self._segmentation_keys_to_features)

    category_mask = tf.io.decode_image(
        parsed_tensors['image/panoptic/category_mask'], channels=1)
    instance_mask = tf.io.decode_image(
        parsed_tensors['image/panoptic/instance_mask'], channels=1)
    category_mask.set_shape([None, None, 1])
    instance_mask.set_shape([None, None, 1])

    decoded_tensors.update({
        'groundtruth_panoptic_category_mask': category_mask,
        'groundtruth_panoptic_instance_mask': instance_mask
    })
    return decoded_tensors


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      output_size,
      resize_eval_groundtruth=True,
      groundtruth_padded_size=None,
      ignore_label=0,
      aug_rand_hflip=False,
      aug_scale_min=1.0,
      aug_scale_max=1.0,
      sigma=8.0,
      dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      resize_eval_groundtruth: `bool`, if True, eval groundtruth masks are
        resized to output_size.
      groundtruth_padded_size: `Tensor` or `list` for [height, width]. When
        resize_eval_groundtruth is set to False, the groundtruth masks are
        padded to this size.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      sigma: `float`, standard deviation for generating 2D Gaussian to encode
        centers.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._output_size = output_size
    self._resize_eval_groundtruth = resize_eval_groundtruth
    if (not resize_eval_groundtruth) and (groundtruth_padded_size is None):
      raise ValueError(
          'groundtruth_padded_size ([height, width]) needs to be'
          'specified when resize_eval_groundtruth is False.')
    self._groundtruth_padded_size = groundtruth_padded_size
    self._ignore_label = ignore_label

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # dtype.
    self._dtype = dtype

    self._sigma = sigma
    self._gaussian, self._gaussian_size = _compute_gaussian_from_std(
        self._sigma)
    self._gaussian = tf.reshape(self._gaussian, shape=[-1])

  def _resize_and_crop_mask(self, mask, image_info, is_training):
    """Resizes and crops mask using `image_info` dict"""
    height = image_info[0][0]
    width = image_info[0][1]
    mask = tf.reshape(mask, shape=[1, height, width, 1])
    mask += 1

    if is_training or self._resize_eval_groundtruth:
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      mask = preprocess_ops.resize_and_crop_masks(
          mask,
          image_scale,
          self._output_size,
          offset)
    else:
      mask = tf.image.pad_to_bounding_box(
          mask, 0, 0,
          self._groundtruth_padded_size[0],
          self._groundtruth_padded_size[1])
    mask -= 1
    # Assign ignore label to the padded region.
    mask = tf.where(
        tf.equal(mask, -1),
        self._ignore_label * tf.ones_like(mask),
        mask)
    mask = tf.squeeze(mask, axis=0)
    return mask

  def _parse_train_data(self, data):
    """Parses data for training."""
    image = data['image']
    image = preprocess_ops.normalize_image(image)

    category_mask = tf.cast(
        data['groundtruth_panoptic_category_mask'][:, :, 0],
        dtype=tf.float32)
    instance_mask = tf.cast(
        data['groundtruth_panoptic_instance_mask'][:, :, 0],
        dtype=tf.float32)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      masks = tf.stack([category_mask, instance_mask], axis=0)
      image, _, masks = preprocess_ops.random_horizontal_flip(
          image=image, masks=masks)
      category_mask = masks[0]
      instance_mask = masks[1]

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    category_mask = self._resize_and_crop_mask(
        category_mask,
        image_info,
        is_training=True)
    instance_mask = self._resize_and_crop_mask(
        instance_mask,
        image_info,
        is_training=True)

    centers_heatmap, centers_offset = self._encode_centers_and_offets(
        instance_mask=instance_mask[:, :, 0])

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)
    category_mask = tf.cast(category_mask, dtype=self._dtype)
    instance_mask = tf.cast(instance_mask, dtype=self._dtype)
    centers_heatmap = tf.cast(centers_heatmap, dtype=self._dtype)
    centers_offset = tf.cast(centers_offset, dtype=self._dtype)
    things_mask = tf.cast(
        tf.not_equal(instance_mask, self._ignore_label),
        dtype=self._dtype)

    labels = {
        'category_mask': category_mask,
        'instance_mask': instance_mask,
        'centers_heatmap': centers_heatmap,
        'centers_offset': centers_offset,
        'things_mask': things_mask
    }
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation."""
    image = data['image']
    image = preprocess_ops.normalize_image(image)

    # shape of masks: [H, W]
    category_mask = tf.cast(
        data['groundtruth_panoptic_category_mask'][:, :, 0],
        dtype=tf.float32)
    instance_mask = tf.cast(
        data['groundtruth_panoptic_instance_mask'][:, :, 0],
        dtype=tf.float32)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        self._output_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)

    category_mask = self._resize_and_crop_mask(
        category_mask,
        image_info,
        is_training=False)
    instance_mask = self._resize_and_crop_mask(
        instance_mask,
        image_info,
        is_training=False)

    centers_heatmap, centers_offset = self._encode_centers_and_offets(
        instance_mask=instance_mask[:, :, 0])

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)
    category_mask = tf.cast(category_mask, dtype=self._dtype)
    instance_mask = tf.cast(instance_mask, dtype=self._dtype)
    centers_heatmap = tf.cast(centers_heatmap, dtype=self._dtype)
    centers_offset = tf.cast(centers_offset, dtype=self._dtype)
    things_mask = tf.cast(
        tf.not_equal(instance_mask, self._ignore_label),
        dtype=self._dtype)

    labels = {
        'category_mask': category_mask,
        'instance_mask': instance_mask,
        'centers_heatmap': centers_heatmap,
        'centers_offset': centers_offset,
        'things_mask': things_mask
    }
    return image, labels

  def _encode_centers_and_offets(self, instance_mask):
    """Generates center heatmaps and offets from instance id mask

    Args:
      instance_mask: `tf.Tensor` of shape [height, width] representing
        groundtruth instance id mask.
    Returns:
      centers_heatmap: `tf.Tensor` of shape [height, width, 1]
      centers_offset: `tf.Tensor` of shape [height, width, 2]
    """
    shape = tf.shape(instance_mask)
    height, width = shape[0], shape[1]

    padding_start = int(3 * self._sigma + 1)
    padding_end = int(3 * self._sigma + 2)
    padding = padding_start + padding_end

    centers_heatmap = tf.zeros(
        shape=[height + padding, width + padding],
        dtype=self._dtype)
    centers_offset_y = tf.zeros(
        shape=[height, width])
    centers_offset_x = tf.zeros(
        shape=[height, width])

    unique_instance_ids, _ = tf.unique(tf.reshape(instance_mask, [-1]))

    # The following method for encoding center heatmaps and offets is inspired
    # by the reference implementation available at 
    # https://github.com/google-research/deeplab2/blob/main/data/sample_generator.py  # pylint: disable=line-too-long
    for instance_id in unique_instance_ids:
      if instance_id == self._ignore_label:
        continue

      mask = tf.equal(instance_mask, instance_id)
      mask_indices = tf.cast(tf.where(mask), dtype=tf.float32)
      mask_center = tf.reduce_mean(mask_indices, axis=0)
      mask_center_y = tf.cast(tf.round(mask_center[0]), dtype=tf.int32)
      mask_center_x = tf.cast(tf.round(mask_center[1]), dtype=tf.int32)

      gaussian_size = self._gaussian_size
      indices_y = tf.range(mask_center_y, mask_center_y + gaussian_size)
      indices_x = tf.range(mask_center_x, mask_center_x + gaussian_size)

      indices = tf.stack(tf.meshgrid(indices_y, indices_x))
      indices = tf.reshape(
          indices, shape=[2, gaussian_size * gaussian_size])
      indices = tf.transpose(indices)

      centers_heatmap = tf.tensor_scatter_nd_max(
          tensor=centers_heatmap,
          indices=indices,
          updates=self._gaussian)

      centers_offset_y = tf.tensor_scatter_nd_update(
          tensor=centers_offset_y,
          indices=tf.cast(mask_indices, dtype=tf.int32),
          updates=tf.cast(mask_center_y, dtype=tf.float32) - mask_indices[:, 0])

      centers_offset_x = tf.tensor_scatter_nd_update(
          tensor=centers_offset_x,
          indices=tf.cast(mask_indices, dtype=tf.int32),
          updates=tf.cast(mask_center_x, dtype=tf.float32) - mask_indices[:, 1])

    centers_heatmap = centers_heatmap[
        padding_start:padding_start + height,
        padding_start:padding_start + width]
    centers_heatmap = tf.expand_dims(centers_heatmap, axis=-1)
    centers_offset = tf.stack([centers_offset_y, centers_offset_x], axis=-1)
    return centers_heatmap, centers_offset
