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

"""Panoptic CopyPaste."""
import random

import tensorflow as tf, tf_keras

from official.vision.ops import preprocess_ops


GLOBAL_SEED_SET = False
PAD_VALUE = 0


def random_uniform_strong(minval,
                          maxval,
                          dtype=tf.float32,
                          seed=None,
                          shape=None):
  """A unified function for consistent random number generation.

  Equivalent to tf.random.uniform, except that minval and maxval are flipped if
  minval is greater than maxval. Seed Safe random number generator.

  Args:
    minval: An `int` for a lower or upper endpoint of the interval from which to
      choose the random number.
    maxval: An `int` for the other endpoint.
    dtype: The output type of the tensor.
    seed: An `int` used to set the seed.
    shape: List or 1D tf.Tensor, output shape of the random generator.

  Returns:
    A random tensor of type `dtype` that falls between `minval` and `maxval`
    excluding the larger one.
  """
  if GLOBAL_SEED_SET:
    seed = None

  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform(
      shape=shape or [], minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class CopyPaste:
  """Panoptic CopyPaste."""

  def __init__(self,
               output_size,
               copypaste_frequency=0.5,
               stuff_mask_drop_rate=0.0,
               copypaste_aug_scale_max=1.0,
               copypaste_aug_scale_min=1.0,
               num_thing_classes=91,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               random_flip=False,
               pad_value=PAD_VALUE,
               seed=None):
    """Initializes parameters for Copy Paste.

    Args:
      output_size: `Tensor` or `List` for [height, width] of output image.
      copypaste_frequency: `float` indicating how often to apply copypaste.
      stuff_mask_drop_rate: `float` indicating drop rate for stuff masks.
      copypaste_aug_scale_max: `float`, how much to scale the copypaste
        image.
      copypaste_aug_scale_min: `float`, how much to scale the copypaste
        image.
      num_thing_classes: `int`, number of thing classes.
      aug_scale_min: `float` indicating the minimum scaling value for image
        scale jitter.
      aug_scale_max: `float` indicating the maximum scaling value for image
        scale jitter.
      random_flip: `bool` whether or not to random flip the image.
      pad_value: `int` padding value.
      seed: `int` the seed for random number generation.
    """

    self._output_size = output_size
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._copypaste_aug_scale_min = copypaste_aug_scale_min
    self._copypaste_aug_scale_max = copypaste_aug_scale_max
    self._random_flip = random_flip
    self._pad_value = pad_value
    self._copypaste_frequency = copypaste_frequency
    self._stuff_mask_drop_rate = stuff_mask_drop_rate
    self._num_thing_classes = num_thing_classes

    self._deterministic = seed is not None
    self._seed = seed if seed is not None else random.randint(0, 2**30)

  def _process_image(self, sample, aug_min, aug_max, seed=None):
    """Process and augment each image."""
    if self._random_flip:
      instance_mask = sample['groundtruth_panoptic_instance_mask']
      category_mask = sample['groundtruth_panoptic_category_mask']
      image_mask = tf.concat(
          [tf.cast(sample['image'], tf.uint8), category_mask, instance_mask],
          axis=2)

      image_mask, _, _ = preprocess_ops.random_horizontal_flip(
          image_mask)

      instance_mask = image_mask[:, :, -1:]
      category_mask = image_mask[:, :, -2:-1]
      image = tf.cast(image_mask[:, :, :-2], tf.uint8)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=aug_min,
        aug_scale_max=aug_max,
        seed=seed)

    def _process_mask(mask, ignore_label, image_info):
      mask = tf.cast(mask, dtype=tf.float32)
      mask = tf.reshape(mask, shape=[1, sample['height'], sample['width'], 1])
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

    panoptic_category_mask = _process_mask(
        category_mask,
        0, image_info)
    panoptic_instance_mask = _process_mask(
        instance_mask,
        0, image_info)
    sample['image'] = image
    sample['height'] = tf.cast(self._output_size[0], tf.int32)
    sample['width'] = tf.cast(self._output_size[1], tf.int32)
    sample['groundtruth_panoptic_category_mask'] = panoptic_category_mask
    sample['groundtruth_panoptic_instance_mask'] = panoptic_instance_mask

    return sample

  def _patch(self, one, two):
    """Stitch together 2 images in totality."""
    sample = one
    unique_instance_ids, _ = tf.unique(
        tf.reshape(two['groundtruth_panoptic_instance_mask'], [-1]))
    first_image = one['image']
    first_instance_mask = one['groundtruth_panoptic_instance_mask']
    second_instance_mask = two['groundtruth_panoptic_instance_mask']
    first_category_mask = one['groundtruth_panoptic_category_mask']
    second_category_mask = two['groundtruth_panoptic_category_mask']
    max_id = tf.reduce_max(one['groundtruth_panoptic_instance_mask'])

    for inst_id in unique_instance_ids:
      num = random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed)
      if tf.logical_and(inst_id > 0, num < self._copypaste_frequency):
        first_instance_mask = tf.where(second_instance_mask == inst_id,
                                       second_instance_mask + max_id + 1,
                                       first_instance_mask)
        first_image = tf.where(second_instance_mask == inst_id, two['image'],
                               first_image)
        first_category_mask = tf.where(second_instance_mask == inst_id,
                                       second_category_mask,
                                       first_category_mask)
    stuff_classes, _ = tf.unique(
        tf.reshape(two['groundtruth_panoptic_category_mask'], [-1]))
    stuff_classes = tf.boolean_mask(
        stuff_classes, stuff_classes >= self._num_thing_classes)

    for stuff_class in stuff_classes:
      num = random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed)
      if num < self._copypaste_frequency:
        random_tensor = tf.random.uniform(
            self._output_size + [1], minval=0.0, maxval=1.0, seed=self._seed)
        stuff_mask_to_copy = tf.logical_and(
            second_category_mask == stuff_class,
            random_tensor > self._stuff_mask_drop_rate)
        first_image = tf.where(stuff_mask_to_copy, two['image'],
                               first_image)
        first_category_mask = tf.where(stuff_mask_to_copy,
                                       second_category_mask,
                                       first_category_mask)
        first_instance_mask = tf.where(stuff_mask_to_copy,
                                       tf.zeros_like(first_instance_mask),
                                       first_instance_mask)

    sample['image'] = first_image
    sample['groundtruth_panoptic_instance_mask'] = first_instance_mask
    sample['groundtruth_panoptic_category_mask'] = first_category_mask

    sample['image'] = tf.cast(first_image, tf.uint8)
    return sample

  def _copypaste(self, one, two):
    """Apply copypaste on 2 images."""
    one = self._process_image(one, self._aug_scale_min, self._aug_scale_max,
                              self._seed)
    two = self._process_image(
        two, self._copypaste_aug_scale_min, self._copypaste_aug_scale_max,
        self._seed + 1)
    copypasted = self._patch(one, two)
    return copypasted

  def _apply(self, dataset):
    """Apply copypaste to an input dataset."""
    determ = self._deterministic
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    one = dataset.shuffle(1000, seed=self._seed, reshuffle_each_iteration=True)
    two = dataset.shuffle(
        1000, seed=self._seed + 1, reshuffle_each_iteration=True)

    dataset = tf.data.Dataset.zip((one, two))
    dataset = dataset.map(
        self._copypaste,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=determ)

    return dataset

  def copypaste_fn(self, is_training=True):
    """Determine which function to apply based on whether model is training."""
    if is_training:
      return self._apply
    else:
      return lambda dataset: dataset
