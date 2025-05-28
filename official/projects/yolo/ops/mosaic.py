# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Mosaic op."""
import random

import tensorflow as tf, tf_keras

from official.projects.yolo.ops import preprocessing_ops
from official.vision.ops import augment
from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops


class Mosaic:
  """Stitch together sets of 4 (2x2) or 9 (3x3) images to generate samples with more boxes."""

  def __init__(
      self,
      output_size,
      mosaic_frequency=1.0,
      mosaic9_frequency=0.0,
      mixup_frequency=0.0,
      letter_box=True,
      jitter=0.0,
      mosaic_crop_mode='scale',
      mosaic_center=0.25,
      mosaic9_center=0.33,
      aug_scale_min=1.0,
      aug_scale_max=1.0,
      aug_rand_angle=0.0,
      aug_rand_perspective=0.0,
      aug_rand_translate=0.0,
      random_pad=False,
      random_flip=False,
      area_thresh=0.1,
      pad_value=preprocessing_ops.PAD_VALUE,
      seed=None,
  ):
    """Initializes parameters for mosaic.

    Args:
      output_size: `Tensor` or `List` for [height, width] of output image.
      mosaic_frequency: `float` indicating how often to apply mosaic.
      mosaic9_frequency: `float` indicating how often to apply a 3x3 mosaic
        instead of 2x2.
      mixup_frequency: `float` indicating how often to apply mixup.
      letter_box: `boolean` indicating whether upon start of the datapipeline
        regardless of the preprocessing ops that are used, the aspect ratio of
        the images should be preserved.
      jitter: `float` for the maximum change in aspect ratio expected in each
        preprocessing step.
      mosaic_crop_mode: `str` the type of mosaic to apply. The options are
        {crop, scale, None}, crop will construct a mosaic by slicing images
        togther, scale will create a mosaic by concatnating and shifting the
        image, and None will default to scale and apply no post processing to
        the created mosaic.
      mosaic_center: `float` indicating how much to randomly deviate from the
        center of the image when creating a mosaic.
      mosaic9_center: `float` indicating how much to randomly deviate from the
        center of the image when creating a mosaic9.
      aug_scale_min: `float` indicating the minimum scaling value for image
        scale jitter.
      aug_scale_max: `float` indicating the maximum scaling value for image
        scale jitter.
      aug_rand_angle: `float` indicating the maximum angle value for angle.
        angle will be changes between 0 and value.
      aug_rand_perspective: `float` ranging from 0.000 to 0.001 indicating how
        much to prespective warp the image.
      aug_rand_translate: `float` ranging from 0 to 1 indicating the maximum
        amount to randomly translate an image.
      random_pad: `bool` indiccating wether to use padding to apply random
        translation true for darknet yolo false for scaled yolo.
      random_flip: `bool` whether or not to random flip the image.
      area_thresh: `float` for the minimum area of a box to allow to pass
        through for optimization.
      pad_value: `int` padding value.
      seed: `int` the seed for random number generation.
    """

    self._output_size = output_size
    self._area_thresh = area_thresh

    self._mosaic_frequency = mosaic_frequency
    self._mosaic9_frequency = mosaic9_frequency
    self._mixup_frequency = mixup_frequency

    self._letter_box = letter_box
    self._random_crop = jitter

    self._mosaic_crop_mode = mosaic_crop_mode
    self._mosaic_center = mosaic_center
    self._mosaic9_center = mosaic9_center

    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._random_pad = random_pad
    self._aug_rand_translate = aug_rand_translate
    self._aug_rand_angle = aug_rand_angle
    self._aug_rand_perspective = aug_rand_perspective
    self._random_flip = random_flip
    self._pad_value = pad_value

    self._deterministic = seed is not None
    self._seed = seed if seed is not None else random.randint(0, 2**30)

  def _generate_cut(self, num_tiles, mosaic_center):
    """Generate a random center to use for slicing and patching the images."""
    if self._mosaic_crop_mode == 'crop':
      min_offset = mosaic_center
      cut_x = preprocessing_ops.random_uniform_strong(
          self._output_size[1] * min_offset,
          self._output_size[1] * (1 - min_offset),
          seed=self._seed)
      cut_y = preprocessing_ops.random_uniform_strong(
          self._output_size[0] * min_offset,
          self._output_size[0] * (1 - min_offset),
          seed=self._seed)
      cut = [cut_y, cut_x]
      ishape = tf.convert_to_tensor(
          [self._output_size[0], self._output_size[1], 3])
    else:
      cut = None
      ishape = tf.convert_to_tensor([
          self._output_size[0] * num_tiles,
          self._output_size[1] * num_tiles,
          3,
      ])
    return cut, ishape

  def scale_boxes(self, patch, ishape, boxes, x_offset, y_offset):
    """Scale and translate the boxes for each image prior to patching."""
    x_offset = tf.cast(x_offset, boxes.dtype)
    y_offset = tf.cast(y_offset, boxes.dtype)
    pshape = tf.cast(tf.shape(patch), boxes.dtype)
    ishape = tf.cast(ishape, boxes.dtype)
    y_offset = ishape[0] * y_offset
    x_offset = ishape[1] * x_offset

    boxes = box_ops.denormalize_boxes(boxes, pshape[:2])
    boxes = boxes + tf.cast(
        [y_offset, x_offset, y_offset, x_offset], boxes.dtype
    )
    boxes = box_ops.normalize_boxes(boxes, ishape[:2])
    return boxes

  def _select_ind(self, inds, *args):
    items = []
    for item in args:
      items.append(tf.gather(item, inds))
    return items

  def _augment_image(
      self,
      image,
      boxes,
      classes,
      is_crowd,
      area,
      xs=0.0,
      ys=0.0,
      cut=None,
      letter_box=False,
  ):
    """Process a single image prior to the application of patching."""
    if self._random_flip:
      # Randomly flip the image horizontally.
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    # Augment the image without resizing
    image, infos, crop_points = preprocessing_ops.resize_and_jitter_image(
        image,
        [self._output_size[0], self._output_size[1]],
        random_pad=False,
        letter_box=letter_box,
        jitter=self._random_crop,
        shiftx=xs,
        shifty=ys,
        cut=cut,
        seed=self._seed,
    )

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.transform_and_clip_boxes(
        boxes,
        infos,
        area_thresh=self._area_thresh,
        shuffle_boxes=False,
        filter_and_clip_boxes=True,
        seed=self._seed)
    classes, is_crowd, area = self._select_ind(inds, classes, is_crowd, area)  # pylint:disable=unbalanced-tuple-unpacking
    return image, boxes, classes, is_crowd, area, crop_points

  def _mosaic_crop_image(
      self, image, boxes, classes, is_crowd, area, mosaic_center):
    """Process a patched image in preperation for final output."""
    if self._mosaic_crop_mode != 'crop':
      shape = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)
      center = shape * mosaic_center

      # shift the center of the image by applying a translation to the whole
      # image
      ch = tf.math.round(
          preprocessing_ops.random_uniform_strong(
              -center[0], center[0], seed=self._seed))
      cw = tf.math.round(
          preprocessing_ops.random_uniform_strong(
              -center[1], center[1], seed=self._seed))

      # clip the boxes to fit within the image
      image = augment.translate(
          image, [cw, ch], fill_value=self._pad_value, fill_mode='constant'
      )
      boxes = box_ops.denormalize_boxes(boxes, shape[:2])
      boxes = boxes + tf.cast([ch, cw, ch, cw], boxes.dtype)
      boxes = box_ops.clip_boxes(boxes, shape[:2])
      inds = box_ops.get_non_empty_box_indices(boxes)

      boxes = box_ops.normalize_boxes(boxes, shape[:2])
      boxes, classes, is_crowd, area = self._select_ind(inds, boxes, classes,  # pylint:disable=unbalanced-tuple-unpacking
                                                        is_crowd, area)

    # warp and scale the fully stitched sample
    image, _, affine = preprocessing_ops.affine_warp_image(
        image, [self._output_size[0], self._output_size[1]],
        scale_min=self._aug_scale_min,
        scale_max=self._aug_scale_max,
        translate=self._aug_rand_translate,
        degrees=self._aug_rand_angle,
        perspective=self._aug_rand_perspective,
        random_pad=self._random_pad,
        seed=self._seed)
    height, width = self._output_size[0], self._output_size[1]
    image = tf.image.resize(image, (height, width))

    # clip and clean boxes
    boxes, inds = preprocessing_ops.transform_and_clip_boxes(
        boxes,
        None,
        affine=affine,
        area_thresh=self._area_thresh,
        seed=self._seed)
    classes, is_crowd, area = self._select_ind(inds, classes, is_crowd, area)  # pylint:disable=unbalanced-tuple-unpacking
    return image, boxes, classes, is_crowd, area, area

  # mosaic full frequency doubles model speed
  def _process_image(self, sample, shiftx, shifty, cut, letter_box):
    """Process and augment an image."""
    (image, boxes, classes, is_crowd, area, crop_points) = self._augment_image(
        sample['image'],
        sample['groundtruth_boxes'],
        sample['groundtruth_classes'],
        sample['groundtruth_is_crowd'],
        sample['groundtruth_area'],
        shiftx,
        shifty,
        cut,
        letter_box,
    )

    # Make a copy so this method is functional.
    sample = sample.copy()
    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = classes
    sample['groundtruth_is_crowd'] = is_crowd
    sample['groundtruth_area'] = area
    sample['shiftx'] = shiftx
    sample['shifty'] = shifty
    sample['crop_points'] = crop_points
    return sample

  def _update_patched_sample(
      self, sample, image, boxes, classes, is_crowds, areas, mosaic_center
  ):
    """Returns a shallow copy of sample with updated values."""
    boxes = tf.concat(boxes, axis=0)
    classes = tf.concat(classes, axis=0)
    is_crowds = tf.concat(is_crowds, axis=0)
    areas = tf.concat(areas, axis=0)

    if self._mosaic_crop_mode is not None:
      image, boxes, classes, is_crowds, areas, _ = self._mosaic_crop_image(
          image, boxes, classes, is_crowds, areas, mosaic_center
      )

    height, width = preprocessing_ops.get_image_shape(image)
    # Shallow copy of dict is needed to keep this method functional and
    # AutoGraph happy.
    sample = sample.copy()
    sample['image'] = tf.cast(image, tf.uint8)
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_area'] = areas
    sample['groundtruth_classes'] = tf.cast(
        classes, sample['groundtruth_classes'].dtype
    )
    sample['groundtruth_is_crowd'] = tf.cast(is_crowds, tf.bool)
    sample['width'] = tf.cast(width, sample['width'].dtype)
    sample['height'] = tf.cast(height, sample['height'].dtype)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[1]
    sample['is_mosaic'] = tf.cast(1.0, tf.bool)

    del sample['shiftx']
    del sample['shifty']
    del sample['crop_points']

    return sample

  def _patch(self, patches, ishape, num_rows, num_cols, mosaic_center):
    """Combines patches into a num_patches x num_patches mosaic and translates the bounding boxes."""
    rows = []
    for row_idx in range(num_rows):
      row_patches = [
          patches[row_idx * num_cols + col_idx]['image']
          for col_idx in range(num_cols)
      ]
      rows.append(tf.concat(row_patches, axis=-2))
    image = tf.concat(rows, axis=-3)

    boxes = []
    classes = []
    is_crowds = []
    areas = []
    # Shift boxes to their new coordinates in the mosaic.
    for row_idx in range(num_rows):
      for col_idx in range(num_cols):
        patch = patches[row_idx * num_cols + col_idx]
        transformed_boxes = self.scale_boxes(
            patch['image'],
            ishape,
            patch['groundtruth_boxes'],
            col_idx / num_cols,
            row_idx / num_rows,
        )
        boxes.append(transformed_boxes)
        classes.append(patch['groundtruth_classes'])
        is_crowds.append(patch['groundtruth_is_crowd'])
        areas.append(patch['groundtruth_area'])

    return self._update_patched_sample(
        patches[0], image, boxes, classes, is_crowds, areas, mosaic_center
    )

  def _mosaic(self, *patch_samples):
    """Builds a 2x2 or 3x3 mosaic."""
    if self._mosaic_frequency >= 1.0:
      mosaic_prob = 1.0
    else:
      mosaic_prob = preprocessing_ops.random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed
      )
      sample = patch_samples[0].copy()

    if mosaic_prob >= (1 - self._mosaic_frequency):
      mosaic9_prob = preprocessing_ops.random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed + 1
      )
      if self._mosaic9_frequency > 0 and mosaic9_prob >= (
          1 - self._mosaic9_frequency
      ):
        return self._mosaic9(*patch_samples)
      else:
        return self._mosaic4(*patch_samples)
    else:
      return self._add_param(sample)

  def _mosaic4(self, *samples):
    """Stitches together 4 images to build a 2x2 mosaic."""
    cut, ishape = self._generate_cut(2, self._mosaic_center)
    samples = [
        self._process_image(
            samples[0], 1.0, 1.0, cut, letter_box=self._letter_box
        ),
        self._process_image(
            samples[1], 0.0, 1.0, cut, letter_box=self._letter_box
        ),
        self._process_image(
            samples[2], 1.0, 0.0, cut, letter_box=self._letter_box
        ),
        self._process_image(
            samples[3], 0.0, 0.0, cut, letter_box=self._letter_box
        ),
    ]
    stitched = self._patch(samples, ishape, 2, 2, self._mosaic_center)
    return stitched

  def _mosaic9(self, *samples):
    """Stitches together 9 images to build a 3x3 mosaic."""
    cut, ishape = self._generate_cut(3, self._mosaic9_center)
    # Only corner images can be letterboxed to prevent gaps in the image.
    samples = [
        self._process_image(
            samples[0], 1.0, 1.0, cut, letter_box=self._letter_box
        ),
        self._process_image(samples[1], 0.0, 0.0, cut, letter_box=False),
        self._process_image(
            samples[2], 0.0, 1.0, cut, letter_box=self._letter_box
        ),
        self._process_image(samples[3], 0.0, 0.0, cut, letter_box=False),
        self._process_image(samples[4], 0.0, 0.0, cut, letter_box=False),
        self._process_image(samples[5], 0.0, 0.0, cut, letter_box=False),
        self._process_image(
            samples[6], 1.0, 0.0, cut, letter_box=self._letter_box
        ),
        self._process_image(samples[7], 0.0, 0.0, cut, letter_box=False),
        self._process_image(
            samples[8], 0.0, 0.0, cut, letter_box=self._letter_box
        ),
    ]
    stitched = self._patch(samples, ishape, 3, 3, self._mosaic9_center)
    return stitched

  def _beta(self, alpha, beta):
    """Generates a random number using the beta distribution."""
    a = tf.random.gamma([], alpha)
    b = tf.random.gamma([], beta)
    return b / (a + b)

  def _mixup(self, one, two):
    """Blend together 2 images for the mixup data augmentation."""
    if self._mixup_frequency >= 1.0:
      domo = 1.0
    else:
      domo = preprocessing_ops.random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed)
      noop = one.copy()

    if domo >= (1 - self._mixup_frequency):
      sample = one
      otype = one['image'].dtype

      r = self._beta(8.0, 8.0)
      sample['image'] = (
          r * tf.cast(one['image'], tf.float32) +
          (1 - r) * tf.cast(two['image'], tf.float32))

      sample['image'] = tf.cast(sample['image'], otype)
      sample['groundtruth_boxes'] = tf.concat(
          [one['groundtruth_boxes'], two['groundtruth_boxes']], axis=0)
      sample['groundtruth_classes'] = tf.concat(
          [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
      sample['groundtruth_is_crowd'] = tf.concat(
          [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
      sample['groundtruth_area'] = tf.concat(
          [one['groundtruth_area'], two['groundtruth_area']], axis=0)
      return sample
    else:
      return self._add_param(noop)

  def _add_param(self, sample):
    """Add parameters to handle skipped images."""
    if 'is_mosaic' not in sample:
      sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    return sample

  def _apply(self, dataset):
    """Apply mosaic to an input dataset."""
    determ = self._deterministic
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    patch_datasets = []
    num_patches = 9 if self._mosaic9_frequency > 0.0 else 4
    for i in range(num_patches):
      patch_datasets.append(
          dataset.shuffle(
              100, seed=self._seed + i, reshuffle_each_iteration=True
          )
      )

    dataset = tf.data.Dataset.zip(tuple(patch_datasets))
    dataset = dataset.map(
        self._mosaic, num_parallel_calls=tf.data.AUTOTUNE, deterministic=determ)

    if self._mixup_frequency > 0:
      one = dataset.shuffle(
          100, seed=self._seed + num_patches, reshuffle_each_iteration=True
      )
      two = dataset.shuffle(
          100,
          seed=self._seed + num_patches + 1,
          reshuffle_each_iteration=True,
      )
      dataset = tf.data.Dataset.zip((one, two))
      dataset = dataset.map(
          self._mixup,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=determ)
    return dataset

  def _skip(self, dataset):
    """Skip samples in a dataset."""
    determ = self._deterministic
    return dataset.map(
        self._add_param,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=determ)

  def mosaic_fn(self, is_training=True):
    """Determine which function to apply based on whether model is training."""
    if is_training and self._mosaic_frequency > 0.0:
      return self._apply
    else:
      return self._skip
