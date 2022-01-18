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

"""Mosaic op."""
import random
import tensorflow as tf
import tensorflow_addons as tfa

from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.projects.yolo.ops import preprocessing_ops


class Mosaic:
  """Stitch together sets of 4 images to generate samples with more boxes."""

  def __init__(self,
               output_size,
               mosaic_frequency=1.0,
               mixup_frequency=0.0,
               letter_box=True,
               jitter=0.0,
               mosaic_crop_mode='scale',
               mosaic_center=0.25,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               aug_rand_angle=0.0,
               aug_rand_perspective=0.0,
               aug_rand_translate=0.0,
               random_pad=False,
               random_flip=False,
               area_thresh=0.1,
               pad_value=preprocessing_ops.PAD_VALUE,
               seed=None):
    """Initializes parameters for mosaic.

    Args:
      output_size: `Tensor` or `List` for [height, width] of output image.
      mosaic_frequency: `float` indicating how often to apply mosaic.
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
        from the center of the image when creating a mosaic.
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
    self._mixup_frequency = mixup_frequency

    self._letter_box = letter_box
    self._random_crop = jitter

    self._mosaic_crop_mode = mosaic_crop_mode
    self._mosaic_center = mosaic_center

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

  def _generate_cut(self):
    """Generate a random center to use for slicing and patching the images."""
    if self._mosaic_crop_mode == 'crop':
      min_offset = self._mosaic_center
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
      ishape = tf.convert_to_tensor(
          [self._output_size[0] * 2, self._output_size[1] * 2, 3])
    return cut, ishape

  def scale_boxes(self, patch, ishape, boxes, classes, xs, ys):
    """Scale and translate the boxes for each image prior to patching."""
    xs = tf.cast(xs, boxes.dtype)
    ys = tf.cast(ys, boxes.dtype)
    pshape = tf.cast(tf.shape(patch), boxes.dtype)
    ishape = tf.cast(ishape, boxes.dtype)
    translate = tf.cast((ishape - pshape), boxes.dtype)

    boxes = box_ops.denormalize_boxes(boxes, pshape[:2])
    boxes = boxes + tf.cast([
        translate[0] * ys, translate[1] * xs, translate[0] * ys,
        translate[1] * xs
    ], boxes.dtype)
    boxes = box_ops.normalize_boxes(boxes, ishape[:2])
    return boxes, classes

  def _select_ind(self, inds, *args):
    items = []
    for item in args:
      items.append(tf.gather(item, inds))
    return items

  def _augment_image(self,
                     image,
                     boxes,
                     classes,
                     is_crowd,
                     area,
                     xs=0.0,
                     ys=0.0,
                     cut=None):
    """Process a single image prior to the application of patching."""
    if self._random_flip:
      # Randomly flip the image horizontally.
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    # Augment the image without resizing
    image, infos, crop_points = preprocessing_ops.resize_and_jitter_image(
        image, [self._output_size[0], self._output_size[1]],
        random_pad=False,
        letter_box=self._letter_box,
        jitter=self._random_crop,
        shiftx=xs,
        shifty=ys,
        cut=cut,
        seed=self._seed)

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

  def _mosaic_crop_image(self, image, boxes, classes, is_crowd, area):
    """Process a patched image in preperation for final output."""
    if self._mosaic_crop_mode != 'crop':
      shape = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)
      center = shape * self._mosaic_center

      # shift the center of the image by applying a translation to the whole
      # image
      ch = tf.math.round(
          preprocessing_ops.random_uniform_strong(
              -center[0], center[0], seed=self._seed))
      cw = tf.math.round(
          preprocessing_ops.random_uniform_strong(
              -center[1], center[1], seed=self._seed))

      # clip the boxes to those with in the image
      image = tfa.image.translate(image, [cw, ch], fill_value=self._pad_value)
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
  def _process_image(self, sample, shiftx, shifty, cut, ishape):
    """Process and augment each image."""
    (image, boxes, classes, is_crowd, area, crop_points) = self._augment_image(
        sample['image'], sample['groundtruth_boxes'],
        sample['groundtruth_classes'], sample['groundtruth_is_crowd'],
        sample['groundtruth_area'], shiftx, shifty, cut)

    (boxes, classes) = self.scale_boxes(image, ishape, boxes, classes,
                                        1 - shiftx, 1 - shifty)

    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = classes
    sample['groundtruth_is_crowd'] = is_crowd
    sample['groundtruth_area'] = area
    sample['shiftx'] = shiftx
    sample['shifty'] = shifty
    sample['crop_points'] = crop_points
    return sample

  def _patch2(self, one, two):
    """Stitch together 2 images in totality."""
    sample = one
    sample['image'] = tf.concat([one['image'], two['image']], axis=-2)

    sample['groundtruth_boxes'] = tf.concat(
        [one['groundtruth_boxes'], two['groundtruth_boxes']], axis=0)
    sample['groundtruth_classes'] = tf.concat(
        [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
    sample['groundtruth_is_crowd'] = tf.concat(
        [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
    sample['groundtruth_area'] = tf.concat(
        [one['groundtruth_area'], two['groundtruth_area']], axis=0)
    return sample

  def _patch(self, one, two):
    """Build the full 4 patch of images from sets of 2 images."""
    image = tf.concat([one['image'], two['image']], axis=-3)
    boxes = tf.concat([one['groundtruth_boxes'], two['groundtruth_boxes']],
                      axis=0)
    classes = tf.concat(
        [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
    is_crowd = tf.concat(
        [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
    area = tf.concat([one['groundtruth_area'], two['groundtruth_area']], axis=0)

    if self._mosaic_crop_mode is not None:
      image, boxes, classes, is_crowd, area, _ = self._mosaic_crop_image(
          image, boxes, classes, is_crowd, area)

    sample = one
    height, width = preprocessing_ops.get_image_shape(image)
    sample['image'] = tf.cast(image, tf.uint8)
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_area'] = area
    sample['groundtruth_classes'] = tf.cast(classes,
                                            sample['groundtruth_classes'].dtype)
    sample['groundtruth_is_crowd'] = tf.cast(is_crowd, tf.bool)
    sample['width'] = tf.cast(width, sample['width'].dtype)
    sample['height'] = tf.cast(height, sample['height'].dtype)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[1]
    sample['is_mosaic'] = tf.cast(1.0, tf.bool)

    del sample['shiftx']
    del sample['shifty']
    del sample['crop_points']
    return sample

  def _mosaic(self, one, two, three, four):
    """Stitch together 4 images to build a mosaic."""
    if self._mosaic_frequency >= 1.0:
      domo = 1.0
    else:
      domo = preprocessing_ops.random_uniform_strong(
          0.0, 1.0, dtype=tf.float32, seed=self._seed)
      noop = one.copy()

    if domo >= (1 - self._mosaic_frequency):
      cut, ishape = self._generate_cut()
      one = self._process_image(one, 1.0, 1.0, cut, ishape)
      two = self._process_image(two, 0.0, 1.0, cut, ishape)
      three = self._process_image(three, 1.0, 0.0, cut, ishape)
      four = self._process_image(four, 0.0, 0.0, cut, ishape)
      patch1 = self._patch2(one, two)
      patch2 = self._patch2(three, four)
      stitched = self._patch(patch1, patch2)
      return stitched
    else:
      return self._add_param(noop)

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
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    return sample

  def _apply(self, dataset):
    """Apply mosaic to an input dataset."""
    determ = self._deterministic
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    one = dataset.shuffle(100, seed=self._seed, reshuffle_each_iteration=True)
    two = dataset.shuffle(
        100, seed=self._seed + 1, reshuffle_each_iteration=True)
    three = dataset.shuffle(
        100, seed=self._seed + 2, reshuffle_each_iteration=True)
    four = dataset.shuffle(
        100, seed=self._seed + 3, reshuffle_each_iteration=True)

    dataset = tf.data.Dataset.zip((one, two, three, four))
    dataset = dataset.map(
        self._mosaic, num_parallel_calls=tf.data.AUTOTUNE, deterministic=determ)

    if self._mixup_frequency > 0:
      one = dataset.shuffle(
          100, seed=self._seed + 4, reshuffle_each_iteration=True)
      two = dataset.shuffle(
          100, seed=self._seed + 5, reshuffle_each_iteration=True)
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
