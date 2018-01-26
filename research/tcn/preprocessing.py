# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Image preprocessing helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  TODO(coreylynch): add as a dependency, when slim or tensorflow/models are
  pipfied.
  Source:
  https://raw.githubusercontent.com/tensorflow/models/a9d0e6e8923a4/slim/preprocessing/inception_preprocessing.py

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  TODO(coreylynch): add as a dependency, when slim or tensorflow/models are
  pipfied.
  Source:
  https://raw.githubusercontent.com/tensorflow/models/a9d0e6e8923a4/slim/preprocessing/inception_preprocessing.py

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  TODO(coreylynch): add as a dependency, when slim or tensorflow/models are
  pipfied.
  Source:
  https://raw.githubusercontent.com/tensorflow/models/a9d0e6e8923a4/slim/preprocessing/inception_preprocessing.py

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def crop_center(image):
  """Returns a cropped square image."""
  shape = tf.shape(image)
  new_shape = tf.minimum(shape[0], shape[1])
  offset_y = tf.maximum(shape[0] - shape[1], 0) // 2
  offset_x = tf.maximum(shape[1] - shape[0], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


def pad(image):
  """Returns an image padded to be square."""
  shape = tf.shape(image)
  new_shape = tf.maximum(shape[0], shape[1])
  height = shape[0]
  width = shape[1]
  offset_x = tf.maximum((height-width), 0) // 2
  offset_y = tf.maximum((width-height), 0) // 2
  image = tf.image.pad_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


def pad_200(image):
  """Returns an image padded width-padded with 200 pixels."""
  shape = tf.shape(image)
  image = tf.image.pad_to_bounding_box(
      image, 0, 200, shape[0], shape[1]+400)
  shape = tf.shape(image)
  new_shape = tf.minimum(shape[0], shape[1])
  offset_y = tf.maximum(shape[0] - shape[1], 0) // 2
  offset_x = tf.maximum(shape[1] - shape[0], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


def pad_crop_central(image, central_fraction=0.875):
  """Pads the image to the maximum length, crops the central fraction."""
  # Pad the image to be square.
  image = pad(image)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image = tf.image.central_crop(image, central_fraction=central_fraction)
  return image


def crop_image_by_strategy(image, cropping):
  """Crops an image according to a strategy defined in config.

  Args:
    image: 3-d image tensor.
    cropping: str, name of cropping strategy.
  Returns:
    image: cropped image.
  Raises:
    ValueError: When unknown cropping strategy is specified.
  """
  strategy_to_method = {
      'crop_center': crop_center,
      'pad': pad,
      'pad200': pad_200,
      'pad_crop_central': pad_crop_central
  }
  tf.logging.info('Cropping strategy: %s.' % cropping)
  if cropping not in strategy_to_method:
    raise ValueError('Unknown cropping strategy: %s' % cropping)
  return strategy_to_method[cropping](image)


def scale_augment_crop(image, central_bbox, area_range, min_object_covered):
  """Training time scale augmentation.

  Args:
    image: 3-d float tensor.
    central_bbox: Bounding box defining the central region of interest.
    area_range: Range of allowed areas for the augmented bounding box.
    min_object_covered: Constraint for the fraction of original image in
      augmented bounding box.
  Returns:
    distort_image: The scaled, cropped image.
  """
  (distorted_image, _) = distorted_bounding_box_crop(
      image, central_bbox, area_range=area_range,
      aspect_ratio_range=(1.0, 1.0),
      min_object_covered=min_object_covered)
  # Restore the shape since the dynamic slice based upon the bbox_size loses
  # the third dimension.
  distorted_image.set_shape([None, None, 3])
  return distorted_image


def scale_to_inception_range(image):
  """Scales an image in the range [0,1] to [-1,1] as expected by inception."""
  # Assert that incoming images have been properly scaled to [0,1].
  with tf.control_dependencies(
      [tf.assert_less_equal(tf.reduce_max(image), 1.),
       tf.assert_greater_equal(tf.reduce_min(image), 0.)]):
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def resize_image(image, height, width):
  """Resizes an image to a target height and width."""
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
  image = tf.squeeze(image, [0])
  return image


def crop_or_pad(image, curr_height, curr_width, new, height=True, crop=True):
  """Crops or pads an image.

  Args:
    image: 3-D float32 `Tensor` image.
    curr_height: Int, current height.
    curr_width: Int, current width.
    new: Int, new width or height.
    height: Boolean, cropping or padding for height.
    crop: Boolean, True if we're cropping, False if we're padding.
  Returns:
    image: 3-D float32 `Tensor` image.
  """
  # Crop the image to fit the new shape.
  abs_diff = tf.abs(new-curr_height)//2 if height else tf.abs(new-curr_width)//2
  offset_x = 0 if height else abs_diff
  offset_y = abs_diff if height else 0

  # We process height first, so always pad/crop to new height.
  target_height = new
  # We process height first, so pad/crop to new width only if not doing height.
  target_width = curr_width if height else new

  if crop:
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)
  else:
    image = tf.image.pad_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)
  return image


def get_central_bbox(min_side, new_size):
  """Gets the central bounding box for an image.

  If image is square, returns bounding box [0,0,1,1].
  Otherwise, returns the bounding box containing the central
  smallest side x smallest side square.

  Args:
    min_side: Int, size of smallest side in pixels.
    new_size: Int, resize image to a square of new_size x new_size pixels.
  Returns:
    bbox: A 4-D Int `Tensor`, holding the coordinates of the central bounding
      box.
  """
  max_shape = tf.cast(new_size, tf.float32)
  min_shape = tf.cast(min_side, tf.float32)
  top_xy = ((max_shape-min_shape)/2)/max_shape
  bottom_xy = (min_shape+(max_shape-min_shape)/2)/max_shape
  # Create a bbox for the center region of interest.
  bbox = tf.stack([[[top_xy, top_xy, bottom_xy, bottom_xy]]])
  bbox.set_shape([1, 1, 4])
  return bbox


def pad_to_max(image, max_scale):
  """Pads an image to max_scale times the current center crop size.

  E.g.: For an image with dimensions 1920x1080 and a max_scale of 1.5,
  returns an image that is 1.5 * (1080x1080).

  Args:
    image: 3-D float32 `Tensor` image.
    max_scale: Float, maximum scale of the image, as a multiplier on the
      central bounding box.
  Returns:
    image: 3-D float32 `Tensor` image.
  """
  orig_shape = tf.shape(image)
  orig_height = orig_shape[0]
  orig_width = orig_shape[1]

  # Find the smallest side and corresponding new size.
  min_side = tf.cast(tf.minimum(orig_height, orig_width), tf.float32)
  new_shape = tf.cast(tf.sqrt(max_scale*min_side*min_side), tf.int32)

  # Crop or pad height.
  # pylint: disable=g-long-lambda
  image = tf.cond(
      orig_height >= new_shape,
      lambda: crop_or_pad(
          image, orig_height, orig_width, new_shape, height=True, crop=True),
      lambda: crop_or_pad(
          image, orig_height, orig_width, new_shape, height=True, crop=False))

  # Crop or pad width.
  image = tf.cond(
      orig_width >= new_shape,
      lambda: crop_or_pad(
          image, orig_height, orig_width, new_shape, height=False, crop=True),
      lambda: crop_or_pad(
          image, orig_height, orig_width, new_shape, height=False, crop=False))

  # Get the bounding box of the original centered box in the new resized image.
  original_bounding_box = get_central_bbox(min_side, new_shape)
  return image, original_bounding_box


def scale_up_augmentation(image, max_scale):
  """Scales an image randomly >100% up to some max scale."""
  # Pad to max size.
  image, original_central_bbox = pad_to_max(image, max_scale)

  # Determine area range of the augmented crop, as a percentage of the
  # new max area.
  # aug_max == 100% of new max area.
  aug_max = 1.0
  # aug_min == original_area/new_area == original_area/(max_scale*original_area)
  # == 1/max_scale.
  aug_min = 1.0/max_scale
  area_range = (aug_min, aug_max)
  # Since we're doing >100% scale, always have the full original crop in frame.
  min_object_covered = 1.0
  # Get a random scaled, cropped image.
  image = scale_augment_crop(image, original_central_bbox, area_range,
                             min_object_covered)
  return image


def scale_down_augmentation(image, min_scale):
  """Scales an image randomly <100% down to some min scale."""
  # Crop the center, and consider the whole image the bounding box ROI.
  image = crop_center(image)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  # Determine area range of the augmented crop, as a percentage of the
  # original crop center area.
  # aug_max == 100% of original area.
  area_range = (min_scale, 1.0)
  # Get a random scaled, cropped image.
  image = scale_augment_crop(image, bbox, area_range, min_scale)
  return image


def augment_image_scale(image, min_scale, max_scale, p_scale_up):
  """Training time scale augmentation.

  Args:
    image: 3-d float tensor representing image.
    min_scale: minimum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    max_scale: maximum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    p_scale_up: Fraction of images scaled up.
  Returns:
    image: The scale-augmented image.
  """
  assert max_scale >= 1.0
  assert min_scale <= 1.0
  if min_scale == max_scale == 1.0:
    tf.logging.info('Min and max scale are 1.0, don`t augment.')
    # Do no augmentation, just crop the center.
    return crop_center(image)
  elif (max_scale == 1.0) and (min_scale < 1.0):
    tf.logging.info('Max scale is 1.0, only scale down augment.')
    # Always do <100% augmentation.
    return scale_down_augmentation(image, min_scale)
  elif (min_scale == 1.0) and (max_scale > 1.0):
    tf.logging.info('Min scale is 1.0, only scale up augment.')
    # Always do >100% augmentation.
    return scale_up_augmentation(image, max_scale)
  else:
    tf.logging.info('Sample both augmentations.')
    # Choose to scale image up or down.
    rn = tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32)
    image = tf.cond(rn >= p_scale_up,
                    lambda: scale_up_augmentation(image, max_scale),
                    lambda: scale_down_augmentation(image, min_scale))
  return image


def decode_image(image_str):
  """Decodes a jpeg-encoded image string into a image in range [0,1]."""
  # Decode jpeg string into np.uint8 tensor.
  image = tf.image.decode_jpeg(image_str, channels=3)
  # Convert the image to range [0,1].
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image


def decode_images(image_strs):
  """Decodes a tensor of image strings."""
  return tf.map_fn(decode_image, image_strs, dtype=tf.float32)


def preprocess_training_images(images, height, width, min_scale, max_scale,
                               p_scale_up, aug_color=True, fast_mode=True):
  """Preprocesses a batch of images for training.

  This applies training-time scale and color augmentation, crops/resizes,
  and scales images to the [-1,1] range expected by pre-trained Inception nets.

  Args:
    images: A 4-D float32 `Tensor` holding raw images to be preprocessed.
    height: Int, height in pixels to resize image to.
    width: Int, width in pixels to resize image to.
    min_scale: Float, minimum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    max_scale: Float, maximum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    p_scale_up: Float, fraction of images scaled up.
    aug_color: Whether or not to do color augmentation.
    fast_mode: Boolean, avoids slower ops (random_hue and random_contrast).
  Returns:
    preprocessed_images: A 4-D float32 `Tensor` holding preprocessed images.
  """
  def _prepro_train(im):
    """Map this preprocessing function over each image in the batch."""
    return preprocess_training_image(
        im, height, width, min_scale, max_scale, p_scale_up,
        aug_color=aug_color, fast_mode=fast_mode)
  return tf.map_fn(_prepro_train, images)


def preprocess_training_image(
    image, height, width, min_scale, max_scale, p_scale_up,
    aug_color=True, fast_mode=True):
  """Preprocesses an image for training.

  Args:
    image: A 3-d float tensor representing the image.
    height: Target image height.
    width: Target image width.
    min_scale: Minimum scale of bounding box (as a percentage of full
      bounding box) used to crop image during scale augmentation.
    max_scale: Minimum scale of bounding box (as a percentage of full
      bounding box) used to crop image during scale augmentation.
    p_scale_up: Fraction of images to scale >100%.
    aug_color: Whether or not to do color augmentation.
    fast_mode: Avoids slower ops (random_hue and random_contrast).
  Returns:
    scaled_image: An scaled image tensor in the range [-1,1].
  """
  # Get a random scaled, cropped image.
  image = augment_image_scale(image, min_scale, max_scale, p_scale_up)

  # Resize image to desired height, width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
  image = tf.squeeze(image, [0])

  # Optionally augment the color.
  # pylint: disable=g-long-lambda
  if aug_color:
    image = apply_with_random_selector(
        image,
        lambda x, ordering: distort_color(
            x, ordering, fast_mode=fast_mode), num_cases=4)

  # Scale to [-1,1] range as expected by inception.
  scaled_image = scale_to_inception_range(image)
  return scaled_image


def preprocess_test_image(image, height, width, crop_strategy):
  """Preprocesses an image for test/inference.

  Args:
    image: A 3-d float tensor representing the image.
    height: Target image height.
    width: Target image width.
    crop_strategy: String, name of the strategy used to crop test-time images.
      Can be: 'crop_center', 'pad', 'pad_200', 'pad_crop_central'.
  Returns:
    scaled_image: An scaled image tensor in the range [-1,1].
  """
  image = crop_image_by_strategy(image, crop_strategy)
  # Resize.
  image = resize_image(image, height, width)
  # Scale the input range to [-1,1] as expected by inception.
  image = scale_to_inception_range(image)
  return image


def preprocess_test_images(images, height, width, crop_strategy):
  """Apply test-time preprocessing to a batch of images.

  This crops images (given a named strategy for doing so), resizes them,
  and scales them to the [-1,1] range expected by pre-trained Inception nets.

  Args:
    images: A 4-D float32 `Tensor` holding raw images to be preprocessed.
    height: Int, height in pixels to resize image to.
    width: Int, width in pixels to resize image to.
    crop_strategy: String, name of the strategy used to crop test-time images.
      Can be: 'crop_center', 'pad', 'pad_200', 'pad_crop_central'.
  Returns:
    preprocessed_images: A 4-D float32 `Tensor` holding preprocessed images.
  """
  def _prepro_test(im):
    """Map this preprocessing function over each image in the batch."""
    return preprocess_test_image(im, height, width, crop_strategy)
  if len(images.shape) == 3:
    return _prepro_test(images)
  else:
    return tf.map_fn(_prepro_test, images)


def preprocess_images(
    images, is_training, height, width,
    min_scale=1.0, max_scale=1.0, p_scale_up=0.0,
    aug_color=True, fast_mode=True,
    crop_strategy='pad_crop_central'):
  """Preprocess a batch of images.

  Args:
    images: A 4-D float32 `Tensor` holding raw images to be preprocessed.
    is_training: Boolean, whether to preprocess them for training or test.
    height: Int, height in pixels to resize image to.
    width: Int, width in pixels to resize image to.
    min_scale: Float, minimum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    max_scale: Float, maximum scale augmentation allowed, as a fraction of the
      central min_side * min_side area of the original image.
    p_scale_up: Float, fraction of images scaled up.
    aug_color: Whether or not to do color augmentation.
    fast_mode: Boolean, avoids slower ops (random_hue and random_contrast).
    crop_strategy: String, name of the strategy used to crop test-time images.
      Can be: 'crop_center', 'pad', 'pad_200', 'pad_crop_central'.
  Returns:
    preprocessed_images: A 4-D float32 `Tensor` holding preprocessed images.
  """
  if is_training:
    return preprocess_training_images(
        images, height, width, min_scale, max_scale,
        p_scale_up, aug_color, fast_mode)
  else:
    return preprocess_test_images(
        images, height, width, crop_strategy)


def cv2rotateimage(image, angle):
  """Efficient rotation if 90 degrees rotations, slow otherwise.

  Not a tensorflow function, using cv2 and scipy on numpy arrays.

  Args:
    image: a numpy array with shape [height, width, channels].
    angle: the rotation angle in degrees in the range [-180, 180].
  Returns:
    The rotated image.
  """
  # Limit angle to [-180, 180] degrees.
  assert angle <= 180 and angle >= -180
  if angle == 0:
    return image
  # Efficient rotations.
  if angle == -90:
    image = cv2.transpose(image)
    image = cv2.flip(image, 0)
  elif angle == 90:
    image = cv2.transpose(image)
    image = cv2.flip(image, 1)
  elif angle == 180 or angle == -180:
    image = cv2.flip(image, 0)
    image = cv2.flip(image, 1)
  else:  # Slow rotation.
    image = ndimage.interpolation.rotate(image, 270)
  return image


def cv2resizeminedge(image, min_edge_size):
  """Resize smallest edge of image to min_edge_size."""
  assert min_edge_size >= 0
  height, width = (image.shape[0], image.shape[1])
  new_height, new_width = (0, 0)
  if height > width:
    new_width = min_edge_size
    new_height = int(height * new_width / float(width))
  else:
    new_height = min_edge_size
    new_width = int(width * new_height / float(height))
  return cv2.resize(image, (new_width, new_height),
                    interpolation=cv2.INTER_AREA)


def shapestring(array):
  """Returns a compact string describing shape of an array."""
  shape = array.shape
  s = str(shape[0])
  for i in range(1, len(shape)):
    s += 'x' + str(shape[i])
  return s


def unscale_jpeg_encode(ims):
  """Unscales pixel values and jpeg encodes preprocessed image.

  Args:
    ims: A 4-D float32 `Tensor` holding preprocessed images.
  Returns:
    im_strings: A 1-D string `Tensor` holding images that have been unscaled
      (reversing the inception [-1,1] scaling), and jpeg encoded.
  """
  ims /= 2.0
  ims += 0.5
  ims *= 255.0
  ims = tf.clip_by_value(ims, 0, 255)
  ims = tf.cast(ims, tf.uint8)
  im_strings = tf.map_fn(
      lambda x: tf.image.encode_jpeg(x, format='rgb', quality=100),
      ims, dtype=tf.string)
  return im_strings
