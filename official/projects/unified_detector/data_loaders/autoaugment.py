# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719

This library is adapted from:
`models/official/efficientnet/autoaugment.py` of
`https://github.com/tensorflow/tpu`.
Several changes are made. They are inspired by the TIMM library:
`https://github.com/rwightman/pytorch-image-models/`

Changes include:
(1) Random Erasing / Cutout is added, and separated from the random augmentation
    pool (not sampled as an operation).
(2) For `posterize` and `solarize`, the arguments are changed such that the
    level of corruption increases as the `magnitude` argument increases.
(3) `color`, `contrast`, `brightness`, `sharpness` are randomly enhanced or
    diminished.
(4) Magnitude is randomly sampled from a normal distribution.
(5) Operations are applied with a probability.
"""

import inspect
import math
import tensorflow as tf, tf_keras
import tensorflow_addons.image as tfa_image

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def policy_v0():
  """Autoaugment policy that was used in AutoAugment Paper."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
      [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
      [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
      [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
      [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
      [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
      [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
      [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
      [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
      [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
      [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
      [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
      [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
      [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
      [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
      [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
      [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
      [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
      [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
      [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
      [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
      [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
      [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
      [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
  ]
  return policy


def policy_vtest():
  """Autoaugment test policy for debugging."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
  ]
  return policy


# pylint: disable=g-long-lambda
blend = tf.function(lambda i1, i2, factor: tf.cast(
    tfa_image.blend(tf.cast(i1, tf.float32), tf.cast(i2, tf.float32), factor),
    tf.uint8))
# pylint: enable=g-long-lambda


def random_erase(image,
                 prob,
                 min_area=0.02,
                 max_area=1 / 3,
                 min_aspect=1 / 3,
                 max_aspect=10 / 3,
                 mode='pixel'):
  """The random erasing augmentations: https://arxiv.org/pdf/1708.04896.pdf.

  This augmentation is applied after image normalization.

  Args:
    image: Input image after all other augmentation and normalization. It has
      type tf.float32.
    prob: Probability of applying the random erasing operation.
    min_area: As named.
    max_area: As named.
    min_aspect: As named.
    max_aspect: As named.
    mode: How the erased area is filled. 'pixel' means white noise (uniform
      dist).

  Returns:
    Randomly erased image.
  """

  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  image_area = tf.cast(image_width * image_height, tf.float32)

  # Sample width, height
  erase_area = tf.random.uniform([], min_area, max_area) * image_area
  log_max_target_ar = tf.math.log(
      tf.minimum(
          tf.math.divide(
              tf.math.square(tf.cast(image_width, tf.float32)), erase_area),
          max_aspect))
  log_min_target_ar = tf.math.log(
      tf.maximum(
          tf.math.divide(erase_area,
                         tf.math.square(tf.cast(image_height, tf.float32))),
          min_aspect))
  erase_aspect_ratio = tf.math.exp(
      tf.random.uniform([], log_min_target_ar, log_max_target_ar))
  erase_h = tf.cast(tf.math.sqrt(erase_area / erase_aspect_ratio), tf.int32)
  erase_w = tf.cast(tf.math.sqrt(erase_area * erase_aspect_ratio), tf.int32)

  # Sample (left, top) of the rectangle to erase
  erase_left = tf.random.uniform(
      shape=[], minval=0, maxval=image_width - erase_w, dtype=tf.int32)
  erase_top = tf.random.uniform(
      shape=[], minval=0, maxval=image_height - erase_h, dtype=tf.int32)
  pad_right = image_width - erase_w - erase_left
  pad_bottom = image_height - erase_h - erase_top
  mask = tf.pad(
      tf.zeros([erase_h, erase_w], dtype=image.dtype),
      [[erase_top, pad_bottom], [erase_left, pad_right]],
      constant_values=1)
  mask = tf.expand_dims(mask, -1)  # [H, W, 1]
  if mode == 'pixel':
    fill = tf.random.truncated_normal(
        tf.shape(image), 0.0, 1.0, dtype=image.dtype)
  else:
    fill = tf.zeros(tf.shape(image), dtype=image.dtype)

  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(should_apply_op,
                            lambda: mask * image + (1 - mask) * fill,
                            lambda: image)
  return augmented_image


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize. Smaller `bits` means larger degradation."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by the
      rotate operation.

  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  if isinstance(replace, list) or isinstance(replace, tuple):
    replace = replace[0]
  image = tfa_image.rotate(image, radians, fill_value=replace)
  return image


def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  return tfa_image.translate_xy(image, [-pixels, 0], replace)


def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  return tfa_image.translate_xy(image, [0, -pixels], replace)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                       dtype=tf.float32,
                       shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  with tf.device('/cpu:0'):
    # Some augmentation that uses depth-wise conv will cause crashing when
    # training on GPU. See (b/156242594) for details.
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding='VALID')
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step, 0), lambda: im,
        lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image):
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': tfa_image.shear_x,
    'ShearY': tfa_image.shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': tfa_image.random_cutout,
    'Hue': tf.image.adjust_hue,
}


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: -tensor, lambda: tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level / _MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level):
  level = (level / _MAX_LEVEL) * .9
  level = 1.0 + _randomly_negate_tensor(level)
  return (level,)


def _shear_level_to_arg(level):
  level = (level / _MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = level / _MAX_LEVEL * translate_const
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _posterize_level_to_arg(level):
  return (tf.cast(level / _MAX_LEVEL * 4, tf.uint8),)


def _posterize_increase_level_to_arg(level):
  return (4 - _posterize_level_to_arg(level)[0],)


def _solarize_level_to_arg(level):
  return (tf.cast(level / _MAX_LEVEL * 256, tf.uint8),)


def _solarize_increase_level_to_arg(level):
  return (256 - _solarize_level_to_arg(level)[0],)


def _solarize_add_level_to_arg(level):
  return (tf.cast(level / _MAX_LEVEL * 110, tf.int64),)


def _cutout_arg(level, cutout_size):
  pad_size = tf.cast(level / _MAX_LEVEL * cutout_size, tf.int32)
  return (2 * pad_size, 2 * pad_size)


def level_to_arg(hparams):
  return {
      'AutoContrast':
          lambda level: (),
      'Equalize':
          lambda level: (),
      'Invert':
          lambda level: (),
      'Rotate':
          _rotate_level_to_arg,
      'Posterize':
          _posterize_level_to_arg,
      'PosterizeIncreasing':
          _posterize_increase_level_to_arg,
      'Solarize':
          _solarize_level_to_arg,
      'SolarizeIncreasing':
          _solarize_increase_level_to_arg,
      'SolarizeAdd':
          _solarize_add_level_to_arg,
      'Color':
          _enhance_level_to_arg,
      'ColorIncreasing':
          _enhance_increasing_level_to_arg,
      'Contrast':
          _enhance_level_to_arg,
      'ContrastIncreasing':
          _enhance_increasing_level_to_arg,
      'Brightness':
          _enhance_level_to_arg,
      'BrightnessIncreasing':
          _enhance_increasing_level_to_arg,
      'Sharpness':
          _enhance_level_to_arg,
      'SharpnessIncreasing':
          _enhance_increasing_level_to_arg,
      'ShearX':
          _shear_level_to_arg,
      'ShearY':
          _shear_level_to_arg,
      # pylint:disable=g-long-lambda
      'Cutout':
          lambda level: _cutout_arg(level, hparams['cutout_const']),
      # pylint:disable=g-long-lambda
      'TranslateX':
          lambda level: _translate_level_to_arg(level, hparams['translate_const'
                                                              ]),
      'TranslateY':
          lambda level: _translate_level_to_arg(level, hparams['translate_const'
                                                              ]),
      'Hue':
          lambda level: ((level / _MAX_LEVEL) * 0.25,),
      # pylint:enable=g-long-lambda
  }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(augmentation_hparams)[name](level)

  # Add in replace arg if it is required for the function that is being called.
  # pytype:disable=wrong-arg-types
  if 'replace' in inspect.signature(func).parameters.keys():  # pylint: disable=deprecated-method
    args = tuple(list(args) + [replace_value])
  # pytype:enable=wrong-arg-types

  return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(should_apply_op, lambda: func(image, *args),
                            lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies, image):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


def build_and_apply_nas_policy(policies, image, augmentation_hparams):
  """Build a policy from the given policies passed in and apply to image.

  Args:
    policies: list of lists of tuples in the form `(func, prob, level)`, `func`
      is a string name of the augmentation function, `prob` is the probability
      of applying the `func` operation, `level` is the input argument for
      `func`.
    image: tf.Tensor that the resulting policy will be applied to.
    augmentation_hparams: Hparams associated with the NAS learned policy.

  Returns:
    A version of image that now has data augmentation applied to it based on
    the `policies` pass into the function.
  """
  replace_value = [128, 128, 128]

  # func is the string name of the augmentation function, prob is the
  # probability of applying the operation and level is the parameter associated
  # with the tf op.

  # tf_policies are functions that take in an image and return an augmented
  # image.
  tf_policies = []
  for policy in policies:
    tf_policy = []
    # Link string name to the correct python function and make sure the correct
    # argument is passed into that function.
    for policy_info in policy:
      policy_info = list(policy_info) + [replace_value, augmentation_hparams]

      tf_policy.append(_parse_policy_info(*policy_info))
    # Now build the tf policy that will apply the augmentation procedue
    # on image.
    def make_final_policy(tf_policy_):

      def final_policy(image_):
        for func, prob, args in tf_policy_:
          image_ = _apply_func_with_prob(func, image_, args, prob)
        return image_

      return final_policy

    tf_policies.append(make_final_policy(tf_policy))

  augmented_image = select_and_apply_random_policy(tf_policies, image)
  return augmented_image


def distort_image_with_autoaugment(image, augmentation_name):
  """Applies the AutoAugment policy to `image`.

  AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    augmentation_name: The name of the AutoAugment policy to use. The available
      options are `v0` and `test`. `v0` is the policy used for all of the
      results in the paper and was found to achieve the best results on the COCO
      dataset. `v1`, `v2` and `v3` are additional good policies found on the
      COCO dataset that have slight variation in what operations were used
      during the search procedure along with how many operations are applied in
      parallel to a single image (2 vs 3).

  Returns:
    A tuple containing the augmented versions of `image`.
  """
  available_policies = {'v0': policy_v0, 'test': policy_vtest}
  if augmentation_name not in available_policies:
    raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

  policy = available_policies[augmentation_name]()
  # Hparams that will be used for AutoAugment.
  augmentation_hparams = dict(cutout_const=100, translate_const=250)

  return build_and_apply_nas_policy(policy, image, augmentation_hparams)


# Cutout is implemented separately.
_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'SolarizeAdd',
    'Hue',
]

# Cutout is implemented separately.
_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'Hue',
]

# These augmentations are not suitable for detection task.
_NON_COLOR_DISTORTION_OPS = [
    'Rotate',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
]


def distort_image_with_randaugment(image,
                                   num_layers,
                                   magnitude,
                                   mag_std,
                                   inc,
                                   prob,
                                   color_only=False):
  """Applies the RandAugment policy to `image`.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image. The image
      should have uint8 type in [0, 255].
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range [5,
      30].
    mag_std: Randomness of magnitude. The magnitude will be sampled from a
      normal distribution on the fly.
    inc: Whether to select aug that increases as magnitude increases.
    prob: Probability of any aug being applied.
    color_only: Whether only apply operations that distort color and do not
      change spatial layouts.

  Returns:
    The augmented version of `image`.
  """
  replace_value = [128] * 3
  augmentation_hparams = dict(cutout_const=40, translate_const=100)
  available_ops = _RAND_INCREASING_TRANSFORMS if inc else _RAND_TRANSFORMS
  if color_only:
    available_ops = list(
        filter(lambda op: op not in _NON_COLOR_DISTORTION_OPS, available_ops))

  for layer_num in range(num_layers):
    op_to_select = tf.random.uniform([],
                                     maxval=len(available_ops),
                                     dtype=tf.int32)
    random_magnitude = tf.clip_by_value(
        tf.random.normal([], magnitude, mag_std), 0., _MAX_LEVEL)
    with tf.name_scope('randaug_layer_{}'.format(layer_num)):
      for (i, op_name) in enumerate(available_ops):
        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_hparams)
        image = tf.cond(
            tf.equal(i, op_to_select),
            # pylint:disable=g-long-lambda
            lambda s_func=func, s_args=args: _apply_func_with_prob(
                s_func, image, s_args, prob),
            # pylint:enable=g-long-lambda
            lambda: image)
  return image
