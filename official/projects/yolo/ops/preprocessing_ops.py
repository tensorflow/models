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

"""Preprocessing ops for yolo."""
import random

import numpy as np
import tensorflow as tf, tf_keras

from official.vision.ops import augment
from official.vision.ops import box_ops as bbox_ops

PAD_VALUE = 114
GLOBAL_SEED_SET = False


def set_random_seeds(seed=0):
  """Sets all accessible global seeds to properly apply randomization.

  This is not the same as passing the seed as a variable to each call
  to tf.random.For more, see the documentation for tf.random on the tensorflow
  website https://www.tensorflow.org/api_docs/python/tf/random/set_seed. Note
  that passing the seed to each random number generator will not give you the
  expected behavior if you use more than one generator in a single function.

  Args:
    seed: `Optional[int]` representing the seed you want to use.
  """
  if seed is not None:
    global GLOBAL_SEED_SET
    random.seed(seed)
    GLOBAL_SEED_SET = True
  tf.random.set_seed(seed)
  np.random.seed(seed)


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


def random_scale(val, dtype=tf.float32, seed=None):
  """Generates a random number for scaling a parameter by multiplication.

  Generates a random number for the scale. Half of the time, the value is
  between [1.0, val) with uniformly distributed probability. In the other half,
  the value is the reciprocal of this value. The function is identical to the
  one in the original implementation:
  https://github.com/AlexeyAB/darknet/blob/a3714d0a/src/utils.c#L708-L713

  Args:
    val: A float representing the maximum scaling allowed.
    dtype: The output type of the tensor.
    seed: An `int` used to set the seed.

  Returns:
    The random scale.
  """
  scale = random_uniform_strong(1.0, val, dtype=dtype, seed=seed)
  do_ret = random_uniform_strong(minval=0, maxval=2, dtype=tf.int32, seed=seed)
  if do_ret == 1:
    return scale
  return 1.0 / scale


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  """Pad or clip the tensor value to a fixed length along a given axis.

  Pads a dimension of the tensor to have a maximum number of instances filling
  additional entries with the `pad_value`. Allows for selection of the padding
  axis.

  Args:
    value: An input tensor.
    instances: An `int` representing the maximum number of instances.
    pad_value: An `int` representing the value used for padding until the
      maximum number of instances is obtained.
    pad_axis: An `int` representing the axis index to pad.

  Returns:
    The output tensor whose dimensions match the input tensor except with the
    size along the `pad_axis` replaced by `instances`.
  """

  # get the real shape of value
  shape = tf.shape(value)

  # compute the padding axis
  if pad_axis < 0:
    pad_axis = tf.rank(value) + pad_axis

  # determin how much of the tensor value to keep
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(value, [take, -1], axis=pad_axis)

  # pad the clipped tensor to the right shape
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)

  if isinstance(instances, int):
    vshape = value.get_shape().as_list()
    vshape[pad_axis] = instances
    value.set_shape(vshape)
  return value


def get_image_shape(image):
  """Consistently gets the width and height of the image.

  Gets the shape of the image regardless of if the image is in the
  (batch_size, x, y, c) format or the (x, y, c) format.

  Args:
    image: A tensor who has either 3 or 4 dimensions.

  Returns:
    A tuple (height, width), where height is the height of the image
    and width is the width of the image.
  """
  shape = tf.shape(image)
  if shape.get_shape().as_list()[0] == 4:
    width = shape[2]
    height = shape[1]
  else:
    width = shape[1]
    height = shape[0]
  return height, width


def _augment_hsv_darknet(image, rh, rs, rv, seed=None):
  """Randomize the hue, saturation, and brightness via the darknet method."""
  if rh > 0.0:
    deltah = random_uniform_strong(-rh, rh, seed=seed)
    image = tf.image.adjust_hue(image, deltah)
  if rs > 0.0:
    deltas = random_scale(rs, seed=seed)
    image = tf.image.adjust_saturation(image, deltas)
  if rv > 0.0:
    deltav = random_scale(rv, seed=seed)
    image *= tf.cast(deltav, image.dtype)

  # clip the values of the image between 0.0 and 1.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def _augment_hsv_torch(image, rh, rs, rv, seed=None):
  """Randomize the hue, saturation, and brightness via the pytorch method."""
  dtype = image.dtype
  image = tf.cast(image, tf.float32)
  image = tf.image.rgb_to_hsv(image)
  gen_range = tf.cast([rh, rs, rv], image.dtype)
  scale = tf.cast([180, 255, 255], image.dtype)
  r = random_uniform_strong(
      -1, 1, shape=[3], dtype=image.dtype, seed=seed) * gen_range + 1

  image = tf.math.floor(tf.cast(image, scale.dtype) * scale)
  image = tf.math.floor(tf.cast(image, r.dtype) * r)
  h, s, v = tf.split(image, 3, axis=-1)
  h = h % 180
  s = tf.clip_by_value(s, 0, 255)
  v = tf.clip_by_value(v, 0, 255)

  image = tf.concat([h, s, v], axis=-1)
  image = tf.cast(image, scale.dtype) / scale
  image = tf.image.hsv_to_rgb(image)
  return tf.cast(image, dtype)


def image_rand_hsv(image, rh, rs, rv, seed=None, darknet=False):
  """Randomly alters the hue, saturation, and brightness of an image.

  Args:
    image: `Tensor` of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be multiplied to
      the hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to
      the saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to
      the brightness.
    seed: `Optional[int]` for the seed to use in the random number generation.
    darknet: `bool` indicating whether the model was originally built in the
      Darknet or PyTorch library.

  Returns:
    The HSV altered image in the same datatype as the input image.
  """

  if darknet:
    image = _augment_hsv_darknet(image, rh, rs, rv, seed=seed)
  else:
    image = _augment_hsv_torch(image, rh, rs, rv, seed=seed)
  return image


def mosaic_cut(image, original_width, original_height, width, height, center,
               ptop, pleft, pbottom, pright, shiftx, shifty):
  """Generates a random center location to use for the mosaic operation.

  Given a center location, cuts the input image into a slice that will be
  concatenated with other slices with the same center in order to construct
  a final mosaicked image.

  Args:
    image: `Tensor` of shape [None, None, 3] that needs to be altered.
    original_width: `float` value indicating the original width of the image.
    original_height: `float` value indicating the original height of the image.
    width: `float` value indicating the final width of the image.
    height: `float` value indicating the final height of the image.
    center: `float` value indicating the desired center of the final patched
      image.
    ptop: `float` value indicating the top of the image without padding.
    pleft: `float` value indicating the left of the image without padding.
    pbottom: `float` value indicating the bottom of the image without padding.
    pright: `float` value indicating the right of the image without padding.
    shiftx: `float` 0.0 or 1.0 value indicating if the image is on the left or
      right.
    shifty: `float` 0.0 or 1.0 value indicating if the image is at the top or
      bottom.

  Returns:
    image: The cropped image in the same datatype as the input image.
    crop_info: `float` tensor that is applied to the boxes in order to select
      the boxes still contained within the image.
  """

  def cast(values, dtype):
    return [tf.cast(value, dtype) for value in values]

  with tf.name_scope('mosaic_cut'):
    center = tf.cast(center, width.dtype)
    zero = tf.cast(0.0, width.dtype)
    cut_x, cut_y = center[1], center[0]

    # Select the crop of the image to use
    left_shift = tf.minimum(
        tf.minimum(cut_x, tf.maximum(zero, -pleft * width / original_width)),
        width - cut_x)
    top_shift = tf.minimum(
        tf.minimum(cut_y, tf.maximum(zero, -ptop * height / original_height)),
        height - cut_y)
    right_shift = tf.minimum(
        tf.minimum(width - cut_x,
                   tf.maximum(zero, -pright * width / original_width)), cut_x)
    bot_shift = tf.minimum(
        tf.minimum(height - cut_y,
                   tf.maximum(zero, -pbottom * height / original_height)),
        cut_y)

    (left_shift, top_shift, right_shift, bot_shift,
     zero) = cast([left_shift, top_shift, right_shift, bot_shift, zero],
                  tf.float32)
    # Build a crop offset and a crop size tensor to use for slicing.
    crop_offset = [zero, zero, zero]
    crop_size = [zero - 1, zero - 1, zero - 1]
    if shiftx == 0.0 and shifty == 0.0:
      crop_offset = [top_shift, left_shift, zero]
      crop_size = [cut_y, cut_x, zero - 1]
    elif shiftx == 1.0 and shifty == 0.0:
      crop_offset = [top_shift, cut_x - right_shift, zero]
      crop_size = [cut_y, width - cut_x, zero - 1]
    elif shiftx == 0.0 and shifty == 1.0:
      crop_offset = [cut_y - bot_shift, left_shift, zero]
      crop_size = [height - cut_y, cut_x, zero - 1]
    elif shiftx == 1.0 and shifty == 1.0:
      crop_offset = [cut_y - bot_shift, cut_x - right_shift, zero]
      crop_size = [height - cut_y, width - cut_x, zero - 1]

    # Contain and crop the image.
    ishape = tf.cast(tf.shape(image)[:2], crop_size[0].dtype)
    crop_size[0] = tf.minimum(crop_size[0], ishape[0])
    crop_size[1] = tf.minimum(crop_size[1], ishape[1])

    crop_offset = tf.cast(crop_offset, tf.int32)
    crop_size = tf.cast(crop_size, tf.int32)

    image = tf.slice(image, crop_offset, crop_size)
    crop_info = tf.stack([
        tf.cast(ishape, tf.float32),
        tf.cast(tf.shape(image)[:2], dtype=tf.float32),
        tf.ones_like(ishape, dtype=tf.float32),
        tf.cast(crop_offset[:2], tf.float32)
    ])

  return image, crop_info


def resize_and_jitter_image(image,
                            desired_size,
                            jitter=0.0,
                            letter_box=None,
                            random_pad=True,
                            crop_only=False,
                            shiftx=0.5,
                            shifty=0.5,
                            cut=None,
                            method=tf.image.ResizeMethod.BILINEAR,
                            seed=None):
  """Resize, Pad, and distort a given input image.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    jitter: an `int` representing the maximum jittering that can be applied to
      the image.
    letter_box: a `bool` representing if letterboxing should be applied.
    random_pad: a `bool` representing if random padding should be applied.
    crop_only: a `bool` representing if only cropping will be applied.
    shiftx: a `float` indicating if the image is in the left or right.
    shifty: a `float` value indicating if the image is in the top or bottom.
    cut: a `float` value indicating the desired center of the final patched
      image.
    method: function to resize input image to scaled image.
    seed: seed for random scale jittering.

  Returns:
    image_: a `Tensor` of shape [height, width, 3] where [height, width]
      equals to `desired_size`.
    infos: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
    cast([original_width, original_height, width, height, ptop, pleft, pbottom,
      pright], tf.float32): a `Tensor` containing the information of the image
        andthe applied preprocessing.
  """

  def intersection(a, b):
    """Finds the intersection between 2 crops."""
    minx = tf.maximum(a[0], b[0])
    miny = tf.maximum(a[1], b[1])
    maxx = tf.minimum(a[2], b[2])
    maxy = tf.minimum(a[3], b[3])
    return tf.convert_to_tensor([minx, miny, maxx, maxy])

  def cast(values, dtype):
    return [tf.cast(value, dtype) for value in values]

  if jitter > 0.5 or jitter < 0:
    raise ValueError('maximum change in aspect ratio must be between 0 and 0.5')

  with tf.name_scope('resize_and_jitter_image'):
    # Cast all parameters to a usable float data type.
    jitter = tf.cast(jitter, tf.float32)
    original_dtype, original_dims = image.dtype, tf.shape(image)[:2]

    # original width, original height, desigered width, desired height
    original_width, original_height, width, height = cast(
        [original_dims[1], original_dims[0], desired_size[1], desired_size[0]],
        tf.float32)

    # Compute the random delta width and height etc. and randomize the
    # location of the corner points.
    jitter_width = original_width * jitter
    jitter_height = original_height * jitter
    pleft = random_uniform_strong(
        -jitter_width, jitter_width, jitter_width.dtype, seed=seed)
    pright = random_uniform_strong(
        -jitter_width, jitter_width, jitter_width.dtype, seed=seed)
    ptop = random_uniform_strong(
        -jitter_height, jitter_height, jitter_height.dtype, seed=seed)
    pbottom = random_uniform_strong(
        -jitter_height, jitter_height, jitter_height.dtype, seed=seed)

    # Letter box the image.
    if letter_box:
      (image_aspect_ratio,
       input_aspect_ratio) = original_width / original_height, width / height
      distorted_aspect = image_aspect_ratio / input_aspect_ratio

      delta_h, delta_w = 0.0, 0.0
      pullin_h, pullin_w = 0.0, 0.0
      if distorted_aspect > 1:
        delta_h = ((original_width / input_aspect_ratio) - original_height) / 2
      else:
        delta_w = ((original_height * input_aspect_ratio) - original_width) / 2

      ptop = ptop - delta_h - pullin_h
      pbottom = pbottom - delta_h - pullin_h
      pright = pright - delta_w - pullin_w
      pleft = pleft - delta_w - pullin_w

    # Compute the width and height to crop or pad too, and clip all crops to
    # to be contained within the image.
    swidth = original_width - pleft - pright
    sheight = original_height - ptop - pbottom
    src_crop = intersection([ptop, pleft, sheight + ptop, swidth + pleft],
                            [0, 0, original_height, original_width])

    # Random padding used for mosaic.
    h_ = src_crop[2] - src_crop[0]
    w_ = src_crop[3] - src_crop[1]
    if random_pad:
      rmh = tf.maximum(0.0, -ptop)
      rmw = tf.maximum(0.0, -pleft)
    else:
      rmw = (swidth - w_) * shiftx
      rmh = (sheight - h_) * shifty

    # Cast cropping params to usable dtype.
    src_crop = tf.cast(src_crop, tf.int32)

    # Compute padding parmeters.
    dst_shape = [rmh, rmw, rmh + h_, rmw + w_]
    ptop, pleft, pbottom, pright = dst_shape
    pad = dst_shape * tf.cast([1, 1, -1, -1], ptop.dtype)
    pad += tf.cast([0, 0, sheight, swidth], ptop.dtype)
    pad = tf.cast(pad, tf.int32)

    infos = []

    # Crop the image to desired size.
    cropped_image = tf.slice(
        image, [src_crop[0], src_crop[1], 0],
        [src_crop[2] - src_crop[0], src_crop[3] - src_crop[1], -1])
    crop_info = tf.stack([
        tf.cast(original_dims, tf.float32),
        tf.cast(tf.shape(cropped_image)[:2], dtype=tf.float32),
        tf.ones_like(original_dims, dtype=tf.float32),
        tf.cast(src_crop[:2], tf.float32)
    ])
    infos.append(crop_info)

    if crop_only:
      if not letter_box:
        h_, w_ = cast(get_image_shape(cropped_image), width.dtype)
        width = tf.cast(tf.round((w_ * width) / swidth), tf.int32)
        height = tf.cast(tf.round((h_ * height) / sheight), tf.int32)
        cropped_image = tf.image.resize(
            cropped_image, [height, width], method=method)
        cropped_image = tf.cast(cropped_image, original_dtype)
      return cropped_image, infos, cast([
          original_width, original_height, width, height, ptop, pleft, pbottom,
          pright
      ], tf.int32)

    # Pad the image to desired size.
    image_ = tf.pad(
        cropped_image, [[pad[0], pad[2]], [pad[1], pad[3]], [0, 0]],
        constant_values=PAD_VALUE)

    # Pad and scale info
    isize = tf.cast(tf.shape(image_)[:2], dtype=tf.float32)
    osize = tf.cast((desired_size[0], desired_size[1]), dtype=tf.float32)
    pad_info = tf.stack([
        tf.cast(tf.shape(cropped_image)[:2], tf.float32),
        osize,
        osize/isize,
        (-tf.cast(pad[:2], tf.float32)*osize/isize)
    ])
    infos.append(pad_info)

    temp = tf.shape(image_)[:2]
    cond = temp > tf.cast(desired_size, temp.dtype)
    if tf.reduce_any(cond):
      size = tf.cast(desired_size, temp.dtype)
      size = tf.where(cond, size, temp)
      image_ = tf.image.resize(
          image_, (size[0], size[1]), method=tf.image.ResizeMethod.AREA)
      image_ = tf.cast(image_, original_dtype)

    image_ = tf.image.resize(
        image_, (desired_size[0], desired_size[1]),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False)

    image_ = tf.cast(image_, original_dtype)
    if cut is not None:
      image_, crop_info = mosaic_cut(image_, original_width, original_height,
                                     width, height, cut, ptop, pleft, pbottom,
                                     pright, shiftx, shifty)
      infos.append(crop_info)
    return image_, infos, cast([
        original_width, original_height, width, height, ptop, pleft, pbottom,
        pright
    ], tf.float32)


def _build_transform(image,
                     perspective=0.00,
                     degrees=0.0,
                     scale_min=1.0,
                     scale_max=1.0,
                     translate=0.0,
                     random_pad=False,
                     desired_size=None,
                     seed=None):
  """Builds a unified affine transformation to spatially augment the image."""

  height, width = get_image_shape(image)
  ch = height = tf.cast(height, tf.float32)
  cw = width = tf.cast(width, tf.float32)
  deg_to_rad = lambda x: tf.cast(x, tf.float32) * np.pi / 180.0

  if desired_size is not None:
    desired_size = tf.cast(desired_size, tf.float32)
    ch = desired_size[0]
    cw = desired_size[1]

  # Compute the center of the image in the output resulution.
  center = tf.eye(3, dtype=tf.float32)
  center = tf.tensor_scatter_nd_update(center, [[0, 2], [1, 2]],
                                       [-cw / 2, -ch / 2])
  center_boxes = tf.tensor_scatter_nd_update(center, [[0, 2], [1, 2]],
                                             [cw / 2, ch / 2])

  # Compute a random rotation to apply.
  rotation = tf.eye(3, dtype=tf.float32)
  a = deg_to_rad(random_uniform_strong(-degrees, degrees, seed=seed))
  cos = tf.math.cos(a)
  sin = tf.math.sin(a)
  rotation = tf.tensor_scatter_nd_update(rotation,
                                         [[0, 0], [0, 1], [1, 0], [1, 1]],
                                         [cos, -sin, sin, cos])
  rotation_boxes = tf.tensor_scatter_nd_update(rotation,
                                               [[0, 0], [0, 1], [1, 0], [1, 1]],
                                               [cos, sin, -sin, cos])

  # Compute a random prespective change to apply.
  prespective_warp = tf.eye(3)
  px = random_uniform_strong(-perspective, perspective, seed=seed)
  py = random_uniform_strong(-perspective, perspective, seed=seed)
  prespective_warp = tf.tensor_scatter_nd_update(prespective_warp,
                                                 [[2, 0], [2, 1]], [px, py])
  prespective_warp_boxes = tf.tensor_scatter_nd_update(prespective_warp,
                                                       [[2, 0], [2, 1]],
                                                       [-px, -py])

  # Compute a random scaling to apply.
  scale = tf.eye(3, dtype=tf.float32)
  s = random_uniform_strong(scale_min, scale_max, seed=seed)
  scale = tf.tensor_scatter_nd_update(scale, [[0, 0], [1, 1]], [1 / s, 1 / s])
  scale_boxes = tf.tensor_scatter_nd_update(scale, [[0, 0], [1, 1]], [s, s])

  # Compute a random Translation to apply.
  translation = tf.eye(3)
  if (random_pad and height * s < ch and width * s < cw):
    # The image is contained within the image and arbitrarily translated to
    # locations with in the image.
    center = center_boxes = tf.eye(3, dtype=tf.float32)
    tx = random_uniform_strong(-1, 0, seed=seed) * (cw / s - width)
    ty = random_uniform_strong(-1, 0, seed=seed) * (ch / s - height)
  else:
    # The image can be translated outside of the output resolution window
    # but the image is translated relative to the output resolution not the
    # input image resolution.
    tx = random_uniform_strong(0.5 - translate, 0.5 + translate, seed=seed)
    ty = random_uniform_strong(0.5 - translate, 0.5 + translate, seed=seed)

    # Center and Scale the image such that the window of translation is
    # contained to the output resolution.
    dx, dy = (width - cw / s) / width, (height - ch / s) / height
    sx, sy = 1 - dx, 1 - dy
    bx, by = dx / 2, dy / 2
    tx, ty = bx + (sx * tx), by + (sy * ty)

    # Scale the translation to width and height of the image.
    tx *= width
    ty *= height

  translation = tf.tensor_scatter_nd_update(translation, [[0, 2], [1, 2]],
                                            [tx, ty])
  translation_boxes = tf.tensor_scatter_nd_update(translation, [[0, 2], [1, 2]],
                                                  [-tx, -ty])

  # Use repeated matric multiplications to combine all the image transforamtions
  # into a single unified augmentation operation M is applied to the image
  # Mb is to apply to the boxes. The order of matrix multiplication is
  # important. First, Translate, then Scale, then Rotate, then Center, then
  # finally alter the Prepsective.
  affine = (translation @ scale @ rotation @ center @ prespective_warp)
  affine_boxes = (
      prespective_warp_boxes @ center_boxes @ rotation_boxes @ scale_boxes
      @ translation_boxes)
  return affine, affine_boxes, s


def affine_warp_image(image,
                      desired_size,
                      perspective=0.00,
                      degrees=0.0,
                      scale_min=1.0,
                      scale_max=1.0,
                      translate=0.0,
                      random_pad=False,
                      seed=None):
  """Applies random spatial augmentation to the image.

  Args:
    image: A `Tensor` for the image.
    desired_size: A `tuple` for desired output image size.
    perspective: An `int` for the maximum that can be applied to random
      perspective change.
    degrees: An `int` for the maximum degrees that can be applied to random
      rotation.
    scale_min: An `int` for the minimum scaling factor that can be applied to
      random scaling.
    scale_max: An `int` for the maximum scaling factor that can be applied to
      random scaling.
    translate: An `int` for the maximum translation that can be applied to
      random translation.
    random_pad: A `bool` for using random padding.
    seed: An `Optional[int]` for the seed to use in random number generation.

  Returns:
    image: A `Tensor` representing the augmented image.
    affine_matrix: A `Tensor` representing the augmenting matrix for the image.
    affine_info: A `List` containing the size of the original image, the desired
      output_size of the image and the augmenting matrix for the boxes.
  """

  # Build an image transformation matrix.
  image_size = tf.cast(get_image_shape(image), tf.float32)
  affine_matrix, affine_boxes, _ = _build_transform(
      image,
      perspective=perspective,
      degrees=degrees,
      scale_min=scale_min,
      scale_max=scale_max,
      translate=translate,
      random_pad=random_pad,
      desired_size=desired_size,
      seed=seed)
  affine = tf.reshape(affine_matrix, [-1])
  affine = tf.cast(affine[:-1], tf.float32)

  # Apply the transformation to image.
  image = augment.transform(
      image,
      affine,
      fill_value=PAD_VALUE,
      output_shape=desired_size,
      interpolation='bilinear',
      fill_mode='constant',
  )

  desired_size = tf.cast(desired_size, tf.float32)
  affine_info = [image_size, desired_size, affine_boxes]
  return image, affine_matrix, affine_info


def affine_warp_boxes(affine, boxes, output_size, box_history):
  """Applies random rotation, random perspective change and random translation.

  and random scaling to the boxes.

  Args:
    affine: A `Tensor` for the augmenting matrix for the boxes.
    boxes: A `Tensor` for the boxes.
    output_size: A `list` of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
    box_history: A `Tensor` for the boxes history, which are the boxes that
      undergo the same augmentations as `boxes`, but no clipping was applied. We
      can keep track of how much changes are done to the boxes by keeping track
      of this tensor.

  Returns:
    clipped_boxes: A `Tensor` representing the augmented boxes.
    box_history: A `Tensor` representing the augmented box_history.
  """

  def _get_corners(box):
    """Get the corner of each box as a tuple of (x, y) coordinates."""
    ymi, xmi, yma, xma = tf.split(box, 4, axis=-1)
    tl = tf.concat([xmi, ymi], axis=-1)
    bl = tf.concat([xmi, yma], axis=-1)
    tr = tf.concat([xma, ymi], axis=-1)
    br = tf.concat([xma, yma], axis=-1)
    return tf.concat([tl, bl, tr, br], axis=-1)

  def _corners_to_boxes(corner):
    """Convert (x, y) corners back into boxes [ymin, xmin, ymax, xmax]."""
    corner = tf.reshape(corner, [-1, 4, 2])
    y = corner[..., 1]
    x = corner[..., 0]
    y_min = tf.reduce_min(y, axis=-1)
    x_min = tf.reduce_min(x, axis=-1)
    y_max = tf.reduce_max(y, axis=-1)
    x_max = tf.reduce_max(x, axis=-1)
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)

  def _aug_boxes(affine_matrix, box):
    """Apply an affine transformation matrix M to the boxes augment boxes."""
    corners = _get_corners(box)
    corners = tf.reshape(corners, [-1, 4, 2])
    z = tf.expand_dims(tf.ones_like(corners[..., 1]), axis=-1)
    corners = tf.concat([corners, z], axis=-1)

    corners = tf.transpose(
        tf.matmul(affine_matrix, corners, transpose_b=True), perm=(0, 2, 1))

    corners, p = tf.split(corners, [2, 1], axis=-1)
    corners /= p
    corners = tf.reshape(corners, [-1, 8])
    box = _corners_to_boxes(corners)
    return box

  boxes = _aug_boxes(affine, boxes)
  box_history = _aug_boxes(affine, box_history)

  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, box_history


def boxes_candidates(clipped_boxes,
                     box_history,
                     wh_thr=2,
                     ar_thr=20,
                     area_thr=0.1):
  """Filters the boxes that don't satisfy the width/height and area constraints.

  Args:
    clipped_boxes: A `Tensor` for the boxes.
    box_history: A `Tensor` for the boxes history, which are the boxes that
      undergo the same augmentations as `boxes`, but no clipping was applied. We
      can keep track of how much changes are done to the boxes by keeping track
      of this tensor.
    wh_thr: An `int` for the width/height threshold.
    ar_thr: An `int` for the aspect ratio threshold.
    area_thr: An `int` for the area threshold.

  Returns:
    indices[:, 0]: A `Tensor` representing valid boxes after filtering.
  """
  if area_thr == 0.0:
    wh_thr = 0
    ar_thr = np.inf
  area_thr = tf.math.abs(area_thr)

  # Get the scaled and shifted heights of the original
  # unclipped boxes.
  og_height = tf.maximum(box_history[:, 2] - box_history[:, 0], 0.0)
  og_width = tf.maximum(box_history[:, 3] - box_history[:, 1], 0.0)

  # Get the scaled and shifted heights of the clipped boxes.
  clipped_height = tf.maximum(clipped_boxes[:, 2] - clipped_boxes[:, 0], 0.0)
  clipped_width = tf.maximum(clipped_boxes[:, 3] - clipped_boxes[:, 1], 0.0)

  # Determine the aspect ratio of the clipped boxes.
  ar = tf.maximum(clipped_width / (clipped_height + 1e-16),
                  clipped_height / (clipped_width + 1e-16))

  # Ensure the clipped width adn height are larger than a preset threshold.
  conda = clipped_width >= wh_thr
  condb = clipped_height >= wh_thr

  # Ensure the area of the clipped box is larger than the area threshold.
  area = (clipped_height * clipped_width) / (og_width * og_height + 1e-16)
  condc = area > area_thr

  # Ensure the aspect ratio is not too extreme.
  condd = ar < ar_thr

  cond = tf.expand_dims(
      tf.logical_and(
          tf.logical_and(conda, condb), tf.logical_and(condc, condd)),
      axis=-1)

  # Set all the boxes that fail the test to be equal to zero.
  indices = tf.where(cond)
  return indices[:, 0]


def resize_and_crop_boxes(boxes, image_scale, output_size, offset, box_history):
  """Resizes and crops the boxes.

  Args:
    boxes: A `Tensor` for the boxes.
    image_scale: A `Tensor` for the scaling factor of the image.
    output_size: A `list` of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
    offset: A `Tensor` for how much translation was applied to the image.
    box_history: A `Tensor` for the boxes history, which are the boxes that
      undergo the same augmentations as `boxes`, but no clipping was applied. We
      can keep track of how much changes are done to the boxes by keeping track
      of this tensor.

  Returns:
    clipped_boxes: A `Tensor` representing the augmented boxes.
    box_history: A `Tensor` representing the augmented box_history.
  """

  # Shift and scale the input boxes.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Check the hitory of the boxes.
  box_history *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  box_history -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Clip the shifted and scaled boxes.
  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, box_history


def transform_and_clip_boxes(boxes,
                             infos,
                             affine=None,
                             shuffle_boxes=False,
                             area_thresh=0.1,
                             seed=None,
                             filter_and_clip_boxes=True):
  """Clips and cleans the boxes.

  Args:
    boxes: A `Tensor` for the boxes.
    infos: A `list` that contains the image infos.
    affine: A `list` that contains parameters for resize and crop.
    shuffle_boxes: A `bool` for shuffling the boxes.
    area_thresh: An `int` for the area threshold.
    seed: seed for random number generation.
    filter_and_clip_boxes: A `bool` for filtering and clipping the boxes to
      [0, 1].

  Returns:
    boxes: A `Tensor` representing the augmented boxes.
    ind: A `Tensor` valid box indices.
  """

  # Clip and clean boxes.
  def get_valid_boxes(boxes):
    """Get indices for non-empty boxes."""
    # Convert the boxes to center width height formatting.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    base = tf.logical_and(tf.greater(height, 0), tf.greater(width, 0))
    return base

  # Initialize history to track operation applied to boxes
  box_history = boxes

  # Make sure all boxes are valid to start, clip to [0, 1] and get only the
  # valid boxes.
  output_size = None
  if filter_and_clip_boxes:
    boxes = tf.math.maximum(tf.math.minimum(boxes, 1.0), 0.0)
  cond = get_valid_boxes(boxes)

  if infos is None:
    infos = []

  for info in infos:
    # Denormalize the boxes.
    boxes = bbox_ops.denormalize_boxes(boxes, info[0])
    box_history = bbox_ops.denormalize_boxes(box_history, info[0])

    # Shift and scale all boxes, and keep track of box history with no
    # box clipping, history is used for removing boxes that have become
    # too small or exit the image area.
    (boxes, box_history) = resize_and_crop_boxes(
        boxes, info[2, :], info[1, :], info[3, :], box_history=box_history)

    # Get all the boxes that still remain in the image and store
    # in a bit vector for later use.
    cond = tf.logical_and(get_valid_boxes(boxes), cond)

    # Normalize the boxes to [0, 1].
    output_size = info[1]
    boxes = bbox_ops.normalize_boxes(boxes, output_size)
    box_history = bbox_ops.normalize_boxes(box_history, output_size)

  if affine is not None:
    # Denormalize the boxes.
    boxes = bbox_ops.denormalize_boxes(boxes, affine[0])
    box_history = bbox_ops.denormalize_boxes(box_history, affine[0])

    # Clipped final boxes.
    (boxes, box_history) = affine_warp_boxes(
        affine[2], boxes, affine[1], box_history=box_history)

    # Get all the boxes that still remain in the image and store
    # in a bit vector for later use.
    cond = tf.logical_and(get_valid_boxes(boxes), cond)

    # Normalize the boxes to [0, 1].
    output_size = affine[1]
    boxes = bbox_ops.normalize_boxes(boxes, output_size)
    box_history = bbox_ops.normalize_boxes(box_history, output_size)

  # Remove the bad boxes.
  boxes *= tf.cast(tf.expand_dims(cond, axis=-1), boxes.dtype)

  # Threshold the existing boxes.
  if filter_and_clip_boxes:
    if output_size is not None:
      boxes_ = bbox_ops.denormalize_boxes(boxes, output_size)
      box_history_ = bbox_ops.denormalize_boxes(box_history, output_size)
      inds = boxes_candidates(boxes_, box_history_, area_thr=area_thresh)
    else:
      inds = boxes_candidates(
          boxes, box_history, wh_thr=0.0, area_thr=area_thresh)
    # Select and gather the good boxes.
    if shuffle_boxes:
      inds = tf.random.shuffle(inds, seed=seed)
  else:
    inds = bbox_ops.get_non_empty_box_indices(boxes)
  boxes = tf.gather(boxes, inds)
  return boxes, inds
