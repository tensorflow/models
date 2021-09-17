# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Preproceesing operations for YOLO."""
import tensorflow as tf
import numpy as np
import random
import os

import tensorflow_addons as tfa
from official.vision.beta.projects.yolo.ops import box_ops
from official.vision.beta.projects.yolo.ops import loss_utils
from official.vision.beta.ops import box_ops as bbox_ops

PAD_VALUE = 114
GLOBAL_SEED_SET = False

def set_random_seeds(seed=0):
  """Sets all accessible global seeds to properly apply randomization.

  This is not the same as passing seed as a variable to each call to tf.random.
  For more, see the documentation for tf.random on the tensorflow website 
  https://www.tensorflow.org/api_docs/python/tf/random/set_seed. Note that 
  passing seed to each random number generator will not giv you the expected 
  behavior IF you use more than one generator in a single function. 

  Args: 
    seed: `Optional[int]` representing the seed you want to use.
  """
  if seed is not None:
    global GLOBAL_SEED_SET
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    GLOBAL_SEED_SET = True
  tf.random.set_seed(seed)
  np.random.seed(seed)


def get_pad_value():
  return PAD_VALUE


def rand_uniform_strong(minval, maxval, dtype=tf.float32, seed=None, shape=[]):
  """A unified fucntion for consistant random number generation. 

  Equivalent to tf.random.uniform, except that minval and maxval are flipped if
  minval is greater than maxval. Seed Safe random number generator.

  Args:
    minval: An `int` for a lower or upper endpoint of the interval from which to
      choose the random number.
    maxval: An `int` for the other endpoint.
    dtype: The output type of the tensor.
  
  Returns:
    A random tensor of type dtype that falls between minval and maxval excluding
    the bigger one.
  """
  if GLOBAL_SEED_SET:
    seed = None

  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform(
      shape=shape, minval=minval, maxval=maxval, seed=seed, dtype=dtype)


def rand_scale(val, dtype=tf.float32, seed=None):
  """Generate a random number for scaling a parameter by multiplication.

  Generates a random number for the scale. Half the time, the value is between
  [1.0, val) with uniformly distributed probability. The other half, the value
  is the reciprocal of this value.
  
  The function is identical to the one in the original implementation:
  https://github.com/AlexeyAB/darknet/blob/a3714d0a/src/utils.c#L708-L713
  
  Args:
    val: A float representing the maximum scaling allowed.
    dtype: The output type of the tensor.
  Returns:
    The random scale.
  """
  scale = rand_uniform_strong(1.0, val, dtype=dtype, seed=seed)
  do_ret = rand_uniform_strong(minval=0, maxval=2, dtype=tf.int32, seed=seed)
  if (do_ret == 1):
    return scale
  return 1.0 / scale


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  """Pad pr clip the tensor value to a fixed length along a given axis.

  Pad a dimension of the tensor to have a maximum number of instances filling
  additional entries with the `pad_value`. Allows for selection of the padding 
  axis
   
  Args:
    value: An input tensor.
    instances: An int representing the maximum number of instances.
    pad_value: An int representing the value used for padding until the maximum
      number of instances is obtained.
    pad_axis: An int representing the axis index to pad.
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
  return value


def get_image_shape(image):
  """ Consitently get the width and height of the image. 
  
  Get the shape of the image regardless of if the image is in the
  (batch_size, x, y, c) format or the (x, y, c) format.
  
  Args:
    image: A tensor who has either 3 or 4 dimensions.
  
  Returns:
    A tuple representing the (height, width) of the image.
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
  """Randomly alter the hue, saturation, and brightness of an image. 

  Applies ranomdization the same way as Darknet by scaling the saturation and 
  brightness of the image and adding/rotating the hue.

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be added to hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  if rh > 0.0:
    delta = rand_uniform_strong(-rh, rh, seed=seed)
    image = tf.image.adjust_hue(image, delta)
  if rs > 0.0:
    delta = rand_scale(rs, seed=seed)
    image = tf.image.adjust_saturation(image, delta)
  if rv > 0.0:
    delta = rand_scale(rv, seed=seed)
    image *= delta

  # clip the values of the image between 0.0 and 1.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def _augment_hsv_torch(image, rh, rs, rv, seed=None):
  """Randomly alter the hue, saturation, and brightness of an image. 

  Applies ranomdization the same way as Darknet by scaling the saturation and 
  brightness and hue of the image.

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be  multiplied to 
      hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  dtype = image.dtype
  image = tf.cast(image, tf.float32)
  image = tf.image.rgb_to_hsv(image)
  gen_range = tf.cast([rh, rs, rv], image.dtype)
  scale = tf.cast([180, 255, 255], image.dtype)
  r = rand_uniform_strong(
      -1, 1, shape=[3], dtype=image.dtype, seed=seed) * gen_range + 1

  # image = tf.cast(tf.cast(image, r.dtype) * (r * scale), tf.int32)
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
  """Randomly alter the hue, saturation, and brightness of an image. 

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be  multiplied to 
      hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
    darknet: `bool` indicating wether the model was orignally built in the 
      darknet or the pytorch library.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  if darknet:
    image = _augment_hsv_darknet(image, rh, rs, rv, seed=seed)
  else:
    image = _augment_hsv_torch(image, rh, rs, rv, seed=seed)
  return image


def mosaic_cut(image, original_width, original_height, width, height, center,
               ptop, pleft, pbottom, pright, shiftx, shifty):
  """Use a provided center to take slices of 4 images to apply mosaic. 
  
  Given a center location, cut the input image into a slice that will be 
  concatnated with other slices with the same center in order to construct 
  a final mosaiced image. 
  
  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    original_width: `float` value indicating the orignal width of the image.
    original_height: `float` value indicating the orignal height of the image.
    width: `float` value indicating the final width image.
    height: `float` value indicating the final height image.
    center: `float` value indicating the desired center of the final patched 
      image.
    ptop: `float` value indicating the top of the image without padding.
    pleft: `float` value indicating the left of the image without padding. 
    pbottom: `float` value indicating the bottom of the image without padding. 
    pright: `float` value indicating the right of the image without padding. 
    shiftx: `float` 0.0 or 1.0 value indicating if the image is in the 
      left or right.
    shifty: `float` 0.0 or 1.0 value indicating if the image is in the 
      top or bottom.
  
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
  """Resize, Pad, and distort a given input image following Darknet.
  
  """

  def intersection(a, b):
    minx = tf.maximum(a[0], b[0])
    miny = tf.maximum(a[1], b[1])
    maxx = tf.minimum(a[2], b[2])
    maxy = tf.minimum(a[3], b[3])
    return tf.convert_to_tensor([minx, miny, maxx, maxy])

  def cast(values, dtype):
    return [tf.cast(value, dtype) for value in values]

  if jitter > 0.5 or jitter < 0:
    raise Exception("maximum change in aspect ratio must be between 0 and 0.5")

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
    pleft = rand_uniform_strong(
        -jitter_width, jitter_width, jitter_width.dtype, seed=seed)
    pright = rand_uniform_strong(
        -jitter_width, jitter_width, jitter_width.dtype, seed=seed)
    ptop = rand_uniform_strong(
        -jitter_height, jitter_height, jitter_height.dtype, seed=seed)
    pbottom = rand_uniform_strong(
        -jitter_height, jitter_height, jitter_height.dtype, seed=seed)

    # Letter box the image.
    if letter_box == True or letter_box is None:
      image_aspect_ratio, input_aspect_ratio = original_width / original_height, width / height
      distorted_aspect = image_aspect_ratio / input_aspect_ratio

      delta_h, delta_w = 0.0, 0.0
      pullin_h, pullin_w = 0.0, 0.0
      if distorted_aspect > 1:
        delta_h = ((original_width / input_aspect_ratio) - original_height) / 2
      else:
        delta_w = ((original_height * input_aspect_ratio) - original_width) / 2

      if letter_box is None:
        rwidth = original_width + delta_w + delta_w
        rheight = original_height + delta_h + delta_h
        if rheight < height and rwidth < width:
          pullin_h = ((height - rheight) * rheight / height) / 2
          pullin_w = ((width - rwidth) * rwidth / width) / 2

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
        constant_values=get_pad_value())
    pad_info = tf.stack([
        tf.cast(tf.shape(cropped_image)[:2], tf.float32),
        tf.cast(tf.shape(image_)[:2], dtype=tf.float32),
        tf.ones_like(original_dims, dtype=tf.float32),
        (-tf.cast(pad[:2], tf.float32))
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
  """Builds a unifed affine transformation to spatially augment the image."""

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
  a = deg_to_rad(rand_uniform_strong(-degrees, degrees, seed=seed))
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
  Px = rand_uniform_strong(-perspective, perspective, seed=seed)
  Py = rand_uniform_strong(-perspective, perspective, seed=seed)
  prespective_warp = tf.tensor_scatter_nd_update(prespective_warp,
                                                 [[2, 0], [2, 1]], [Px, Py])
  prespective_warp_boxes = tf.tensor_scatter_nd_update(prespective_warp,
                                                       [[2, 0], [2, 1]],
                                                       [-Px, -Py])

  # Compute a random scaling to apply.
  scale = tf.eye(3, dtype=tf.float32)
  s = rand_uniform_strong(scale_min, scale_max, seed=seed)
  scale = tf.tensor_scatter_nd_update(scale, [[0, 0], [1, 1]], [1 / s, 1 / s])
  scale_boxes = tf.tensor_scatter_nd_update(scale, [[0, 0], [1, 1]], [s, s])

  # Compute a random Translation to apply.
  translation = tf.eye(3)
  if (random_pad and height * s < ch and width * s < cw):
    # The image is contained within the image and arbitrarily translated to
    # locations with in the image.
    center = center_boxes = tf.eye(3, dtype=tf.float32)
    Tx = rand_uniform_strong(-1, 0, seed=seed) * (cw / s - width)
    Ty = rand_uniform_strong(-1, 0, seed=seed) * (ch / s - height)
  else:
    # The image can be translated outside of the output resolution window
    # but the image is translated relative to the output resolution not the
    # input image resolution.
    Tx = rand_uniform_strong(0.5 - translate, 0.5 + translate, seed=seed)
    Ty = rand_uniform_strong(0.5 - translate, 0.5 + translate, seed=seed)

    # Center and Scale the image such that the window of translation is
    # contained to the output resolution.
    dx, dy = (width - cw / s) / width, (height - ch / s) / height
    sx, sy = 1 - dx, 1 - dy
    bx, by = dx / 2, dy / 2
    Tx, Ty = bx + (sx * Tx), by + (sy * Ty)

    # Scale the translation to width and height of the image.
    Tx *= width
    Ty *= height

  translation = tf.tensor_scatter_nd_update(translation, [[0, 2], [1, 2]],
                                            [Tx, Ty])
  translation_boxes = tf.tensor_scatter_nd_update(translation, [[0, 2], [1, 2]],
                                                  [-Tx, -Ty])

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
  image = tfa.image.transform(
      image,
      affine,
      fill_value=get_pad_value(),
      output_shape=desired_size,
      interpolation='bilinear')

  desired_size = tf.cast(desired_size, tf.float32)
  return image, affine_matrix, [image_size, desired_size, affine_boxes]


# ops for box clipping and cleaning
def affine_warp_boxes(affine, boxes, output_size, box_history):

  def _get_corners(box):
    """Get the corner of each box as a tuple of (x, y) coordinates"""
    ymi, xmi, yma, xma = tf.split(box, 4, axis=-1)
    tl = tf.concat([xmi, ymi], axis=-1)
    bl = tf.concat([xmi, yma], axis=-1)
    tr = tf.concat([xma, ymi], axis=-1)
    br = tf.concat([xma, yma], axis=-1)
    return tf.concat([tl, bl, tr, br], axis=-1)

  def _corners_to_boxes(corner):
    """Convert (x, y) corner tuples back into boxes in the format
    [ymin, xmin, ymax, xmax]"""
    corner = tf.reshape(corner, [-1, 4, 2])
    y = corner[..., 1]
    x = corner[..., 0]
    y_min = tf.reduce_min(y, axis=-1)
    x_min = tf.reduce_min(x, axis=-1)
    y_max = tf.reduce_max(y, axis=-1)
    x_max = tf.reduce_max(x, axis=-1)
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)

  def _aug_boxes(affine_matrix, box):
    """Apply an affine transformation matrix M to the boxes to get the 
    randomly augmented boxes"""
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
  conda = clipped_width > wh_thr
  condb = clipped_height > wh_thr

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
  # Shift and scale the input boxes.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Check the hitory of the boxes.
  box_history *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  box_history -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Clip the shifted and scaled boxes.
  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, box_history


def apply_infos(boxes,
                infos,
                affine=None,
                shuffle_boxes=False,
                area_thresh=0.1,
                seed=None,
                augment=True):
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
  output_size = tf.cast([640, 640], tf.float32)
  if augment:
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
    (
        boxes,  # Clipped final boxes. 
        box_history) = resize_and_crop_boxes(
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

    (
        boxes,  # Clipped final boxes. 
        box_history) = affine_warp_boxes(
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
  if augment:
    boxes_ = bbox_ops.denormalize_boxes(boxes, output_size)
    box_history_ = bbox_ops.denormalize_boxes(box_history, output_size)
    inds = boxes_candidates(boxes_, box_history_, area_thr=area_thresh)
    # Select and gather the good boxes.
    if shuffle_boxes:
      inds = tf.random.shuffle(inds, seed=seed)
  else:
    boxes = box_history
    boxes_ = bbox_ops.denormalize_boxes(boxes, output_size)
    inds = bbox_ops.get_non_empty_box_indices(boxes_)
  boxes = tf.gather(boxes, inds)
  return boxes, inds


def _gen_viable_box_mask(boxes):
  """Generate a mask to filter the boxes to only those with in the image. """
  equal = tf.reduce_all(tf.math.less_equal(boxes[..., 2:4], 0), axis=-1)
  lower_bound = tf.reduce_any(tf.math.less(boxes[..., 0:2], 0.0), axis=-1)
  upper_bound = tf.reduce_any(
      tf.math.greater_equal(boxes[..., 0:2], 1.0), axis=-1)

  negative_mask = tf.logical_or(tf.logical_or(equal, lower_bound), upper_bound)
  return tf.logical_not(negative_mask)


def _get_box_locations(anchors, mask, boxes):
  """Calculate the number of anchors associated with each ground truth box."""
  box_mask = _gen_viable_box_mask(boxes)

  mask = tf.reshape(mask, [1, 1, 1, -1])
  box_mask = tf.reshape(box_mask, [-1, 1, 1])
  anchors = tf.expand_dims(anchors, axis=-1)

  # split the anchors into the best matches and other wise
  anchors_primary, anchors_alternate = tf.split(anchors, [1, -1], axis=-2)
  anchors_alternate = tf.concat(
      [-tf.ones_like(anchors_primary), anchors_alternate], axis=-2)

  # convert all the masks into index locations
  viable_primary = tf.where(
      tf.squeeze(tf.logical_and(box_mask, anchors_primary == mask), axis=0))
  viable_alternate = tf.where(
      tf.squeeze(tf.logical_and(box_mask, anchors_alternate == mask), axis=0))
  viable_full = tf.where(
      tf.squeeze(tf.logical_and(box_mask, anchors == mask), axis=0))

  # compute the number of anchors associated with each ground truth box.
  acheck = tf.reduce_any(anchors == mask, axis=-1)
  repititions = tf.squeeze(
      tf.reduce_sum(tf.cast(acheck, mask.dtype), axis=-1), axis=0)

  # cast to int32
  viable_primary = tf.cast(viable_primary, tf.int32)
  viable_alternate = tf.cast(viable_alternate, tf.int32)
  viable_full = tf.cast(viable_full, tf.int32)
  return repititions, viable_primary, viable_alternate, viable_full


def _write_sample(box, anchor_id, offset, sample, ind_val, ind_sample, height,
                  width, num_written):
  """Find the correct x,y indexs for each box in the output groundtruth."""

  anchor_index = tf.convert_to_tensor([tf.cast(anchor_id, tf.int32)])
  gain = tf.cast(tf.convert_to_tensor([width, height]), box.dtype)

  y = box[1] * height
  x = box[0] * width

  y_index = tf.convert_to_tensor([tf.cast(y, tf.int32)])
  x_index = tf.convert_to_tensor([tf.cast(x, tf.int32)])
  grid_idx = tf.concat([y_index, x_index, anchor_index], axis=-1)
  ind_val = ind_val.write(num_written, grid_idx)
  ind_sample = ind_sample.write(num_written, sample)
  num_written += 1

  if offset > 0:
    offset = tf.cast(offset, x.dtype)
    grid_xy = tf.cast(tf.convert_to_tensor([x, y]), x.dtype)
    clamp = lambda x, ma: tf.maximum(
        tf.minimum(x, tf.cast(ma, x.dtype)), tf.zeros_like(x))

    grid_xy_index = grid_xy - tf.floor(grid_xy)
    positive_shift = ((grid_xy_index < offset) & (grid_xy > 1.))
    negative_shift = ((grid_xy_index > (1 - offset)) & (grid_xy < (gain - 1.)))

    shifts = [
        positive_shift[0], positive_shift[1], negative_shift[0],
        negative_shift[1]
    ]
    offset = tf.cast([[1, 0], [0, 1], [-1, 0], [0, -1]], offset.dtype) * offset

    for i in range(4):
      if shifts[i]:
        x_index = tf.convert_to_tensor([tf.cast(x - offset[i, 0], tf.int32)])
        y_index = tf.convert_to_tensor([tf.cast(y - offset[i, 1], tf.int32)])
        grid_idx = tf.concat([
            clamp(y_index, height - 1),
            clamp(x_index, width - 1), anchor_index
        ],
                             axis=-1)
        ind_val = ind_val.write(num_written, grid_idx)
        ind_sample = ind_sample.write(num_written, sample)
        num_written += 1
  return ind_val, ind_sample, num_written


def _write_grid(viable, num_reps, boxes, classes, ious, ind_val, ind_sample,
                height, width, num_written, num_instances, offset):
  """Iterate all viable anchor boxes and write each sample to groundtruth."""

  const = tf.cast(tf.convert_to_tensor([1.]), dtype=boxes.dtype)
  num_viable = tf.shape(viable)[0]
  for val in range(num_viable):
    idx = viable[val]
    obj_id, anchor, anchor_idx = idx[0], idx[1], idx[2]
    if num_written >= num_instances:
      break

    reps = tf.convert_to_tensor([num_reps[obj_id]])
    box = boxes[obj_id]
    cls_ = classes[obj_id]
    iou = tf.convert_to_tensor([ious[obj_id, anchor]])
    sample = tf.concat([box, const, cls_, iou, reps], axis=-1)

    ind_val, ind_sample, num_written = _write_sample(box, anchor_idx, offset,
                                                     sample, ind_val,
                                                     ind_sample, height, width,
                                                     num_written)
  return ind_val, ind_sample, num_written


def _write_anchor_free_grid(boxes,
                            classes,
                            height,
                            width,
                            num_written,
                            stride,
                            fpn_limits,
                            center_radius=2.5):
  """Iterate all boxes and write to grid without anchors boxes."""
  gen = loss_utils.GridGenerator(
      masks=None, anchors=[[1, 1]], scale_anchors=stride)
  grid_points = gen(width, height, 1, boxes.dtype)[0]
  grid_points = tf.squeeze(grid_points, axis=0)
  box_list = boxes
  class_list = classes

  grid_points = (grid_points + 0.5) * stride
  x_centers, y_centers = grid_points[..., 0], grid_points[..., 1]
  boxes *= (tf.convert_to_tensor([width, height, width, height]) * stride)
  tlbr_boxes = box_ops.xcycwh_to_yxyx(boxes)

  boxes = tf.reshape(boxes, [1, 1, -1, 4])
  tlbr_boxes = tf.reshape(tlbr_boxes, [1, 1, -1, 4])
  mask = tf.reshape(class_list != -1, [1, 1, -1])

  # check if the box is in the receptive feild of the this fpn level
  b_t = y_centers - tlbr_boxes[..., 0]
  b_l = x_centers - tlbr_boxes[..., 1]
  b_b = tlbr_boxes[..., 2] - y_centers
  b_r = tlbr_boxes[..., 3] - x_centers
  box_delta = tf.stack([b_t, b_l, b_b, b_r], axis=-1)
  if fpn_limits is not None:
    max_reg_targets_per_im = tf.reduce_max(box_delta, axis=-1)
    gt_min = max_reg_targets_per_im >= fpn_limits[0]
    gt_max = max_reg_targets_per_im <= fpn_limits[1]
    is_in_boxes = tf.logical_and(gt_min, gt_max)
  else:
    is_in_boxes = tf.reduce_min(box_delta, axis=-1) > 0.0
  is_in_boxes = tf.logical_and(is_in_boxes, mask)
  is_in_boxes_all = tf.reduce_any(is_in_boxes, axis=(0, 1), keepdims=True)

  # check if the center is in the receptive feild of the this fpn level
  c_t = y_centers - (boxes[..., 1] - center_radius * stride)
  c_l = x_centers - (boxes[..., 0] - center_radius * stride)
  c_b = (boxes[..., 1] + center_radius * stride) - y_centers
  c_r = (boxes[..., 0] + center_radius * stride) - x_centers
  centers_delta = tf.stack([c_t, c_l, c_b, c_r], axis=-1)
  is_in_centers = tf.reduce_min(centers_delta, axis=-1) > 0.0
  is_in_centers = tf.logical_and(is_in_centers, mask)
  is_in_centers_all = tf.reduce_any(is_in_centers, axis=(0, 1), keepdims=True)

  # colate all masks to get the final locations
  is_in_index = tf.logical_or(is_in_boxes_all, is_in_centers_all)
  is_in_boxes_and_center = tf.logical_and(is_in_boxes, is_in_centers)
  is_in_boxes_and_center = tf.logical_and(is_in_index, is_in_boxes_and_center)

  # construct the index update grid
  reps = tf.reduce_sum(tf.cast(is_in_boxes_and_center, tf.int16), axis=-1)
  indexes = tf.cast(tf.where(is_in_boxes_and_center), tf.int32)
  y, x, t = tf.split(indexes, 3, axis=-1)

  boxes = tf.gather_nd(box_list, t)
  classes = tf.cast(tf.gather_nd(class_list, t), boxes.dtype)
  reps = tf.gather_nd(reps, tf.concat([y, x], axis=-1))
  reps = tf.cast(tf.expand_dims(reps, axis=-1), boxes.dtype)
  conf = tf.ones_like(classes)

  # return the samples and the indexes
  samples = tf.concat([boxes, conf, classes, conf, reps], axis=-1)
  indexes = tf.concat([y, x, tf.zeros_like(t)], axis=-1)
  num_written = tf.shape(reps)[0]
  return indexes, samples, num_written


def build_grided_gt_ind(y_true,
                        mask,
                        sizew,
                        sizeh,
                        dtype,
                        scale_xy,
                        scale_num_inst,
                        use_tie_breaker,
                        stride,
                        fpn_limits=None):
  """Convert ground truth for use in loss functions.
  
  Args:
    y_true: tf.Tensor[] ground truth
      [batch, box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
    mask: list of the anchor boxes choresponding to the output,
      ex. [1, 2, 3] tells this layer to predict only the first 3 anchors
      in the total.
    size: the dimensions of this output, for regular, it progresses from
      13, to 26, to 52
    num_classes: `integer` for the number of classes
    dtype: expected output datatype
    scale_xy: A `float` to represent the amount the boxes are scaled in the
      loss function.
    scale_num_inst: A `float` to represent the scale at which to multiply the
      number of predicted boxes by to get the number of instances to write
      to the grid.
  Return:
    tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
  """
  # unpack required components from the input ground truth
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.expand_dims(tf.cast(y_true['classes'], dtype=dtype), axis=-1)
  anchors = tf.cast(y_true['best_anchors'], dtype)
  ious = tf.cast(y_true['best_iou_match'], dtype)

  width = tf.cast(sizew, boxes.dtype)
  height = tf.cast(sizeh, boxes.dtype)
  # get the number of anchor boxes used for this anchor scale
  len_masks = len(mask)
  # number of anchors
  num_instances = tf.shape(boxes)[-2] * scale_num_inst

  # rescale the x and y centers to the size of the grid [size, size]
  pull_in = tf.cast(0.5 * (scale_xy - 1), boxes.dtype)
  mask = tf.cast(mask, dtype=dtype)
  num_reps, viable_primary, viable_alternate, viable = _get_box_locations(
      anchors, mask, boxes)

  # tensor arrays for tracking samples
  num_written = 0

  if fpn_limits is not None:
    (indexes, samples,
     num_written) = _write_anchor_free_grid(boxes, classes, height, width,
                                            num_written, stride, fpn_limits)
  else:
    ind_val = tf.TensorArray(
        tf.int32, size=0, dynamic_size=True, element_shape=[
            3,
        ])
    ind_sample = tf.TensorArray(
        dtype, size=0, dynamic_size=True, element_shape=[
            8,
        ])

    if pull_in > 0.0:
      (ind_val, ind_sample,
       num_written) = _write_grid(viable, num_reps, boxes, classes, ious,
                                  ind_val, ind_sample, height, width,
                                  num_written, num_instances, pull_in)
    else:
      (ind_val, ind_sample,
       num_written) = _write_grid(viable_primary, num_reps, boxes, classes,
                                  ious, ind_val, ind_sample, height, width,
                                  num_written, num_instances, 0.0)

      if use_tie_breaker:
        (ind_val, ind_sample,
         num_written) = _write_grid(viable_alternate, num_reps, boxes, classes,
                                    ious, ind_val, ind_sample, height, width,
                                    num_written, num_instances, 0.0)
    indexes = ind_val.stack()
    samples = ind_sample.stack()

  (_, ind_mask, _, _, num_reps) = tf.split(samples, [4, 1, 1, 1, 1], axis=-1)
  full = tf.zeros([sizeh, sizew, len_masks, 1], dtype=dtype)
  full = tf.tensor_scatter_nd_add(full, indexes, ind_mask)

  if num_written >= num_instances:
    tf.print("clipped")

  indexs = pad_max_instances(indexes, num_instances, pad_value=0, pad_axis=0)
  samples = pad_max_instances(samples, num_instances, pad_value=0, pad_axis=0)
  return indexs, samples, full


def get_best_anchor(y_true,
                    anchors,
                    width=1,
                    height=1,
                    iou_thresh=0.25,
                    best_match_only=False):
  """
  get the correct anchor that is assoiciated with each box using IOU
  
  Args:
    y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
    anchors: list or tensor for the anchor boxes to be used in prediction
      found via Kmeans
    width: int for the image width
    height: int for the image height
  Return:
    tf.Tensor: y_true with the anchor associated with each ground truth
    box known
  """
  with tf.name_scope('get_best_anchor'):
    is_batch = True
    ytrue_shape = y_true.get_shape()
    if ytrue_shape.ndims == 2:
      is_batch = False
      y_true = tf.expand_dims(y_true, 0)
    elif ytrue_shape.ndims is None:
      is_batch = False
      y_true = tf.expand_dims(y_true, 0)
      y_true.set_shape([None] * 3)
    elif ytrue_shape.ndims != 3:
      raise ValueError('\'box\' (shape %s) must have either 3 or 4 dimensions.')

    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)
    scaler = tf.convert_to_tensor([width, height])

    true_wh = tf.cast(y_true[..., 2:4], dtype=tf.float32) * scaler
    anchors = tf.cast(anchors, dtype=tf.float32)
    k = tf.shape(anchors)[0]

    anchors = tf.expand_dims(
        tf.concat([tf.zeros_like(anchors), anchors], axis=-1), axis=0)
    truth_comp = tf.concat([tf.zeros_like(true_wh), true_wh], axis=-1)

    if iou_thresh >= 1.0:
      anchors = tf.expand_dims(anchors, axis=-2)
      truth_comp = tf.expand_dims(truth_comp, axis=-3)

      aspect = truth_comp[..., 2:4] / anchors[..., 2:4]
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.maximum(aspect, 1 / aspect)
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.reduce_max(aspect, axis=-1)

      values, indexes = tf.math.top_k(
          tf.transpose(-aspect, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      values = -values
      ind_mask = tf.cast(values < iou_thresh, dtype=indexes.dtype)
    else:
      # iou_raw = box_ops.compute_iou(truth_comp, anchors)
      truth_comp = box_ops.xcycwh_to_yxyx(truth_comp)
      anchors = box_ops.xcycwh_to_yxyx(anchors)
      iou_raw = box_ops.aggregated_comparitive_iou(
          truth_comp,
          anchors,
          iou_type=3,
      )
      values, indexes = tf.math.top_k(
          iou_raw,  #tf.transpose(iou_raw, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      ind_mask = tf.cast(values >= iou_thresh, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    if best_match_only:
      iou_index = ((indexes[..., 0:] + 1) * ind_mask[..., 0:]) - 1
    else:
      iou_index = tf.concat([
          tf.expand_dims(indexes[..., 0], axis=-1),
          ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
      ],
                            axis=-1)

    true_prod = tf.reduce_prod(true_wh, axis=-1, keepdims=True)
    iou_index = tf.where(true_prod > 0, iou_index, tf.zeros_like(iou_index) - 1)

    if not is_batch:
      iou_index = tf.squeeze(iou_index, axis=0)
      values = tf.squeeze(values, axis=0)
  return tf.cast(iou_index, dtype=tf.float32), tf.cast(values, dtype=tf.float32)
