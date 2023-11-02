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

"""Utils for processing datasets features."""

from typing import Any, MutableMapping, Optional, Tuple

from dmvr import processors
import simclr.data_util as simclr_data
import tensorflow as tf, tf_keras

sample_sequence = processors.sample_sequence
sample_linsapce_sequence = processors.sample_linspace_sequence
decode_jpeg = processors.decode_jpeg
random_flip_left_right = processors.random_flip_left_right
normalize_image = processors.normalize_image

_VGG_EXPANSION_RATIO = 1.25


def update_image_info(state: MutableMapping[str, tf.Tensor],
                      current_image_info: tf.Tensor):
  """Updates image info by merging current augmentation into the last one.

  NOTE: this is not a generic-purposed function and should only be used in this
    codebase. In this function, image_info tensor encodes the information of the
    image and the applied preprocessing. It is in the format of
    [[original_height, original_width],
     [desired_height, desired_width],
     [y_scale, x_scale],
     [y_offset, x_offset]],
    where [desired_height, desired_width] is the actual scaled image size;
    [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
    dimension / original dimension; [y_offset, x_offset] is the upper-left
    coordinates to perform image cropping.

  Args:
    state: dict containing 'image_info' of historical image augmentation.
    current_image_info: the data augmentation info of current step.

  Returns:
    state: updated dict of 'image_info'
  """
  image_info = state.pop('image_info')
  # image_info in shape [4, 2]
  i1, t1, s1, o1 = tf.unstack(image_info, axis=0)
  i2, t2, s2, o2 = tf.unstack(current_image_info, axis=0)

  # last target image_size(t1) should equalt to current input image_size(i2).
  tf.debugging.assert_equal(t1, i2,
                            message='last target size != current input size. '
                            'wrong augmentation order?')
  i3 = i1
  t3 = t2
  s3 = s1 * s2
  o3 = o1 * s2 + o2
  new_image_info = tf.stack([i3, t3, s3, o3], axis=0)
  state['image_info'] = new_image_info


def multi_crop_image(frames: tf.Tensor,
                     target_height: int,
                     target_width: int) -> tf.Tensor:
  """Three uniform crops of the image sequence.

  If requested size is bigger than image size, image is padded with 0.

  Args:
    frames: A Tensor of dimension [timesteps, in_height, in_width, channels].
    target_height: Target cropped image height.
    target_width: Target cropped image width.

  Returns:
    A Tensor of shape [timesteps, out_height, out_width, channels] of type uint8
    with the cropped images.
  """
  shape = tf.shape(frames)
  static_shape = frames.shape.as_list()
  seq_len = shape[0] if static_shape[0] is None else static_shape[0]
  height = shape[1] if static_shape[1] is None else static_shape[1]
  width = shape[2] if static_shape[2] is None else static_shape[2]
  channels = shape[3] if static_shape[3] is None else static_shape[3]

  size = tf.convert_to_tensor(
      (seq_len, target_height, target_width, channels))

  offset_1 = tf.broadcast_to([0, 0, 0, 0], [4])

  portrait_offset_2 = tf.cast(height, tf.float32) / 2 - target_height // 2
  landscape_offset_2 = tf.cast(width, tf.float32) / 2 - target_width // 2
  offset_2 = tf.cond(
      tf.greater_equal(height, width),
      true_fn=lambda: tf.broadcast_to([0, portrait_offset_2, 0, 0], [4]),
      false_fn=lambda: tf.broadcast_to([0, 0, landscape_offset_2, 0], [4]))

  portrait_offset_3 = tf.cast(height, tf.float32) - target_height
  landscape_offset_3 = tf.cast(width, tf.float32) - target_width
  offset_3 = tf.cond(
      tf.greater_equal(height, width),
      true_fn=lambda: tf.broadcast_to([0, portrait_offset_3, 0, 0], [4]),
      false_fn=lambda: tf.broadcast_to([0, 0, landscape_offset_3, 0], [4]))

  crops = []
  for offset in [offset_1, offset_2, offset_3]:
    offset = tf.cast(tf.math.round(offset), tf.int32)
    crops.append(tf.slice(frames, offset, size))
  frames = tf.concat(crops, axis=0)
  return frames


def resize_and_crop(
    frames: tf.Tensor,
    min_resize: int,
    crop_size: int,
    is_flow: bool = False,
    is_random: bool = False,
    seed: Optional[int] = None,
    state: Optional[MutableMapping[str, Any]] = None) -> tf.Tensor:
  """Resizes the smallest and crops frames.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    min_resize: Minimum size of the final image dimensions.
    crop_size: Crop size of the final image dimensions.
    is_flow: If is flow, will modify the raw values to account for the resize.
      For example, if the flow image is resized by a factor k, we need to
      multiply the flow values by the same factor k since one pixel displacement
      in the resized image corresponds to only 1/k pixel displacement in the
      original image.
    is_random: Whether perform random crop or central crop.
    seed: Random seed.
    state: the dictionary contains data processing states.
  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype where min(output_h, output_w) = min_resize.
  """
  if is_flow and frames.dtype != tf.float32:
    raise ValueError('If is_flow, frames should be given in float32.')

  if min_resize < crop_size:
    raise ValueError('min_resize should be larger than crop_size. Got '
                     f'({min_resize}, {crop_size}).')

  if is_random:
    min_resize = tf.random.uniform((),
                                   minval=min_resize,
                                   maxval=_VGG_EXPANSION_RATIO * min_resize,
                                   dtype=tf.float32)

  shape = tf.shape(input=frames)
  image_size = tf.cast(shape[1:3], tf.float32)
  input_h = image_size[0]
  input_w = image_size[1]

  scale = tf.cast(min_resize / input_h, tf.float32)
  scale = tf.maximum(scale, tf.cast(min_resize / input_w, tf.float32))

  scale_h = input_h * scale
  scale_w = input_w * scale

  def resize_fn():
    """Function wraper to perform bilinear image resizing."""
    frames_resized = tf.image.resize(
        frames, (scale_h, scale_w), method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(frames_resized, frames.dtype)

  should_resize = tf.math.logical_or(tf.not_equal(input_w, scale_w),
                                     tf.not_equal(input_h, scale_h))
  frames = tf.cond(
      pred=should_resize, true_fn=resize_fn, false_fn=lambda: frames)

  if is_flow:
    # Apply a multiplier to keep the right magnitude in the flow.
    frames = frames * tf.cast(scale_h / input_h, tf.float32)

  shape = tf.shape(input=frames)
  image_size = tf.cast(shape[1:3], tf.float32)
  # If a static_shape is available (e.g. when using this method from add_image
  # method), it will be used to have an output tensor with static shape.
  static_shape = frames.shape.as_list()
  seq_len = shape[0] if static_shape[0] is None else static_shape[0]
  channels = shape[3] if static_shape[3] is None else static_shape[3]
  size = tf.convert_to_tensor(value=(seq_len, crop_size, crop_size, channels))
  if is_random:
    # Limit of possible offset in order to fit the entire crop:
    # [1, input_h - target_h + 1, input_w - target_w + 1, 1].
    limit = shape - size + 1
    offset = tf.random.uniform(
        shape=(4,),
        dtype=tf.int32,
        maxval=tf.int32.max,
        seed=seed) % limit  # [0, offset_h, offset_w, 0]
  else:
    # Central spatial crop.
    offset = tf.convert_to_tensor(
        (0, tf.cast((image_size[0] - crop_size) / 2, dtype=tf.int32),
         tf.cast((image_size[1] - crop_size) / 2, dtype=tf.int32), 0))

  frames = tf.slice(frames, offset, size)

  if state is not None:
    # Note: image_info encodes the information of the image and the applied
    # preprocessing. It is in the format of
    # [[original_height, original_width], [desired_height, desired_width],
    #  [y_scale, x_scale], [y_offset, x_offset]],
    # where [desired_height, desired_width] is the actual scaled image size,
    # and [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
    # dimension / original dimension. [y_offset, x_offset] is the upper-left
    # coordinates to perform image cropping.
    image_info = tf.stack([
        tf.convert_to_tensor((input_h, input_w), tf.float32),
        tf.convert_to_tensor((crop_size, crop_size), tf.float32),
        tf.convert_to_tensor((scale, scale), tf.float32),
        tf.cast(offset[1:3], tf.float32)])

    if 'image_info' not in state:
      state['image_info'] = image_info
    else:
      update_image_info(state, image_info)

  return frames


def random_crop_resize(frames: tf.Tensor,
                       output_height: int,
                       output_width: int,
                       num_frames: int,
                       num_channels: int,
                       aspect_ratio: Tuple[float, float],
                       area_range: Tuple[float, float],
                       state: MutableMapping[str, tf.Tensor]) -> tf.Tensor:
  """First crops clip with jittering and then resizes.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    output_height: Resized image height.
    output_width: Resized image width.
    num_frames: Number of input frames per clip.
    num_channels: Number of channels of the clip.
    aspect_ratio: Float tuple with the aspect range for cropping.
    area_range: Float tuple with the area range for cropping.
    state: A mutable dictionary passed to the stateful functions and might be
      modified in order to keep metadata.
  Returns:
    A Tensor of shape [timesteps, output_height, output_width, channels] of type
      frames.dtype.
  """
  shape = tf.shape(frames)
  image_size = tf.cast(shape[1:3], tf.float32)
  seq_len, _, _, channels = shape[0], shape[1], shape[2], shape[3]
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  factor = output_width / output_height
  aspect_ratio = (aspect_ratio[0] * factor, aspect_ratio[1] * factor)
  sample_distorted_bbox = tf.image.sample_distorted_bounding_box(
      shape[1:],
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio,
      area_range=area_range,
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bbox
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  size = tf.convert_to_tensor((
      seq_len, target_height, target_width, channels))
  offset = tf.convert_to_tensor((0, offset_y, offset_x, 0))
  frames = tf.slice(frames, offset, size)
  frames = tf.cast(
      tf.image.resize(frames, (output_height, output_width)),
      frames.dtype)
  frames.set_shape((num_frames, output_height, output_width, num_channels))
  image_scale = (
      tf.convert_to_tensor((output_height, output_width), tf.float32) /
      tf.cast(bbox_size[:2], tf.float32))

  # Note: image_info encodes the information of the image and the applied
  # preprocessing. It is in the format of
  # [[original_height, original_width], [desired_height, desired_width],
  #  [y_scale, x_scale], [y_offset, x_offset]],
  # where [desired_height, desired_width] is the actual scaled image size,
  # and [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
  # dimension / original dimension. [y_offset, x_offset] is the upper-left
  # coordinates to perform image cropping.
  offset = tf.convert_to_tensor((offset_y, offset_x), tf.float32)
  offset *= image_scale
  image_info = tf.stack([
      image_size,
      tf.convert_to_tensor((output_height, output_width), tf.float32),
      image_scale,
      offset])
  if 'image_info' not in state:
    state['image_info'] = image_info
  else:
    update_image_info(state, image_info)
  return frames


def resize_smallest(
    frames: tf.Tensor,
    min_resize: int,
    is_flow: bool = False,
    state: Optional[MutableMapping[str, tf.Tensor]] = None) -> tf.Tensor:
  """Resizes frames so that `min(height, width)` is equal to `min_resize`.

  This function will do nothing if the `min(height, width)` is already equal to
  `min_resize`. This allows to save compute time.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    min_resize: Minimum size of the final image dimensions.
    is_flow: If is flow, will modify the raw values to account for the resize.
      For example, if the flow image is resized by a factor k, we need to
      multiply the flow values by the same factor k since one pixel displacement
      in the resized image corresponds to only 1/k pixel displacement in the
      original image.
    state: A mutable dictionary passed to the stateful functions and might be
      modified in order to keep metadata.

  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] of same type as
    input, where `min(output_h, output_w)` is `min_resize`.
  """
  if is_flow and frames.dtype != tf.float32:
    raise ValueError('If `is_flow`, frames should be given in `tf.float32`.')
  shape = tf.shape(input=frames)
  input_h = shape[1]
  input_w = shape[2]

  output_h = tf.maximum(min_resize, (input_h * min_resize) // input_w)
  output_w = tf.maximum(min_resize, (input_w * min_resize) // input_h)

  def resize_fn():
    """Bilinear resize function wrapper."""
    frames_resized = tf.image.resize(
        frames, (output_h, output_w), method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(frames_resized, frames.dtype)

  should_resize = tf.math.logical_or(tf.not_equal(input_w, output_w),
                                     tf.not_equal(input_h, output_h))
  frames = tf.cond(
      pred=should_resize, true_fn=resize_fn, false_fn=lambda: frames)

  if is_flow:
    # Apply a multiplier to keep the right magnitude in the flow.
    frames = frames * tf.cast(output_h / input_h, tf.float32)

  if state is not None:
    # Note: image_info encodes the information of the image and the applied
    # preprocessing. It is in the format of
    # [[original_height, original_width], [desired_height, desired_width],
    #  [y_scale, x_scale], [y_offset, x_offset]],
    # where [desired_height, desired_width] is the actual scaled image size,
    # and [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
    # dimension / original dimension.
    image_info = tf.stack([
        tf.convert_to_tensor((input_h, input_w), tf.float32),
        tf.convert_to_tensor((output_h, output_w), tf.float32),
        tf.convert_to_tensor(
            (output_h / input_h, output_w / input_w), tf.float32),
        tf.zeros([2], tf.float32)])

    if 'image_info' not in state:
      state['image_info'] = image_info
    else:
      update_image_info(state, image_info)

  return frames


def random_square_crop_by_scale(
    image: tf.Tensor,
    max_border: int = 128,
    scale_min: float = 0.6,
    scale_max: float = 1.3,
    num_scales: int = 8,
    seed: Optional[int] = None,
    state: Optional[MutableMapping[str, tf.Tensor]] = None) -> tf.Tensor:
  """Randomly crops a square in proportion to scale and image size.

   Extract a square sized crop from an image whose side length is sampled by
   randomly scaling the maximum spatial dimension of the image. If part of
   the crop falls outside the image, it is filled with zeros.
   The augmentation is borrowed from [1]: https://arxiv.org/abs/1904.07850

  Args:
    image: rank 4 float32 tensor containing images in shape
      [time, height, width, channels].
    max_border: The maximum size of the border. The border defines distance in
      pixels to the image boundaries that will not be considered as a center of
      a crop. To make sure that the border does not go over the center of the
      image, we chose the border value by computing the minimum k, such that
      (max_border / (2**k)) < image_dimension/2.
    scale_min: The minimum value for scale.
    scale_max: The maximum value for scale.
    num_scales: The number of discrete scale values to sample between
      [scale_min, scale_max]
    seed: Random seed.
    state: The dictionary contains random state.

  Returns:
    output_image: image which is the same rank as input image.
  """

  def _random_integer(minval, maxval, seed):
    return tf.random.uniform(
        [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

  def _get_crop_border(border, size):
    border = tf.cast(border, tf.float32)
    size = tf.cast(size, tf.float32)

    i = tf.math.ceil(tf.math.log(2.0 * border / size) / tf.math.log(2.0))
    divisor = tf.math.pow(2.0, i)
    divisor = tf.clip_by_value(divisor, 1, border)
    divisor = tf.cast(divisor, tf.int32)
    return tf.cast(border, tf.int32) // divisor

  img_shape = tf.shape(image)
  height, width = img_shape[1], img_shape[2]
  scales = tf.linspace(scale_min, scale_max, num_scales)
  random_id = _random_integer(0, num_scales, seed=seed)
  scale = scales[random_id]

  image_size = scale * tf.cast(tf.maximum(height, width), tf.float32)
  image_size = tf.cast(image_size, tf.int32)
  h_border = _get_crop_border(max_border, height)
  w_border = _get_crop_border(max_border, width)

  y_center = _random_integer(h_border,
                             tf.cast(height, tf.int32) - h_border + 1, seed)

  x_center = _random_integer(w_border,
                             tf.cast(width, tf.int32) - w_border + 1, seed)

  half_size = tf.cast(image_size / 2, tf.int32)
  crop_ymin, crop_ymax = y_center - half_size, y_center + half_size
  crop_xmin, crop_xmax = x_center - half_size, x_center + half_size

  ymin = tf.maximum(crop_ymin, 0)
  xmin = tf.maximum(crop_xmin, 0)
  ymax = tf.minimum(crop_ymax, height - 1)
  xmax = tf.minimum(crop_xmax, width - 1)

  cropped_image = image[:, ymin:ymax, xmin:xmax, :]
  offset_y = tf.maximum(0, ymin - crop_ymin)
  offset_x = tf.maximum(0, xmin - crop_xmin)

  output_image = tf.image.pad_to_bounding_box(
      cropped_image, offset_height=offset_y, offset_width=offset_x,
      target_height=image_size, target_width=image_size)

  if state is not None:
    # if (padding) else (cropping)
    padding_y = tf.cast(offset_y > 0, tf.int32)
    box_offset_y = padding_y * (-offset_y) + (1 - padding_y) * crop_ymin
    padding_x = tf.cast(offset_x > 0, tf.int32)
    box_offset_x = padding_x * (-offset_x) + (1 - padding_x) * crop_xmin

    # Note: image_info encodes the information of the image and the applied
    # preprocessing. It is in the format of
    # [[original_height, original_width], [desired_height, desired_width],
    #  [y_scale, x_scale], [y_offset, x_offset]],
    # where [desired_height, desired_width] is the actual scaled image size,
    # and [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
    # dimension / original dimension.
    image_info = tf.stack([
        tf.convert_to_tensor((height, width), tf.float32),
        tf.convert_to_tensor((image_size, image_size), tf.float32),
        tf.convert_to_tensor((1.0, 1.0), tf.float32),
        tf.convert_to_tensor((box_offset_y, box_offset_x), tf.float32)])

    if 'image_info' not in state:
      state['image_info'] = image_info
    else:
      update_image_info(state, image_info)

  return output_image


def resize_and_pad(
    frames: tf.Tensor,
    max_resize: int,
    pad_size: int,
    random: bool = False,
    seed: Optional[int] = None,
    state: Optional[MutableMapping[str, tf.Tensor]] = None) -> tf.Tensor:
  """Resizes the largest and pads frames.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    max_resize: Maximum size of the final image dimensions.
    pad_size: Pad size of the final image dimensions.
    random: If true, perform random crop; otherwise, perform central crop.
    seed: Random seed.
    state: The dictionary contains random state.
  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype where min(output_h, output_w) = max_resize.
  """
  if max_resize > pad_size:
    raise ValueError('max_resize should not be larger than pad_size. Got '
                     f'({max_resize}, {pad_size}).')

  pad_color = tf.reduce_mean(frames, axis=[0, 1, 2])

  shape = tf.shape(input=frames)
  image_size = tf.cast(shape[1:3], tf.float32)
  input_h = image_size[0]
  input_w = image_size[1]

  scale = tf.cast(max_resize / input_h, tf.float32)
  scale = tf.minimum(scale, tf.cast(max_resize / input_w, tf.float32))

  scale_h = input_h * scale
  scale_w = input_w * scale

  frames_resized = tf.image.resize(
      frames, (scale_h, scale_w), method=tf.image.ResizeMethod.BILINEAR)
  frames = tf.cast(frames_resized, frames.dtype)

  shape = tf.shape(input=frames)
  image_size = tf.cast(shape[1:3], tf.float32)
  size = tf.convert_to_tensor(value=(pad_size, pad_size))
  if random:
    # Limit of possible offset in order to fit the entire crop:
    # [target_h - input_h + 1, target_w - input_w + 1].
    limit = size - tf.cast(image_size, tf.int32) + 1
    offset = tf.random.uniform(
        shape=(2,),
        dtype=tf.int32,
        maxval=tf.int32.max,
        seed=seed) % limit  # [offset_h, offset_w]
    offset_height, offset_width = offset[0], offset[1]
  else:
    # Central spatial pad.
    offset_height = tf.cast((pad_size - image_size[0]) / 2, tf.int32)
    offset_width = tf.cast((pad_size - image_size[1]) / 2, tf.int32)
    offset = tf.convert_to_tensor([offset_height, offset_width])

  padded_frames = tf.image.pad_to_bounding_box(
      frames,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=pad_size,
      target_width=pad_size)

  # Setting color of the padded pixels
  frames_ones = tf.ones_like(frames)
  frames_ones_padded = tf.image.pad_to_bounding_box(
      frames_ones,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=pad_size,
      target_width=pad_size)
  frames_color_padded = (1 - frames_ones_padded) * pad_color
  padded_frames += frames_color_padded

  if state is not None:
    # Note: image_info encodes the information of the image and the applied
    # preprocessing. It is in the format of
    # [[original_height, original_width], [desired_height, desired_width],
    #  [y_scale, x_scale], [y_offset, x_offset]],
    # where [desired_height, desired_width] is the actual scaled image size,
    # and [y_scale, x_scale] is the scaling factor, which is the ratio of scaled
    # dimension / original dimension.
    image_info = tf.stack([
        tf.convert_to_tensor((input_h, input_w), tf.float32),
        tf.convert_to_tensor((pad_size, pad_size), tf.float32),
        tf.convert_to_tensor((scale, scale), tf.float32),
        # set to -1. * offset because it's padding.
        -1. * tf.cast(offset, tf.float32)])

    if 'image_info' not in state:
      state['image_info'] = image_info
    else:
      update_image_info(state, image_info)
  return padded_frames


def crop_or_pad_features(features: tf.Tensor,
                         max_num_features: int,
                         feature_dimension: int,
                         constant_values: int = 0) -> tf.Tensor:
  """Crops or pads given sequence of features vectors.

  Args:
    features: Tensor features of shape [T, feature_length] or [feature_length],
      features of shape [feature_length] is expanded as [1, feature_length].
    max_num_features: Maximum number of words in final result.
    feature_dimension: The dimensionality of feature vector.
    constant_values: The constant value used to padd the input tensor.

  Returns:
    A Tensor of shape [T, max_num_features * feature_dimension].
  """
  if len(features.shape) == 1:
    features = tf.expand_dims(features, 0)

  num_features = tf.shape(input=features)[1]
  max_length = max_num_features * feature_dimension
  paddings = ((0, 0),
              (0, tf.maximum(0, max_length - num_features)))
  features = tf.pad(
      tensor=features[:, :max_length],
      paddings=paddings,
      constant_values=constant_values)
  features.set_shape((None, max_length))
  return features


def sample_sequence_by_segment(
    inputs: MutableMapping[str, tf.Tensor],
    num_steps: int,
    sample_target_key: str = 'image',
    is_training: bool = True,
) -> MutableMapping[str, tf.Tensor]:
  """Samples a single clip from a given sequence by segments.

  Args:
    inputs: dict with sample_target and keyframe_index features.
    num_steps: Number of steps (e.g. frames) to take.
    sample_target_key: the key for the sample target.
    is_training: whether in training mode.

  Returns:
    Modified inputs with sampled target.
  """
  sequence = inputs[sample_target_key]
  sequence_length = tf.shape(sequence)[0]
  segment_size = (tf.cast(sequence_length, tf.float32) - 1) / num_steps
  indices = []
  for i in range(num_steps):
    start = tf.cast(tf.math.round(segment_size * i), tf.int32)
    end = tf.cast(tf.math.round(segment_size * (i + 1)), tf.int32)
    # special hanle if end == start.
    end = tf.maximum(end, start+1)
    if is_training:
      indices.append(
          tf.random.uniform(shape=(), minval=start, maxval=end, dtype=tf.int32))
    else:
      indices.append((start + end) // 2)
  indices = tf.stack(indices, axis=0)
  output = tf.gather(sequence, indices)
  inputs[sample_target_key] = output
  return inputs


def sample_sequence_around_keyframe(
    inputs: MutableMapping[str, tf.Tensor],
    num_steps: int,
    stride: int,
    sample_target_key: str = 'image',
    keyframe_index_key: str = 'keyframe_index'
) -> MutableMapping[str, tf.Tensor]:
  """Samples a single segment around keyframe from a given sequence.

  Args:
    inputs: dict with sample_target and keyframe_index features.
    num_steps: Number of steps (e.g. frames) to take.
    stride: the stride of sampling.
    sample_target_key: the key for the sample target.
    keyframe_index_key: the key for the keyframe index.

  Returns:
    Modified inputs with sampled target.
  """
  if sample_target_key not in inputs:
    raise ValueError(f'{sample_target_key} is not found in input dictionary.')
  if keyframe_index_key not in inputs:
    raise ValueError(f'{keyframe_index_key} is not found in input dictionary.')

  sequence = inputs[sample_target_key]

  keyframe_index = tf.cast(inputs[keyframe_index_key], dtype=tf.int32)
  keyframe_index = tf.squeeze(keyframe_index)  # assuming there's a single one

  sequence_length = tf.shape(sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)

  early = keyframe_index - (num_steps * stride) // 2
  late = keyframe_index + (num_steps * stride) // 2
  offset = tf.maximum(0, early)

  pad_before = tf.maximum(0, -early)
  pad_after = tf.maximum(0, late - sequence_length)

  # Repeat first and last frames appropriately.
  repeat = sequence.shape.ndims - 1
  pad_before = [pad_before] + [1] * repeat
  pad_before_frame = tf.tile(sequence[:1], pad_before)
  pad_after = [pad_after] + [1] * repeat
  pad_after_frame = tf.tile(sequence[-1:], pad_after)
  sequence = tf.concat([pad_before_frame, sequence, pad_after_frame], axis=0)

  indices = tf.linspace(offset, offset + (num_steps - 1) * stride, num_steps)
  indices = tf.cast(indices, dtype=tf.int32)[:num_steps]
  indices.set_shape((num_steps))

  output = tf.gather(sequence, indices)
  inputs[sample_target_key] = output
  return inputs


def random_color_augmentation(frames: tf.Tensor,
                              zero_centering_image: bool = False,
                              color_jitter_prob: float = 0.8,
                              color_drop_prob: float = 0.0) -> tf.Tensor:
  """Standard color augmentation for video.

  Args:
    frames: the input video frames.
    zero_centering_image: Whether the image frames has been zero centered.
    color_jitter_prob: The probability to apply color jittering.
    color_drop_prob: The probability to apply color dropping.

  Returns:
    The frames with color augmentations.
  """

  def color_jitter_fn(video):
    """Does the color augmentations."""
    if zero_centering_image:
      video = 0.5 * (video + 1.0)
    video = tf.image.random_brightness(video, max_delta=32.0 / 255.0)
    video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
    video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
    video = tf.image.random_hue(video, max_delta=0.2)
    video = tf.clip_by_value(video, 0.0, 1.0)
    if zero_centering_image:
      video = 2 * (video - 0.5)
    return video

  def color_drop_fn(video):
    """Does the color drop."""
    video = tf.image.rgb_to_grayscale(video)
    video = tf.tile(video, [1, 1, 1, 3])
    return video

  frames = simclr_data.random_apply(color_jitter_fn, color_jitter_prob, frames)
  frames = simclr_data.random_apply(color_drop_fn, color_drop_prob, frames)
  return frames


def random_blur_and_solarize(
    frames: tf.Tensor,
    zero_centering_image: bool = False,
    blur_prob: float = 1.0,
    solarize_prob: float = 0.2) -> tf.Tensor:
  """Randomly blur and solarize a video clip.

  Args:
    frames: The input image frames tensor.
    zero_centering_image: Whether the images are zero centered.
    blur_prob: The probability to apply random bluring.
    solarize_prob: The probability to apply random solarization.

  Returns:
    The images with random bluriness and solarization.
  """
  if zero_centering_image:
    frames = 0.5 * (frames + 1.0)

  height = tf.shape(frames)[1]
  width = tf.shape(frames)[2]
  frames = simclr_data.random_blur(frames, height, width, blur_prob)

  def solarize_fn(image):
    """Randomly solarizes images."""
    image = image * tf.cast(tf.less(image, 0.5), tf.float32) + (
        1.0 - image) * tf.cast(tf.greater_equal(image, 0.5), tf.float32)
    return image
  frames = simclr_data.random_apply(solarize_fn, solarize_prob, frames)
  frames = tf.clip_by_value(frames, 0.0, 1.0)

  if zero_centering_image:
    frames = 2 * (frames - 0.5)
  return frames
