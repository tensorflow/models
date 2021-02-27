# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for processing video dataset features."""

from typing import Optional, Tuple
import tensorflow as tf


def _sample_or_pad_sequence_indices(sequence: tf.Tensor,
                                    num_steps: int,
                                    stride: int,
                                    offset: tf.Tensor) -> tf.Tensor:
  """Returns indices to take for sampling or padding sequences to fixed size."""
  sequence_length = tf.shape(sequence)[0]
  sel_idx = tf.range(sequence_length)

  # Repeats sequence until num_steps are available in total.
  max_length = num_steps * stride + offset
  num_repeats = tf.math.floordiv(
      max_length + sequence_length - 1, sequence_length)
  sel_idx = tf.tile(sel_idx, [num_repeats])

  steps = tf.range(offset, offset + num_steps * stride, stride)
  return tf.gather(sel_idx, steps)


def sample_linspace_sequence(sequence: tf.Tensor,
                             num_windows: int,
                             num_steps: int,
                             stride: int) -> tf.Tensor:
  """Samples `num_windows` segments from sequence with linearly spaced offsets.

  The samples are concatenated in a single `tf.Tensor` in order to have the same
  format structure per timestep (e.g. a single frame). If `num_steps` * `stride`
  is bigger than the number of timesteps, the sequence is repeated. This
  function can be used in evaluation in order to extract enough segments to span
  the entire sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_windows: Number of windows retrieved from the sequence.
    num_steps: Number of steps (e.g. frames) to take.
    stride: Distance to sample between timesteps.

  Returns:
    A single `tf.Tensor` with first dimension `num_windows` * `num_steps`. The
    tensor contains the concatenated list of `num_windows` tensors which offsets
    have been linearly spaced from input.
  """
  sequence_length = tf.shape(sequence)[0]
  max_offset = tf.maximum(0, sequence_length - num_steps * stride)
  offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), num_windows)
  offsets = tf.cast(offsets, tf.int32)

  all_indices = []
  for i in range(num_windows):
    all_indices.append(_sample_or_pad_sequence_indices(
        sequence=sequence,
        num_steps=num_steps,
        stride=stride,
        offset=offsets[i]))

  indices = tf.concat(all_indices, axis=0)
  indices.set_shape((num_windows * num_steps,))
  return tf.gather(sequence, indices)


def sample_sequence(sequence: tf.Tensor,
                    num_steps: int,
                    random: bool,
                    stride: int,
                    seed: Optional[int] = None) -> tf.Tensor:
  """Samples a single segment of size `num_steps` from a given sequence.

  If `random` is not `True`, this function will simply sample the central window
  of the sequence. Otherwise, a random offset will be chosen in a way that the
  desired `num_steps` might be extracted from the sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    random: A boolean indicating whether to random sample the single window. If
      `True`, the offset is randomized. If `False`, the middle frame minus half
      of `num_steps` is the first frame.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.

  Returns:
    A single `tf.Tensor` with first dimension `num_steps` with the sampled
    segment.
  """
  sequence_length = tf.shape(sequence)[0]

  if random:
    sequence_length = tf.cast(sequence_length, tf.float32)
    max_offset = tf.cond(
        sequence_length > (num_steps - 1) * stride,
        lambda: sequence_length - (num_steps - 1) * stride,
        lambda: sequence_length)
    offset = tf.random.uniform(
        (),
        maxval=tf.cast(max_offset, dtype=tf.int32),
        dtype=tf.int32,
        seed=seed)
  else:
    offset = (sequence_length - num_steps * stride) // 2
    offset = tf.maximum(0, offset)

  indices = _sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      stride=stride,
      offset=offset)
  indices.set_shape((num_steps,))

  return tf.gather(sequence, indices)


def decode_jpeg(image_string: tf.Tensor, channels: int = 0) -> tf.Tensor:
  """Decodes JPEG raw bytes string into a RGB uint8 Tensor.

  Args:
    image_string: A `tf.Tensor` of type strings with the raw JPEG bytes where
      the first dimension is timesteps.
    channels: Number of channels of the JPEG image. Allowed values are 0, 1 and
      3. If 0, the number of channels will be calculated at runtime and no
      static shape is set.

  Returns:
    A Tensor of shape [T, H, W, C] of type uint8 with the decoded images.
  """
  return tf.map_fn(
      lambda x: tf.image.decode_jpeg(x, channels=channels),
      image_string, back_prop=False, dtype=tf.uint8)


def crop_image(frames: tf.Tensor,
               target_height: int,
               target_width: int,
               random: bool = False,
               num_crops: int = 1,
               seed: Optional[int] = None) -> tf.Tensor:
  """Crops the image sequence of images.

  If requested size is bigger than image size, image is padded with 0. If not
  random cropping, a central crop is performed if num_crops is 1.

  Args:
    frames: A Tensor of dimension [timesteps, in_height, in_width, channels].
    target_height: Target cropped image height.
    target_width: Target cropped image width.
    random: A boolean indicating if crop should be randomized.
    num_crops: Number of crops (support 1 for central crop and 3 for 3-crop).
    seed: A deterministic seed to use when random cropping.

  Returns:
    A Tensor of shape [timesteps, out_height, out_width, channels] of type uint8
    with the cropped images.
  """
  if random:
    # Random spatial crop.
    shape = tf.shape(frames)
    # If a static_shape is available (e.g. when using this method from add_image
    # method), it will be used to have an output tensor with static shape.
    static_shape = frames.shape.as_list()
    seq_len = shape[0] if static_shape[0] is None else static_shape[0]
    channels = shape[3] if static_shape[3] is None else static_shape[3]
    frames = tf.image.random_crop(
        frames, (seq_len, target_height, target_width, channels), seed)
  else:
    if num_crops == 1:
      # Central crop or pad.
      frames = tf.image.resize_with_crop_or_pad(frames, target_height,
                                                target_width)

    elif num_crops == 3:
      # Three-crop evaluation.
      shape = tf.shape(frames)
      static_shape = frames.shape.as_list()
      seq_len = shape[0] if static_shape[0] is None else static_shape[0]
      height = shape[1] if static_shape[1] is None else static_shape[1]
      width = shape[2] if static_shape[2] is None else static_shape[2]
      channels = shape[3] if static_shape[3] is None else static_shape[3]

      size = tf.convert_to_tensor(
          (seq_len, target_height, target_width, channels))

      offset_1 = tf.broadcast_to([0, 0, 0, 0], [4])
      # pylint:disable=g-long-lambda
      offset_2 = tf.cond(
          tf.greater_equal(height, width),
          true_fn=lambda: tf.broadcast_to([
              0, tf.cast(height, tf.float32) / 2 - target_height // 2, 0, 0
          ], [4]),
          false_fn=lambda: tf.broadcast_to([
              0, 0, tf.cast(width, tf.float32) / 2 - target_width // 2, 0
          ], [4]))
      offset_3 = tf.cond(
          tf.greater_equal(height, width),
          true_fn=lambda: tf.broadcast_to(
              [0, tf.cast(height, tf.float32) - target_height, 0, 0], [4]),
          false_fn=lambda: tf.broadcast_to(
              [0, 0, tf.cast(width, tf.float32) - target_width, 0], [4]))
      # pylint:disable=g-long-lambda

      crops = []
      for offset in [offset_1, offset_2, offset_3]:
        offset = tf.cast(tf.math.round(offset), tf.int32)
        crops.append(tf.slice(frames, offset, size))
      frames = tf.concat(crops, axis=0)

    else:
      raise NotImplementedError(
          f"Only 1-crop and 3-crop are supported. Found {num_crops!r}.")

  return frames


def resize_smallest(frames: tf.Tensor,
                    min_resize: int) -> tf.Tensor:
  """Resizes frames so that min(`height`, `width`) is equal to `min_resize`.

  This function will not do anything if the min(`height`, `width`) is already
  equal to `min_resize`. This allows to save compute time.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    min_resize: Minimum size of the final image dimensions.

  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype where min(output_h, output_w) = min_resize.
  """
  shape = tf.shape(frames)
  input_h = shape[1]
  input_w = shape[2]

  output_h = tf.maximum(min_resize, (input_h * min_resize) // input_w)
  output_w = tf.maximum(min_resize, (input_w * min_resize) // input_h)

  def resize_fn():
    frames_resized = tf.image.resize(frames, (output_h, output_w))
    return tf.cast(frames_resized, frames.dtype)

  should_resize = tf.math.logical_or(tf.not_equal(input_w, output_w),
                                     tf.not_equal(input_h, output_h))
  frames = tf.cond(should_resize, resize_fn, lambda: frames)

  return frames


def random_crop_resize(frames: tf.Tensor,
                       output_h: int,
                       output_w: int,
                       num_frames: int,
                       num_channels: int,
                       aspect_ratio: Tuple[float, float],
                       area_range: Tuple[float, float]) -> tf.Tensor:
  """First crops clip with jittering and then resizes to (output_h, output_w).

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    output_h: Resized image height.
    output_w: Resized image width.
    num_frames: Number of input frames per clip.
    num_channels: Number of channels of the clip.
    aspect_ratio: Float tuple with the aspect range for cropping.
    area_range: Float tuple with the area range for cropping.
  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype.
  """
  shape = tf.shape(frames)
  seq_len, _, _, channels = shape[0], shape[1], shape[2], shape[3]
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  factor = output_w / output_h
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
  offset = tf.convert_to_tensor((
      0, offset_y, offset_x, 0))
  frames = tf.slice(frames, offset, size)
  frames = tf.cast(
      tf.image.resize(frames, (output_h, output_w)),
      frames.dtype)
  frames.set_shape((num_frames, output_h, output_w, num_channels))
  return frames


def random_flip_left_right(
    frames: tf.Tensor,
    seed: Optional[int] = None) -> tf.Tensor:
  """Flips all the frames with a probability of 50%.

  Args:
    frames: A Tensor of shape [timesteps, input_h, input_w, channels].
    seed: A seed to use for the random sampling.

  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] eventually
    flipped left right.
  """
  is_flipped = tf.random.uniform(
      (), minval=0, maxval=2, dtype=tf.int32, seed=seed)

  frames = tf.cond(tf.equal(is_flipped, 1),
                   true_fn=lambda: tf.image.flip_left_right(frames),
                   false_fn=lambda: frames)
  return frames


def normalize_image(frames: tf.Tensor,
                    zero_centering_image: bool,
                    dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
  """Normalizes images.

  Args:
    frames: A Tensor of numbers.
    zero_centering_image: If True, results are in [-1, 1], if False, results are
      in [0, 1].
    dtype: Type of output Tensor.

  Returns:
    A Tensor of same shape as the input and of the given type.
  """
  frames = tf.cast(frames, dtype)
  if zero_centering_image:
    return frames * (2.0 / 255.0) - 1.0
  else:
    return frames / 255.0
