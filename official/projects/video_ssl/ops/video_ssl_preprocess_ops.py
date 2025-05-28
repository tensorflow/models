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

"""Utils for customed ops for video ssl."""

import functools
from typing import Optional
import tensorflow as tf, tf_keras


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def random_brightness(image, max_delta):
  """Distort brightness of image (SimCLRv2 style)."""
  factor = tf.random.uniform(
      [], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
  image = image * factor
  return image


def random_solarization(image, p=0.2):
  """Random solarize image."""
  def _transform(image):
    image = image * tf.cast(tf.less(image, 0.5), dtype=image.dtype) + (
        1.0 - image) * tf.cast(tf.greater_equal(image, 0.5), dtype=image.dtype)
    return image
  return random_apply(_transform, p=p, x=image)


def to_grayscale(image, keep_channels=True):
  """Turn the input image to gray scale.

  Args:
    image: The input image tensor.
    keep_channels: Whether maintaining the channel number for the image.
    If true, the transformed image will repeat three times in channel.
    If false, the transformed image will only have one channel.

  Returns:
    The distorted image tensor.
  """
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3])
  return image


def color_jitter(image, strength, random_order=True):
  """Distorts the color of the image (SimCLRv2 style).

  Args:
    image: The input image tensor.
    strength: The floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(
        image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(
        image, brightness, contrast, saturation, hue)


def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0):
  """Distorts the color of the image (jittering order is fixed, SimCLRv2 style).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0):
  """Distorts the color of the image (jittering order is random, SimCLRv2 style).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_transform():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness)
      def contrast_transform():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_transform():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_transform():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      # pylint:disable=g-long-lambda
      x = tf.cond(
          tf.less(i, 2), lambda: tf.cond(
              tf.less(i, 1), brightness_transform, contrast_transform),
          lambda: tf.cond(tf.less(i, 3), saturation_transform, hue_transform))
      # pylint:disable=g-long-lambda
      return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def random_color_jitter_3d(frames):
  """Applies temporally consistent color jittering to one video clip.

  Args:
    frames: `Tensor` of shape [num_frames, height, width, channels].

  Returns:
    A Tensor of shape [num_frames, height, width, channels] being color jittered
    with the same operation.
  """
  def random_color_jitter(image, p=1.0):
    def _transform(image):
      color_jitter_t = functools.partial(
          color_jitter, strength=1.0)
      image = random_apply(color_jitter_t, p=0.8, x=image)
      return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(_transform, p=p, x=image)

  num_frames, width, height, channels = frames.shape.as_list()
  big_image = tf.reshape(frames, [num_frames*width, height, channels])
  big_image = random_color_jitter(big_image)
  return tf.reshape(big_image, [num_frames, width, height, channels])


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.

  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.cast(kernel_size / 2, dtype=tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
  blur_filter = tf.exp(
      -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def random_blur(image, height, width, p=1.0):
  """Randomly blur an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  del width
  def _transform(image):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)


def random_blur_3d(frames, height, width, blur_probability=0.5):
  """Apply efficient batch data transformations.

  Args:
    frames: `Tensor` of shape [timesteps, height, width, 3].
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.

  Returns:
    Preprocessed feature list.
  """
  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf.cast(
        tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
        tf.float32)
    return selector

  frames_new = random_blur(frames, height, width, p=1.)
  selector = generate_selector(blur_probability, 1)
  frames = frames_new * selector + frames * (1 - selector)
  frames = tf.clip_by_value(frames, 0., 1.)

  return frames


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


def sample_ssl_sequence(sequence: tf.Tensor,
                        num_steps: int,
                        random: bool,
                        stride: int = 1,
                        num_windows: Optional[int] = 2) -> tf.Tensor:
  """Samples two segments of size num_steps randomly from a given sequence.

  Currently it only supports images, and specically designed for video self-
  supervised learning.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    random: A boolean indicating whether to random sample the single window. If
      True, the offset is randomized. Only True is supported.
    stride: Distance to sample between timesteps.
    num_windows: Number of sequence sampled.

  Returns:
    A single Tensor with first dimension num_steps with the sampled segment.
  """
  sequence_length = tf.shape(sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.float32)
  if random:
    max_offset = tf.cond(
        tf.greater(sequence_length, (num_steps - 1) * stride),
        lambda: sequence_length - (num_steps - 1) * stride,
        lambda: sequence_length)

    max_offset = tf.cast(max_offset, dtype=tf.float32)
    def cdf(k, power=1.0):
      """Cumulative distribution function for x^power."""
      p = -tf.math.pow(k, power + 1) / (
          power * tf.math.pow(max_offset, power + 1)) + k * (power + 1) / (
              power * max_offset)
      return p

    u = tf.random.uniform(())
    k_low = tf.constant(0, dtype=tf.float32)
    k_up = max_offset
    k = tf.math.floordiv(max_offset, 2.0)

    c = lambda k_low, k_up, k: tf.greater(tf.math.abs(k_up - k_low), 1.0)
    # pylint:disable=g-long-lambda
    b = lambda k_low, k_up, k: tf.cond(
        tf.greater(cdf(k), u),
        lambda: [k_low, k, tf.math.floordiv(k + k_low, 2.0)],
        lambda: [k, k_up, tf.math.floordiv(k_up + k, 2.0)])

    _, _, k = tf.while_loop(c, b, [k_low, k_up, k])
    delta = tf.cast(k, tf.int32)
    max_offset = tf.cast(max_offset, tf.int32)
    sequence_length = tf.cast(sequence_length, tf.int32)

    choice_1 = tf.cond(
        tf.equal(max_offset, sequence_length),
        lambda: tf.random.uniform((),
                                  maxval=tf.cast(max_offset, dtype=tf.int32),
                                  dtype=tf.int32),
        lambda: tf.random.uniform((),
                                  maxval=tf.cast(max_offset - delta,
                                                 dtype=tf.int32),
                                  dtype=tf.int32))
    choice_2 = tf.cond(
        tf.equal(max_offset, sequence_length),
        lambda: tf.random.uniform((),
                                  maxval=tf.cast(max_offset, dtype=tf.int32),
                                  dtype=tf.int32),
        lambda: choice_1 + delta)
    # pylint:disable=g-long-lambda
    shuffle_choice = tf.random.shuffle((choice_1, choice_2))
    offset_1 = shuffle_choice[0]
    offset_2 = shuffle_choice[1]

  else:
    raise NotImplementedError

  indices_1 = _sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      stride=stride,
      offset=offset_1)

  indices_2 = _sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      stride=stride,
      offset=offset_2)

  indices = tf.concat([indices_1, indices_2], axis=0)
  indices.set_shape((num_windows * num_steps,))
  output = tf.gather(sequence, indices)

  return output
