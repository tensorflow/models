# Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
"""Generator and discriminator for a progressive GAN model.

See https://arxiv.org/abs/1710.10196 for details about the model.

See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

import layers


class ResolutionSchedule(object):
  """Image resolution upscaling schedule."""

  def __init__(self, start_resolutions=(4, 4), scale_base=2, num_resolutions=4):
    """Initializer.

    Args:
      start_resolutions: An tuple of integers of HxW format for start image
      resolutions. Defaults to (4, 4).
      scale_base: An integer of resolution base multiplier. Defaults to 2.
      num_resolutions: An integer of how many progressive resolutions (including
          `start_resolutions`). Defaults to 4.
    """
    self._start_resolutions = start_resolutions
    self._scale_base = scale_base
    self._num_resolutions = num_resolutions

  @property
  def start_resolutions(self):
    return tuple(self._start_resolutions)

  @property
  def scale_base(self):
    return self._scale_base

  @property
  def num_resolutions(self):
    return self._num_resolutions

  @property
  def final_resolutions(self):
    """Returns the final resolutions."""
    return tuple([
        r * self._scale_base**(self._num_resolutions - 1)
        for r in self._start_resolutions
    ])

  def scale_factor(self, block_id):
    """Returns the scale factor for network block `block_id`."""
    if block_id < 1 or block_id > self._num_resolutions:
      raise ValueError('`block_id` must be in [1, {}]'.format(
          self._num_resolutions))
    return self._scale_base**(self._num_resolutions - block_id)


def block_name(block_id):
  """Returns the scope name for the network block `block_id`."""
  return 'progressive_gan_block_{}'.format(block_id)


def min_total_num_images(stable_stage_num_images, transition_stage_num_images,
                         num_blocks):
  """Returns the minimum total number of images.

  Computes the minimum total number of images required to reach the desired
  `resolution`.

  Args:
    stable_stage_num_images: Number of images in the stable stage.
    transition_stage_num_images: Number of images in the transition stage.
    num_blocks: Number of network blocks.

  Returns:
    An integer of the minimum total number of images.
  """
  return (num_blocks * stable_stage_num_images +
          (num_blocks - 1) * transition_stage_num_images)


def compute_progress(current_image_id, stable_stage_num_images,
                     transition_stage_num_images, num_blocks):
  """Computes the training progress.

  The training alternates between stable phase and transition phase.
  The `progress` indicates the training progress, i.e. the training is at
  - a stable phase p if progress = p
  - a transition stage between p and p + 1 if progress = p + fraction
  where p = 0,1,2.,...

  Note the max value of progress is `num_blocks` - 1.

  In terms of LOD (of the original implementation):
  progress = `num_blocks` - 1 - LOD

  Args:
    current_image_id: An scalar integer `Tensor` of the current image id, count
        from 0.
    stable_stage_num_images: An integer representing the number of images in
        each stable stage.
    transition_stage_num_images: An integer representing the number of images in
        each transition stage.
    num_blocks: Number of network blocks.

  Returns:
    A scalar float `Tensor` of the training progress.
  """
  # Note when current_image_id >= min_total_num_images - 1 (which means we
  # are already at the highest resolution), we want to keep progress constant.
  # Therefore, cap current_image_id here.
  capped_current_image_id = tf.minimum(
      current_image_id,
      min_total_num_images(stable_stage_num_images, transition_stage_num_images,
                           num_blocks) - 1)

  stage_num_images = stable_stage_num_images + transition_stage_num_images
  progress_integer = tf.floordiv(capped_current_image_id, stage_num_images)
  progress_fraction = tf.maximum(
      0.0,
      tf.to_float(
          tf.mod(capped_current_image_id, stage_num_images) -
          stable_stage_num_images) / tf.to_float(transition_stage_num_images))
  return tf.to_float(progress_integer) + progress_fraction


def _generator_alpha(block_id, progress):
  """Returns the block output parameter for the generator network.

  The generator has N blocks with `block_id` = 1,2,...,N. Each block
  block_id outputs a fake data output(block_id). The generator output is a
  linear combination of all block outputs, i.e.
  SUM_block_id(output(block_id) * alpha(block_id, progress)) where
  alpha(block_id, progress) = _generator_alpha(block_id, progress). Note it
  garantees that SUM_block_id(alpha(block_id, progress)) = 1 for any progress.

  With a fixed block_id, the plot of alpha(block_id, progress) against progress
  is a 'triangle' with its peak at (block_id - 1, 1).

  Args:
    block_id: An integer of generator block id.
    progress: A scalar float `Tensor` of training progress.

  Returns:
    A scalar float `Tensor` of block output parameter.
  """
  return tf.maximum(0.0,
                    tf.minimum(progress - (block_id - 2), block_id - progress))


def _discriminator_alpha(block_id, progress):
  """Returns the block input parameter for discriminator network.

  The discriminator has N blocks with `block_id` = 1,2,...,N. Each block
  block_id accepts an
    - input(block_id) transformed from the real data and
    - the output of block block_id + 1, i.e. output(block_id + 1)
  The final input is a linear combination of them,
  i.e. alpha * input(block_id) + (1 - alpha) * output(block_id + 1)
  where alpha = _discriminator_alpha(block_id, progress).

  With a fixed block_id, alpha(block_id, progress) stays to be 1
  when progress <= block_id - 1, then linear decays to 0 when
  block_id - 1 < progress <= block_id, and finally stays at 0
  when progress > block_id.

  Args:
    block_id: An integer of generator block id.
    progress: A scalar float `Tensor` of training progress.

  Returns:
    A scalar float `Tensor` of block input parameter.
  """
  return tf.clip_by_value(block_id - progress, 0.0, 1.0)


def blend_images(x, progress, resolution_schedule, num_blocks):
  """Blends images of different resolutions according to `progress`.

  When training `progress` is at a stable stage for resolution r, returns
  image `x` downscaled to resolution r and then upscaled to `final_resolutions`,
  call it x'(r).

  Otherwise when training `progress` is at a transition stage from resolution
  r to 2r, returns a linear combination of x'(r) and x'(2r).

  Args:
    x: An image `Tensor` of NHWC format with resolution `final_resolutions`.
    progress: A scalar float `Tensor` of training progress.
    resolution_schedule: An object of `ResolutionSchedule`.
    num_blocks: An integer of number of blocks.

  Returns:
    An image `Tensor` which is a blend of images of different resolutions.
  """
  x_blend = []
  for block_id in range(1, num_blocks + 1):
    alpha = _generator_alpha(block_id, progress)
    scale = resolution_schedule.scale_factor(block_id)
    x_blend.append(alpha * layers.upscale(layers.downscale(x, scale), scale))
  return tf.add_n(x_blend)


def num_filters(block_id, fmap_base=4096, fmap_decay=1.0, fmap_max=256):
  """Computes number of filters of block `block_id`."""
  return int(min(fmap_base / math.pow(2.0, block_id * fmap_decay), fmap_max))


def generator(z,
              progress,
              num_filters_fn,
              resolution_schedule,
              num_blocks=None,
              kernel_size=3,
              colors=3,
              to_rgb_activation=None,
              scope='progressive_gan_generator',
              reuse=None):
  """Generator network for the progressive GAN model.

  Args:
    z: A `Tensor` of latent vector. The first dimension must be batch size.
    progress: A scalar float `Tensor` of training progress.
    num_filters_fn: A function that maps `block_id` to # of filters for the
        block.
    resolution_schedule: An object of `ResolutionSchedule`.
    num_blocks: An integer of number of blocks. None means maximum number of
        blocks, i.e. `resolution.schedule.num_resolutions`. Defaults to None.
    kernel_size: An integer of convolution kernel size.
    colors: Number of output color channels. Defaults to 3.
    to_rgb_activation: Activation function applied when output rgb.
    scope: A string or variable scope.
    reuse: Whether to reuse `scope`. Defaults to None which means to inherit
        the reuse option of the parent scope.

  Returns:
    A `Tensor` of model output and a dictionary of model end points.
  """
  if num_blocks is None:
    num_blocks = resolution_schedule.num_resolutions

  start_h, start_w = resolution_schedule.start_resolutions
  final_h, final_w = resolution_schedule.final_resolutions

  def _conv2d(scope, x, kernel_size, filters, padding='SAME'):
    return layers.custom_conv2d(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=lambda x: layers.pixel_norm(tf.nn.leaky_relu(x)),
        he_initializer_slope=0.0,
        scope=scope)

  def _to_rgb(x):
    return layers.custom_conv2d(
        x=x,
        filters=colors,
        kernel_size=1,
        padding='SAME',
        activation=to_rgb_activation,
        scope='to_rgb')

  end_points = {}

  with tf.variable_scope(scope, reuse=reuse):
    with tf.name_scope('input'):
      x = tf.contrib.layers.flatten(z)
      end_points['latent_vector'] = x

    with tf.variable_scope(block_name(1)):
      x = tf.expand_dims(tf.expand_dims(x, 1), 1)
      x = layers.pixel_norm(x)
      # Pad the 1 x 1 image to 2 * (start_h - 1) x 2 * (start_w - 1)
      # with zeros for the next conv.
      x = tf.pad(x, [[0] * 2, [start_h - 1] * 2, [start_w - 1] * 2, [0] * 2])
      # The output is start_h x start_w x num_filters_fn(1).
      x = _conv2d('conv0', x, (start_h, start_w), num_filters_fn(1), 'VALID')
      x = _conv2d('conv1', x, kernel_size, num_filters_fn(1))
      lods = [x]

    for block_id in range(2, num_blocks + 1):
      with tf.variable_scope(block_name(block_id)):
        x = layers.upscale(x, resolution_schedule.scale_base)
        x = _conv2d('conv0', x, kernel_size, num_filters_fn(block_id))
        x = _conv2d('conv1', x, kernel_size, num_filters_fn(block_id))
        lods.append(x)

    outputs = []
    for block_id in range(1, num_blocks + 1):
      with tf.variable_scope(block_name(block_id)):
        lod = _to_rgb(lods[block_id - 1])
        scale = resolution_schedule.scale_factor(block_id)
        lod = layers.upscale(lod, scale)
        end_points['upscaled_rgb_{}'.format(block_id)] = lod

        # alpha_i is used to replace lod_select. Note sum(alpha_i) is
        # garanteed to be 1.
        alpha = _generator_alpha(block_id, progress)
        end_points['alpha_{}'.format(block_id)] = alpha

        outputs.append(lod * alpha)

    predictions = tf.add_n(outputs)
    batch_size = z.shape[0].value
    predictions.set_shape([batch_size, final_h, final_w, colors])
    end_points['predictions'] = predictions

  return predictions, end_points


def discriminator(x,
                  progress,
                  num_filters_fn,
                  resolution_schedule,
                  num_blocks=None,
                  kernel_size=3,
                  scope='progressive_gan_discriminator',
                  reuse=None):
  """Discriminator network for the progressive GAN model.

  Args:
    x: A `Tensor`of NHWC format representing images of size `resolution`.
    progress: A scalar float `Tensor` of training progress.
    num_filters_fn: A function that maps `block_id` to # of filters for the
        block.
    resolution_schedule: An object of `ResolutionSchedule`.
    num_blocks: An integer of number of blocks. None means maximum number of
        blocks, i.e. `resolution.schedule.num_resolutions`. Defaults to None.
    kernel_size: An integer of convolution kernel size.
    scope: A string or variable scope.
    reuse: Whether to reuse `scope`. Defaults to None which means to inherit
        the reuse option of the parent scope.

  Returns:
    A `Tensor` of model output and a dictionary of model end points.
  """
  if num_blocks is None:
    num_blocks = resolution_schedule.num_resolutions

  def _conv2d(scope, x, kernel_size, filters, padding='SAME'):
    return layers.custom_conv2d(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=tf.nn.leaky_relu,
        he_initializer_slope=0.0,
        scope=scope)

  def _from_rgb(x, block_id):
    return _conv2d('from_rgb', x, 1, num_filters_fn(block_id))

  end_points = {}

  with tf.variable_scope(scope, reuse=reuse):
    x0 = x
    end_points['rgb'] = x0

    lods = []
    for block_id in range(num_blocks, 0, -1):
      with tf.variable_scope(block_name(block_id)):
        scale = resolution_schedule.scale_factor(block_id)
        lod = layers.downscale(x0, scale)
        end_points['downscaled_rgb_{}'.format(block_id)] = lod
        lod = _from_rgb(lod, block_id)
        # alpha_i is used to replace lod_select.
        alpha = _discriminator_alpha(block_id, progress)
        end_points['alpha_{}'.format(block_id)] = alpha
      lods.append((lod, alpha))

    lods_iter = iter(lods)
    x, _ = lods_iter.next()
    for block_id in range(num_blocks, 1, -1):
      with tf.variable_scope(block_name(block_id)):
        x = _conv2d('conv0', x, kernel_size, num_filters_fn(block_id))
        x = _conv2d('conv1', x, kernel_size, num_filters_fn(block_id - 1))
        x = layers.downscale(x, resolution_schedule.scale_base)
        lod, alpha = lods_iter.next()
        x = alpha * lod + (1.0 - alpha) * x

    with tf.variable_scope(block_name(1)):
      x = layers.scalar_concat(x, layers.minibatch_mean_stddev(x))
      x = _conv2d('conv0', x, kernel_size, num_filters_fn(1))
      x = _conv2d('conv1', x, resolution_schedule.start_resolutions,
                  num_filters_fn(0), 'VALID')
      end_points['last_conv'] = x
      logits = layers.custom_dense(x=x, units=1, scope='logits')
      end_points['logits'] = logits

  return logits, end_points
