# Copyright 2017 Google Inc.
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

"""Utilities for PixelDA model."""
import math

# Dependency imports

import tensorflow as tf

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


def remove_depth(images):
  """Takes a batch of images and remove depth channel if present."""
  if images.shape.as_list()[-1] == 4:
    return images[:, :, :, 0:3]
  return images


def image_grid(images, max_grid_size=4):
  """Given images and N, return first N^2 images as an NxN image grid.

  Args:
    images: a `Tensor` of size [batch_size, height, width, channels]
    max_grid_size: Maximum image grid height/width

  Returns:
    Single image batch, of dim [1, h*n, w*n, c]
  """
  images = remove_depth(images)
  batch_size = images.shape.as_list()[0]
  grid_size = min(int(math.sqrt(batch_size)), max_grid_size)
  assert images.shape.as_list()[0] >= grid_size * grid_size

  # If we have a depth channel
  if images.shape.as_list()[-1] == 4:
    images = images[:grid_size * grid_size, :, :, 0:3]
    depth = tf.image.grayscale_to_rgb(images[:grid_size * grid_size, :, :, 3:4])

    images = tf.reshape(images, [-1, images.shape.as_list()[2], 3])
    split = tf.split(0, grid_size, images)
    depth = tf.reshape(depth, [-1, images.shape.as_list()[2], 3])
    depth_split = tf.split(0, grid_size, depth)
    grid = tf.concat(split + depth_split, 1)
    return tf.expand_dims(grid, 0)
  else:
    images = images[:grid_size * grid_size, :, :, :]
    images = tf.reshape(
        images, [-1, images.shape.as_list()[2],
                 images.shape.as_list()[3]])
    split = tf.split(images, grid_size, 0)
    grid = tf.concat(split, 1)
    return tf.expand_dims(grid, 0)


def source_and_output_image_grid(output_images,
                                 source_images=None,
                                 max_grid_size=4):
  """Create NxN image grid for output, concatenate source grid if given.

  Makes grid out of output_images and, if provided, source_images, and
  concatenates them.

  Args:
    output_images: [batch_size, h, w, c] tensor of images
    source_images: optional[batch_size, h, w, c] tensor of images
    max_grid_size: Image grid height/width

  Returns:
    Single image batch, of dim [1, h*n, w*n, c]


  """
  output_grid = image_grid(output_images, max_grid_size=max_grid_size)
  if source_images is not None:
    source_grid = image_grid(source_images, max_grid_size=max_grid_size)
    # Make sure they have the same # of channels before concat
    # Assumes either 1 or 3 channels
    if output_grid.shape.as_list()[-1] != source_grid.shape.as_list()[-1]:
      if output_grid.shape.as_list()[-1] == 1:
        output_grid = tf.tile(output_grid, [1, 1, 1, 3])
      if source_grid.shape.as_list()[-1] == 1:
        source_grid = tf.tile(source_grid, [1, 1, 1, 3])
    output_grid = tf.concat([output_grid, source_grid], 1)
  return output_grid


def summarize_model(end_points):
  """Summarizes the given model via its end_points.

  Args:
    end_points: A dictionary of end_point names to `Tensor`.
  """
  tf.summary.histogram('domain_logits_transferred',
                       tf.sigmoid(end_points['transferred_domain_logits']))

  tf.summary.histogram('domain_logits_target',
                       tf.sigmoid(end_points['target_domain_logits']))


def summarize_transferred_grid(transferred_images,
                               source_images=None,
                               name='Transferred'):
  """Produces a visual grid summarization of the image transferrence.

  Args:
    transferred_images: A `Tensor` of size [batch_size, height, width, c].
    source_images: A `Tensor` of size [batch_size, height, width, c].
    name: Name to use in summary name
  """
  if source_images is not None:
    grid = source_and_output_image_grid(transferred_images, source_images)
  else:
    grid = image_grid(transferred_images)
  tf.summary.image('%s_Images_Grid' % name, grid, max_outputs=1)


def summarize_transferred(source_images,
                          transferred_images,
                          max_images=20,
                          name='Transferred'):
  """Produces a visual summary of the image transferrence.

  This summary displays the source image, transferred image, and a grayscale
  difference image which highlights the differences between input and output.

  Args:
    source_images: A `Tensor` of size [batch_size, height, width, channels].
    transferred_images: A `Tensor` of size [batch_size, height, width, channels]
    max_images: The number of images to show.
    name: Name to use in summary name

  Raises:
    ValueError: If number of channels in source and target are incompatible
  """
  source_channels = source_images.shape.as_list()[-1]
  transferred_channels = transferred_images.shape.as_list()[-1]
  if source_channels < transferred_channels:
    if source_channels != 1:
      raise ValueError(
          'Source must be 1 channel or same # of channels as target')
    source_images = tf.tile(source_images, [1, 1, 1, transferred_channels])
  if transferred_channels < source_channels:
    if transferred_channels != 1:
      raise ValueError(
          'Target must be 1 channel or same # of channels as source')
    transferred_images = tf.tile(transferred_images, [1, 1, 1, source_channels])
  diffs = tf.abs(source_images - transferred_images)
  diffs = tf.reduce_max(diffs, reduction_indices=[3], keep_dims=True)
  diffs = tf.tile(diffs, [1, 1, 1, max(source_channels, transferred_channels)])

  transition_images = tf.concat([
      source_images,
      transferred_images,
      diffs,
  ], 2)

  tf.summary.image(
      '%s_difference' % name, transition_images, max_outputs=max_images)


def summaries_color_distributions(images, name):
  """Produces a histogram of the color distributions of the images.

  Args:
    images: A `Tensor` of size [batch_size, height, width, 3].
    name: The name of the images being summarized.
  """
  tf.summary.histogram('color_values/%s' % name, images)


def summarize_images(images, name):
  """Produces a visual summary of the given images.

  Args:
    images: A `Tensor` of size [batch_size, height, width, 3].
    name: The name of the images being summarized.
  """
  grid = image_grid(images)
  tf.summary.image('%s_Images' % name, grid, max_outputs=1)
