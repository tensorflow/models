# Copyright 2018 Google, Inc. All Rights Reserved.
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



import collections
import functools
import threading
import tensorflow as tf
import matplotlib
import numpy as np
import time
import re
import math
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scipy.signal

from tensorflow.python.util import tf_should_use
from tensorflow.contrib.summary import summary_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.contrib.summary import gen_summary_ops

_DEBUG_DISABLE_SUMMARIES=False

class LoggingFileWriter(tf.summary.FileWriter):
  """A FileWriter that also logs things out.

  This is entirely for ease of debugging / not having to open up Tensorboard
  a lot.
  """

  def __init__(self, logdir, regexes=[], **kwargs):
    self.regexes = regexes
    super(LoggingFileWriter, self).__init__(logdir, **kwargs)

  def add_summary(self, summary, global_step):
    if type(summary) != tf.Summary:
      summary_p = tf.Summary()
      summary_p.ParseFromString(summary)
      summary = summary_p
    for s in summary.value:
      for exists in [re.match(p, s.tag) for p in self.regexes]:
        if exists is not None:
          tf.logging.info("%d ] %s : %f", global_step, s.tag, s.simple_value)
          break
    super(LoggingFileWriter, self).add_summary(summary, global_step)


def image_grid(images, max_grid_size=4, border=1):
  """Given images and N, return first N^2 images as an NxN image grid.

  Args:
    images: a `Tensor` of size [batch_size, height, width, channels]
    max_grid_size: Maximum image grid height/width

  Returns:
    Single image batch, of dim [1, h*n, w*n, c]
  """
  batch_size = images.shape.as_list()[0]
  to_pad = int((np.ceil(np.sqrt(batch_size)))**2 - batch_size)
  images = tf.pad(images, [[0, to_pad], [0, border], [0, border], [0, 0]])

  batch_size = images.shape.as_list()[0]
  grid_size = min(int(np.sqrt(batch_size)), max_grid_size)
  assert images.shape.as_list()[0] >= grid_size * grid_size

  # If we have a depth channel
  if images.shape.as_list()[-1] == 4:
    images = images[:grid_size * grid_size, :, :, 0:3]
    depth = tf.image.grayscale_to_rgb(images[:grid_size * grid_size, :, :, 3:4])

    images = tf.reshape(images, [-1, images.shape.as_list()[2], 3])
    split = tf.split(images, grid_size, axis=0)
    depth = tf.reshape(depth, [-1, images.shape.as_list()[2], 3])
    depth_split = tf.split(depth, grid_size, axis=0)
    grid = tf.concat(split + depth_split, 1)
    return tf.expand_dims(grid, 0)
  else:
    images = images[:grid_size * grid_size, :, :, :]
    images = tf.reshape(
        images, [-1, images.shape.as_list()[2],
                 images.shape.as_list()[3]])
    split = tf.split(value=images, num_or_size_splits=grid_size, axis=0)
    grid = tf.concat(split, 1)
    return tf.expand_dims(grid, 0)


def first_layer_weight_image(weight, shape):
  weight_image = tf.reshape(weight,
                            shape + [tf.identity(weight).shape.as_list()[1]])
  # [winx, winy, wout]
  mean, var = tf.nn.moments(weight_image, [0,1,2], keep_dims=True)
  #mean, var = tf.nn.moments(weight_image, [0,1], keep_dims=True)
  weight_image = (weight_image - mean) / tf.sqrt(var + 1e-5)
  weight_image = (weight_image + 1.0) / 2.0
  weight_image = tf.clip_by_value(weight_image, 0, 1)
  weight_image = tf.transpose(weight_image, (3, 0, 1, 2))
  grid = image_grid(weight_image, max_grid_size=10)
  return grid

def inner_layer_weight_image(weight):
  """Visualize a weight matrix of an inner layer.
  Add padding to make it square, then visualize as a gray scale image
  """
  weight = tf.identity(weight) # turn into a tensor
  weight = weight / (tf.reduce_max(tf.abs(weight), [0], keep_dims=True))
  weight = tf.reshape(weight, [1]+weight.shape.as_list() + [1])
  return weight


def activation_image(activations, label_onehot):
  """Make a row sorted by class for each activation. Put a black line around the activations."""
  labels = tf.argmax(label_onehot, axis=1)
  _, n_classes = label_onehot.shape.as_list()
  mean, var = tf.nn.moments(activations, [0, 1])
  activations = (activations - mean)/tf.sqrt(var+1e-5)

  activations = tf.clip_by_value(activations, -1, 1)
  activations = (activations + 1.0) / 2.0 # shift to [0, 1]

  canvas = []
  for i in xrange(n_classes):
    inds = tf.where(tf.equal(labels, i))

    def _gather():
      return tf.squeeze(tf.gather(activations, inds), 1)

    def _empty():
      return tf.zeros([0, activations.shape.as_list()[1]], dtype=tf.float32)

    assert inds.shape.as_list()[0] is None
    x = tf.cond(tf.equal(tf.shape(inds)[0], 0), _empty, _gather)
    canvas.append(x)
    canvas.append(tf.zeros([1, activations.shape.as_list()[1]]))
  canvas = tf.concat(canvas, 0)
  canvas = tf.reshape(canvas, [1, activations.shape.as_list()[0]+n_classes, canvas.shape.as_list()[1], 1])
  return canvas


def sorted_images(images, label_onehot):
  # images is [bs, x, y, c]
  labels = tf.argmax(label_onehot, axis=1)
  _, n_classes = label_onehot.shape.as_list()
  to_stack = []
  for i in xrange(n_classes):
    inds = tf.where(tf.equal(labels, i))

    def _gather():
      return tf.squeeze(tf.gather(images, inds), 1)

    def _empty():
      return tf.zeros([0] + images.shape.as_list()[1:], dtype=tf.float32)

    assert inds.shape.as_list()[0] is None
    x = tf.cond(tf.equal(tf.shape(inds)[0], 0), _empty, _gather)
    to_stack.append(x)
  # pad / trim all up to 10.
  padded = []
  for t in to_stack:
    n_found = tf.shape(t)[0]
    pad = tf.pad(t[0:10], tf.stack([tf.stack([0,tf.maximum(0, 10-n_found)]), [0,0], [0,0], [0,0]]))
    padded.append(pad)

  xs = [tf.concat(tf.split(p, 10), axis=1) for p in padded]
  ys = tf.concat(xs, axis=2)
  ys = tf.cast(tf.clip_by_value(ys, 0., 1.) * 255., tf.uint8)
  return ys
