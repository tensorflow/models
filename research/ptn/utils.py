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

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as p
# axes3d is being used implictly for visualization.
from mpl_toolkits.mplot3d import axes3d as p3  # pylint:disable=unused-import
import numpy as np
from PIL import Image
from skimage import measure

import tensorflow as tf


def save_image(inp_array, image_file):
  """Function that dumps the image to disk."""
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  image = Image.fromarray(inp_array)
  buf = StringIO.StringIO()
  image.save(buf, format='JPEG')
  with open(image_file, 'w') as f:
    f.write(buf.getvalue())
  return None


def image_flipud(images):
  """Function that flip (up-down) the np image."""
  quantity = images.get_shape().as_list()[0]
  image_list = []
  for k in xrange(quantity):
    image_list.append(tf.image.flip_up_down(images[k, :, :, :]))
  outputs = tf.stack(image_list)
  return outputs


def resize_image(inp_array, new_height, new_width):
  """Function that resize the np image."""
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  image = Image.fromarray(inp_array)
  # Reverse order
  image = image.resize((new_width, new_height))
  return np.array(image)


def display_voxel(points, vis_size=128):
  """Function to display 3D voxel."""
  try:
    data = visualize_voxel_spectral(points, vis_size)
  except ValueError:
    data = visualize_voxel_scatter(points, vis_size)
  return data


def visualize_voxel_spectral(points, vis_size=128):
  """Function to visualize voxel (spectral)."""
  points = np.rint(points)
  points = np.swapaxes(points, 0, 2)
  fig = p.figure(figsize=(1, 1), dpi=vis_size)
  verts, faces = measure.marching_cubes_classic(points, 0, spacing=(0.1, 0.1, 0.1))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(
      verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral_r', lw=0.1)
  ax.set_axis_off()
  fig.tight_layout(pad=0)
  fig.canvas.draw()
  data = np.fromstring(
      fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(
          vis_size, vis_size, 3)
  p.close('all')
  return data


def visualize_voxel_scatter(points, vis_size=128):
  """Function to visualize voxel (scatter)."""
  points = np.rint(points)
  points = np.swapaxes(points, 0, 2)
  fig = p.figure(figsize=(1, 1), dpi=vis_size)
  ax = fig.add_subplot(111, projection='3d')
  x = []
  y = []
  z = []
  (x_dimension, y_dimension, z_dimension) = points.shape
  for i in range(x_dimension):
    for j in range(y_dimension):
      for k in range(z_dimension):
        if points[i, j, k]:
          x.append(i)
          y.append(j)
          z.append(k)
  ax.scatter3D(x, y, z)
  ax.set_axis_off()
  fig.tight_layout(pad=0)
  fig.canvas.draw()
  data = np.fromstring(
      fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(
          vis_size, vis_size, 3)
  p.close('all')
  return data

