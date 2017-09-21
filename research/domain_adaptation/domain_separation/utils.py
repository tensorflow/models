# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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
"""Auxiliary functions for domain adaptation related losses.
"""
import math
import tensorflow as tf


def create_summaries(end_points, prefix='', max_images=3, use_op_name=False):
  """Creates a tf summary per endpoint.

  If the endpoint is a 4 dimensional tensor it displays it as an image
  otherwise if it is a two dimensional one it creates a histogram summary.

  Args:
    end_points: a dictionary of name, tf tensor pairs.
    prefix: an optional string to prefix the summary with.
    max_images: the maximum number of images to display per summary.
    use_op_name: Use the op name as opposed to the shorter end_points key.
  """
  for layer_name in end_points:
    if use_op_name:
      name = end_points[layer_name].op.name
    else:
      name = layer_name
    if len(end_points[layer_name].get_shape().as_list()) == 4:
      # if it's an actual image do not attempt to reshape it
      if end_points[layer_name].get_shape().as_list()[-1] == 1 or end_points[
          layer_name].get_shape().as_list()[-1] == 3:
        visualization_image = end_points[layer_name]
      else:
        visualization_image = reshape_feature_maps(end_points[layer_name])
      tf.summary.image(
          '{}/{}'.format(prefix, name),
          visualization_image,
          max_outputs=max_images)
    elif len(end_points[layer_name].get_shape().as_list()) == 3:
      images = tf.expand_dims(end_points[layer_name], 3)
      tf.summary.image(
          '{}/{}'.format(prefix, name),
          images,
          max_outputs=max_images)
    elif len(end_points[layer_name].get_shape().as_list()) == 2:
      tf.summary.histogram('{}/{}'.format(prefix, name), end_points[layer_name])


def reshape_feature_maps(features_tensor):
  """Reshape activations for tf.summary.image visualization.

  Arguments:
    features_tensor: a tensor of activations with a square number of feature
                     maps, eg 4, 9, 16, etc.
  Returns:
    A composite image with all the feature maps that can be passed as an
    argument to tf.summary.image.
  """
  assert len(features_tensor.get_shape().as_list()) == 4
  num_filters = features_tensor.get_shape().as_list()[-1]
  assert num_filters > 0
  num_filters_sqrt = math.sqrt(num_filters)
  assert num_filters_sqrt.is_integer(
  ), 'Number of filters should be a square number but got {}'.format(
      num_filters)
  num_filters_sqrt = int(num_filters_sqrt)
  conv_summary = tf.unstack(features_tensor, axis=3)
  conv_one_row = tf.concat(axis=2, values=conv_summary[0:num_filters_sqrt])
  ind = 1
  conv_final = conv_one_row
  for ind in range(1, num_filters_sqrt):
    conv_one_row = tf.concat(axis=2,
                             values=conv_summary[
        ind * num_filters_sqrt + 0:ind * num_filters_sqrt + num_filters_sqrt])
    conv_final = tf.concat(
        axis=1, values=[tf.squeeze(conv_final), tf.squeeze(conv_one_row)])
    conv_final = tf.expand_dims(conv_final, -1)
  return conv_final


def accuracy(predictions, labels):
  """Calculates the classificaton accuracy.

  Args:
    predictions: the predicted values, a tensor whose size matches 'labels'.
    labels: the ground truth values, a tensor of any size.

  Returns:
    a tensor whose value on evaluation returns the total accuracy.
  """
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def compute_upsample_values(input_tensor, upsample_height, upsample_width):
  """Compute values for an upsampling op (ops.BatchCropAndResize).

  Args:
    input_tensor: image tensor with shape [batch, height, width, in_channels]
    upsample_height: integer
    upsample_width: integer

  Returns:
    grid_centers: tensor with shape [batch, 1]
    crop_sizes: tensor with shape [batch, 1]
    output_height: integer
    output_width: integer
  """
  batch, input_height, input_width, _ = input_tensor.shape

  height_half = input_height / 2.
  width_half = input_width / 2.
  grid_centers = tf.constant(batch * [[height_half, width_half]])
  crop_sizes = tf.constant(batch * [[input_height, input_width]])
  output_height = input_height * upsample_height
  output_width = input_width * upsample_width

  return grid_centers, tf.to_float(crop_sizes), output_height, output_width


def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.

  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]

  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].

  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.

  We create a sum of multiple gaussian kernels each having a width sigma_i.

  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
