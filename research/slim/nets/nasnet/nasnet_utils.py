# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A custom module for some common operations used by NASNet.

Functions exposed in this file:
- calc_reduction_layers
- get_channel_index
- get_channel_dim
- global_avg_pool
- factorized_reduction
- drop_path

Classes exposed in this file:
- NasNetABaseCell
- NasNetANormalCell
- NasNetAReductionCell
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
INVALID = 'null'


def calc_reduction_layers(num_cells, num_reduction_layers):
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


@tf.contrib.framework.add_arg_scope
def get_channel_index(data_format=INVALID):
  assert data_format != INVALID
  axis = 3 if data_format == 'NHWC' else 1
  return axis


@tf.contrib.framework.add_arg_scope
def get_channel_dim(shape, data_format=INVALID):
  assert data_format != INVALID
  assert len(shape) == 4
  if data_format == 'NHWC':
    return int(shape[3])
  elif data_format == 'NCHW':
    return int(shape[1])
  else:
    raise ValueError('Not a valid data_format', data_format)


@tf.contrib.framework.add_arg_scope
def global_avg_pool(x, data_format=INVALID):
  """Average pool away the height and width spatial dimensions of x."""
  assert data_format != INVALID
  assert data_format in ['NHWC', 'NCHW']
  assert x.shape.ndims == 4
  if data_format == 'NHWC':
    return tf.reduce_mean(x, [1, 2])
  else:
    return tf.reduce_mean(x, [2, 3])


@tf.contrib.framework.add_arg_scope
def factorized_reduction(net, output_filters, stride, data_format=INVALID):
  """Reduces the shape of net without information loss due to striding."""
  assert output_filters % 2 == 0, (
      'Need even number of filters when using this factorized reduction.')
  assert data_format != INVALID
  if stride == 1:
    net = slim.conv2d(net, output_filters, 1, scope='path_conv')
    net = slim.batch_norm(net, scope='path_bn')
    return net
  if data_format == 'NHWC':
    stride_spec = [1, stride, stride, 1]
  else:
    stride_spec = [1, 1, stride, stride]

  # Skip path 1
  path1 = tf.nn.avg_pool(
      net, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)
  path1 = slim.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')

  # Skip path 2
  # First pad with 0's on the right and bottom, then shift the filter to
  # include those 0's that were added.
  if data_format == 'NHWC':
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
    concat_axis = 3
  else:
    pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
    path2 = tf.pad(net, pad_arr)[:, :, 1:, 1:]
    concat_axis = 1

  path2 = tf.nn.avg_pool(
      path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)
  path2 = slim.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')

  # Concat and apply BN
  final_path = tf.concat(values=[path1, path2], axis=concat_axis)
  final_path = slim.batch_norm(final_path, scope='final_path_bn')
  return final_path


@tf.contrib.framework.add_arg_scope
def drop_path(net, keep_prob, is_training=True):
  """Drops out a whole example hiddenstate with the specified probability."""
  if is_training:
    batch_size = tf.shape(net)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    net = tf.div(net, keep_prob) * binary_tensor
  return net


def _operation_to_filter_shape(operation):
  splitted_operation = operation.split('x')
  filter_shape = int(splitted_operation[0][-1])
  assert filter_shape == int(
      splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split('_')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_info(operation):
  """Takes in operation name and returns meta information.

  An example would be 'separable_3x3_4' -> (3, 4).

  Args:
    operation: String that corresponds to convolution operation.

  Returns:
    Tuple of (filter shape, num layers).
  """
  num_layers = _operation_to_num_layers(operation)
  filter_shape = _operation_to_filter_shape(operation)
  return num_layers, filter_shape


def _stacked_separable_conv(net, stride, operation, filter_size):
  """Takes in an operations and parses it to the correct sep operation."""
  num_layers, kernel_size = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    net = tf.nn.relu(net)
    net = slim.separable_conv2d(
        net,
        filter_size,
        kernel_size,
        depth_multiplier=1,
        scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
        stride=stride)
    net = slim.batch_norm(
        net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
    stride = 1
  net = tf.nn.relu(net)
  net = slim.separable_conv2d(
      net,
      filter_size,
      kernel_size,
      depth_multiplier=1,
      scope='separable_{0}x{0}_{1}'.format(kernel_size, num_layers),
      stride=stride)
  net = slim.batch_norm(
      net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers))
  return net


def _operation_to_pooling_type(operation):
  """Takes in the operation string and returns the pooling type."""
  splitted_operation = operation.split('_')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  """Takes in the operation string and returns the pooling kernel shape."""
  splitted_operation = operation.split('_')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  """Parses the pooling operation string to return its type and shape."""
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def _pooling(net, stride, operation):
  """Parses operation and performs the correct pooling operation on net."""
  padding = 'SAME'
  pooling_type, pooling_shape = _operation_to_pooling_info(operation)
  if pooling_type == 'avg':
    net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding=padding)
  elif pooling_type == 'max':
    net = slim.max_pool2d(net, pooling_shape, stride=stride, padding=padding)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return net


class NasNetABaseCell(object):
  """NASNet Cell class that is used as a 'layer' in image architectures.

  Args:
    num_conv_filters: The number of filters for each convolution operation.
    operations: List of operations that are performed in the NASNet Cell in
      order.
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the NASNet cell.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps

  def _reduce_prev_layer(self, prev_layer, curr_layer):
    """Matches dimension of prev_layer to the curr_layer."""
    # Set the prev layer to the current layer if it is none
    if prev_layer is None:
      return curr_layer
    curr_num_filters = self._filter_size
    prev_num_filters = get_channel_dim(prev_layer.shape)
    curr_filter_shape = int(curr_layer.shape[2])
    prev_filter_shape = int(prev_layer.shape[2])
    if curr_filter_shape != prev_filter_shape:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = factorized_reduction(
          prev_layer, curr_num_filters, stride=2)
    elif curr_num_filters != prev_num_filters:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = slim.conv2d(
          prev_layer, curr_num_filters, 1, scope='prev_1x1')
      prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')
    return prev_layer

  def _cell_base(self, net, prev_layer):
    """Runs the beginning of the conv cell before the predicted ops are run."""
    num_filters = self._filter_size

    # Check to be sure prev layer stuff is setup correctly
    prev_layer = self._reduce_prev_layer(prev_layer, net)

    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, 1, scope='1x1')
    net = slim.batch_norm(net, scope='beginning_bn')
    split_axis = get_channel_index()
    net = tf.split(axis=split_axis, num_or_size_splits=1, value=net)
    for split in net:
      assert int(split.shape[split_axis] == int(self._num_conv_filters *
                                                self._filter_scaling))
    net.append(prev_layer)
    return net

  def __call__(self, net, scope=None, filter_scaling=1, stride=1,
               prev_layer=None, cell_num=-1):
    """Runs the conv cell."""
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)

    i = 0
    with tf.variable_scope(scope):
      net = self._cell_base(net, prev_layer)
      for iteration in range(5):
        with tf.variable_scope('comb_iter_{}'.format(iteration)):
          left_hiddenstate_idx, right_hiddenstate_idx = (
              self._hiddenstate_indices[i],
              self._hiddenstate_indices[i + 1])
          original_input_left = left_hiddenstate_idx < 2
          original_input_right = right_hiddenstate_idx < 2
          h1 = net[left_hiddenstate_idx]
          h2 = net[right_hiddenstate_idx]

          operation_left = self._operations[i]
          operation_right = self._operations[i+1]
          i += 2
          # Apply conv operations
          with tf.variable_scope('left'):
            h1 = self._apply_conv_operation(h1, operation_left,
                                            stride, original_input_left)
          with tf.variable_scope('right'):
            h2 = self._apply_conv_operation(h2, operation_right,
                                            stride, original_input_right)

          # Combine hidden states using 'add'.
          with tf.variable_scope('combine'):
            h = h1 + h2

          # Add hiddenstate to the list of hiddenstates we can choose from
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)

      return net

  def _apply_conv_operation(self, net, operation,
                            stride, is_from_original_input):
    """Applies the predicted conv operation to net."""
    # Dont stride if this is not one of the original hiddenstates
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = get_channel_dim(net.shape)
    filter_size = self._filter_size
    if 'separable' in operation:
      net = _stacked_separable_conv(net, stride, operation, filter_size)
    elif operation in ['none']:
      # Check if a stride is needed, then use a strided 1x1 here
      if stride > 1 or (input_filters != filter_size):
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    elif 'pool' in operation:
      net = _pooling(net, stride, operation)
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'none':
      net = self._apply_drop_path(net)
    return net

  def _combine_unused_states(self, net):
    """Concatenate the unused hidden states of the cell."""
    used_hiddenstates = self._used_hiddenstates

    final_height = int(net[-1].shape[2])
    final_num_filters = get_channel_dim(net[-1].shape)
    assert len(used_hiddenstates) == len(net)
    for idx, used_h in enumerate(used_hiddenstates):
      curr_height = int(net[idx].shape[2])
      curr_num_filters = get_channel_dim(net[idx].shape)

      # Determine if a reduction should be applied to make the number of
      # filters match.
      should_reduce = final_num_filters != curr_num_filters
      should_reduce = (final_height != curr_height) or should_reduce
      should_reduce = should_reduce and not used_h
      if should_reduce:
        stride = 2 if final_height != curr_height else 1
        with tf.variable_scope('reduction_{}'.format(idx)):
          net[idx] = factorized_reduction(
              net[idx], final_num_filters, stride)

    states_to_combine = (
        [h for h, is_used in zip(net, used_hiddenstates) if not is_used])

    # Return the concat of all the states
    concat_axis = get_channel_index()
    net = tf.concat(values=states_to_combine, axis=concat_axis)
    return net

  @tf.contrib.framework.add_arg_scope  # No public API. For internal use only.
  def _apply_drop_path(self, net, current_step=None,
                       use_summaries=False, drop_connect_version='v3'):
    """Apply drop_path regularization.

    Args:
      net: the Tensor that gets drop_path regularization applied.
      current_step: a float32 Tensor with the current global_step value,
        to be divided by hparams.total_training_steps. Usually None, which
        defaults to tf.train.get_or_create_global_step() properly casted.
      use_summaries: a Python boolean. If set to False, no summaries are output.
      drop_connect_version: one of 'v1', 'v2', 'v3', controlling whether
        the dropout rate is scaled by current_step (v1), layer (v2), or
        both (v3, the default).

    Returns:
      The dropped-out value of `net`.
    """
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert self._cell_num != -1
        # The added 2 is for the reduction cells
        num_cells = self._total_num_cells
        layer_ratio = (self._cell_num + 1)/float(num_cells)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('layer_ratio', layer_ratio)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        if not current_step:
          current_step = tf.cast(tf.train.get_or_create_global_step(),
                                 tf.float32)
        drop_path_burn_in_steps = self._total_training_steps
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('current_ratio', current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      if use_summaries:
        with tf.device('/cpu:0'):
          tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
      net = drop_path(net, drop_path_keep_prob)
    return net


class NasNetANormalCell(NasNetABaseCell):
  """NASNetA Normal Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    operations = ['separable_5x5_2',
                  'separable_3x3_2',
                  'separable_5x5_2',
                  'separable_3x3_2',
                  'avg_pool_3x3',
                  'none',
                  'avg_pool_3x3',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'none']
    used_hiddenstates = [1, 0, 0, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    super(NasNetANormalCell, self).__init__(num_conv_filters, operations,
                                            used_hiddenstates,
                                            hiddenstate_indices,
                                            drop_path_keep_prob,
                                            total_num_cells,
                                            total_training_steps)


class NasNetAReductionCell(NasNetABaseCell):
  """NASNetA Reduction Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    operations = ['separable_5x5_2',
                  'separable_7x7_2',
                  'max_pool_3x3',
                  'separable_7x7_2',
                  'avg_pool_3x3',
                  'separable_5x5_2',
                  'none',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'max_pool_3x3']
    used_hiddenstates = [1, 1, 1, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 0, 1, 0, 1, 3, 2, 2, 0]
    super(NasNetAReductionCell, self).__init__(num_conv_filters, operations,
                                               used_hiddenstates,
                                               hiddenstate_indices,
                                               drop_path_keep_prob,
                                               total_num_cells,
                                               total_training_steps)
