# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Cell structure used by NAS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deeplab.core.utils import resize_bilinear
from deeplab.core.utils import scale_dimension

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


class NASBaseCell(object):
  """NASNet Cell class that is used as a 'layer' in image architectures.
  See https://arxiv.org/abs/1707.07012 and https://arxiv.org/abs/1712.00559.

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
    if len(hiddenstate_indices) != len(operations):
      raise ValueError(
          'Number of hiddenstate_indices and operations should be the same.')
    if len(operations) % 2:
      raise ValueError('Number of operations should be even.')
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps

  def __call__(self, net, scope, filter_scaling, stride, prev_layer, cell_num):
    """Runs the conv cell."""
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)

    with tf.variable_scope(scope):
      net = self._cell_base(net, prev_layer)
      for i in range(len(self._operations) // 2):
        with tf.variable_scope('comb_iter_{}'.format(i)):
          h1 = net[self._hiddenstate_indices[i * 2]]
          h2 = net[self._hiddenstate_indices[i * 2 + 1]]
          with tf.variable_scope('left'):
            h1 = self._apply_conv_operation(
                h1, self._operations[i * 2], stride,
                self._hiddenstate_indices[i * 2] < 2)
          with tf.variable_scope('right'):
            h2 = self._apply_conv_operation(
                h2, self._operations[i * 2 + 1], stride,
                self._hiddenstate_indices[i * 2 + 1] < 2)
          with tf.variable_scope('combine'):
            h = h1 + h2
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)

      return net

  def _cell_base(self, net, prev_layer):
    """Runs the beginning of the conv cell before the chosen ops are run."""
    filter_size = self._filter_size

    if prev_layer is None:
      prev_layer = net
    else:
      if net.shape[2] != prev_layer.shape[2]:
        prev_layer = resize_bilinear(
            prev_layer, tf.shape(net)[1:3], prev_layer.dtype)
      if filter_size != prev_layer.shape[3]:
        prev_layer = tf.nn.relu(prev_layer)
        prev_layer = slim.conv2d(prev_layer, filter_size, 1, scope='prev_1x1')
        prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')

    net = tf.nn.relu(net)
    net = slim.conv2d(net, filter_size, 1, scope='1x1')
    net = slim.batch_norm(net, scope='beginning_bn')
    net = tf.split(axis=3, num_or_size_splits=1, value=net)
    net.append(prev_layer)
    return net

  def _apply_conv_operation(self, net, operation, stride,
                            is_from_original_input):
    """Applies the predicted conv operation to net."""
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = net.shape[3]
    filter_size = self._filter_size
    if 'separable' in operation:
      num_layers = int(operation.split('_')[-1])
      kernel_size = int(operation.split('x')[0][-1])
      for layer_num in range(num_layers):
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
    elif 'atrous' in operation:
      kernel_size = int(operation.split('x')[0][-1])
      net = tf.nn.relu(net)
      if stride == 2:
        scaled_height = scale_dimension(tf.shape(net)[1], 0.5)
        scaled_width = scale_dimension(tf.shape(net)[2], 0.5)
        net = resize_bilinear(net, [scaled_height, scaled_width], net.dtype)
        net = slim.conv2d(net, filter_size, kernel_size, rate=1,
                          scope='atrous_{0}x{0}'.format(kernel_size))
      else:
        net = slim.conv2d(net, filter_size, kernel_size, rate=2,
                          scope='atrous_{0}x{0}'.format(kernel_size))
      net = slim.batch_norm(net, scope='bn_atr_{0}x{0}'.format(kernel_size))
    elif operation in ['none']:
      if stride > 1 or (input_filters != filter_size):
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    elif 'pool' in operation:
      pooling_type = operation.split('_')[0]
      pooling_shape = int(operation.split('_')[-1].split('x')[0])
      if pooling_type == 'avg':
        net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      elif pooling_type == 'max':
        net = slim.max_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      else:
        raise ValueError('Unimplemented pooling type: ', pooling_type)
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'none':
      net = self._apply_drop_path(net)
    return net

  def _combine_unused_states(self, net):
    """Concatenates the unused hidden states of the cell."""
    used_hiddenstates = self._used_hiddenstates
    states_to_combine = ([
        h for h, is_used in zip(net, used_hiddenstates) if not is_used])
    net = tf.concat(values=states_to_combine, axis=3)
    return net

  @tf.contrib.framework.add_arg_scope
  def _apply_drop_path(self, net):
    """Apply drop_path regularization."""
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      # Scale keep prob by layer number.
      assert self._cell_num != -1
      layer_ratio = (self._cell_num + 1) / float(self._total_num_cells)
      drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      # Decrease keep prob over time.
      current_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
      current_ratio = tf.minimum(1.0, current_step / self._total_training_steps)
      drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      # Drop path.
      noise_shape = [tf.shape(net)[0], 1, 1, 1]
      random_tensor = drop_path_keep_prob
      random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
      binary_tensor = tf.cast(tf.floor(random_tensor), net.dtype)
      keep_prob_inv = tf.cast(1.0 / drop_path_keep_prob, net.dtype)
      net = net * keep_prob_inv * binary_tensor
    return net
