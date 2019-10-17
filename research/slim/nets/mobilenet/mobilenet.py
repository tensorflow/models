# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Mobilenet Base Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import contextlib
import copy
import os

import tensorflow as tf


slim = tf.contrib.slim


@slim.add_arg_scope
def apply_activation(x, name=None, activation_fn=None):
  return activation_fn(x, name=name) if activation_fn else x


def _fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                  [pad_beg[1], pad_end[1]], [0, 0]])
  return padded_inputs


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return int(new_v)


@contextlib.contextmanager
def _set_arg_scope_defaults(defaults):
  """Sets arg scope defaults for all items present in defaults.

  Args:
    defaults: dictionary/list of pairs, containing a mapping from
    function to a dictionary of default args.

  Yields:
    context manager where all defaults are set.
  """
  if hasattr(defaults, 'items'):
    items = list(defaults.items())
  else:
    items = defaults
  if not items:
    yield
  else:
    func, default_arg = items[0]
    with slim.arg_scope(func, **default_arg):
      with _set_arg_scope_defaults(items[1:]):
        yield


@slim.add_arg_scope
def depth_multiplier(output_params,
                     multiplier,
                     divisible_by=8,
                     min_depth=8,
                     **unused_kwargs):
  if 'num_outputs' not in output_params:
    return
  d = output_params['num_outputs']
  output_params['num_outputs'] = _make_divisible(d * multiplier, divisible_by,
                                                 min_depth)


_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])


def op(opfunc, multiplier_func=depth_multiplier, **params):
  multiplier = params.pop('multiplier_transform', multiplier_func)
  return _Op(opfunc, params=params, multiplier_func=multiplier)


class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def safe_arg_scope(funcs, **kwargs):
  """Returns `slim.arg_scope` with all None arguments removed.

  Arguments:
    funcs: Functions to pass to `arg_scope`.
    **kwargs: Arguments to pass to `arg_scope`.

  Returns:
    arg_scope or No-op context manager.

  Note: can be useful if None value should be interpreted as "do not overwrite
    this parameter value".
  """
  filtered_args = {name: value for name, value in kwargs.items()
                   if value is not None}
  if filtered_args:
    return slim.arg_scope(funcs, **filtered_args)
  else:
    return NoOpScope()


@slim.add_arg_scope
def mobilenet_base(  # pylint: disable=invalid-name
    inputs,
    conv_defs,
    multiplier=1.0,
    final_endpoint=None,
    output_stride=None,
    use_explicit_padding=False,
    scope=None,
    is_training=False):
  """Mobilenet base network.

  Constructs a network from inputs to the given final endpoint. By default
  the network is constructed in inference mode. To create network
  in training mode use:

  with slim.arg_scope(mobilenet.training_scope()):
     logits, endpoints = mobilenet_base(...)

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    conv_defs: A list of op(...) layers specifying the net architecture.
    multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    final_endpoint: The name of last layer, for early termination for
    for V1-based networks: last layer is "layer_14", for V2: "layer_20"
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 1 or any even number, excluding
      zero. Typical values are 8 (accurate fully convolutional mode), 16
      (fast fully convolutional mode), and 32 (classification mode).

      NOTE- output_stride relies on all consequent operators to support dilated
      operators via "rate" parameter. This might require wrapping non-conv
      operators to operate properly.

    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    scope: optional variable scope.
    is_training: How to setup batch_norm and other ops. Note: most of the time
      this does not need be set directly. Use mobilenet.training_scope() to set
      up training instead. This parameter is here for backward compatibility
      only. It is safe to set it to the value matching
      training_scope(is_training=...). It is also safe to explicitly set
      it to False, even if there is outer training_scope set to to training.
      (The network will be built in inference mode). If this is set to None,
      no arg_scope is added for slim.batch_norm's is_training parameter.

  Returns:
    tensor_out: output tensor.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  if multiplier <= 0:
    raise ValueError('multiplier is not greater than zero.')

  # Set conv defs defaults and overrides.
  conv_defs_defaults = conv_defs.get('defaults', {})
  conv_defs_overrides = conv_defs.get('overrides', {})
  if use_explicit_padding:
    conv_defs_overrides = copy.deepcopy(conv_defs_overrides)
    conv_defs_overrides[
        (slim.conv2d, slim.separable_conv2d)] = {'padding': 'VALID'}

  if output_stride is not None:
    if output_stride == 0 or (output_stride > 1 and output_stride % 2):
      raise ValueError('Output stride must be None, 1 or a multiple of 2.')

  # a) Set the tensorflow scope
  # b) set padding to default: note we might consider removing this
  # since it is also set by mobilenet_scope
  # c) set all defaults
  # d) set all extra overrides.
  # pylint: disable=g-backslash-continuation
  with _scope_all(scope, default_scope='Mobilenet'), \
      safe_arg_scope([slim.batch_norm], is_training=is_training), \
      _set_arg_scope_defaults(conv_defs_defaults), \
      _set_arg_scope_defaults(conv_defs_overrides):
    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    # Insert default parameters before the base scope which includes
    # any custom overrides set in mobilenet.
    end_points = {}
    scopes = {}
    for i, opdef in enumerate(conv_defs['spec']):
      params = dict(opdef.params)
      opdef.multiplier_func(params, multiplier)
      stride = params.get('stride', 1)
      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= stride
      else:
        layer_stride = stride
        layer_rate = 1
        current_stride *= stride
      # Update params.
      params['stride'] = layer_stride
      # Only insert rate to params if rate > 1 and kernel size is not [1, 1].
      if layer_rate > 1:
        if tuple(params.get('kernel_size', [])) != (1, 1):
          # We will apply atrous rate in the following cases:
          # 1) When kernel_size is not in params, the operation then uses
          #   default kernel size 3x3.
          # 2) When kernel_size is in params, and if the kernel_size is not
          #   equal to (1, 1) (there is no need to apply atrous convolution to
          #   any 1x1 convolution).
          params['rate'] = layer_rate
      # Set padding
      if use_explicit_padding:
        if 'kernel_size' in params:
          net = _fixed_padding(net, params['kernel_size'], layer_rate)
        else:
          params['use_explicit_padding'] = True

      end_point = 'layer_%d' % (i + 1)
      try:
        net = opdef.op(net, **params)
      except Exception:
        print('Failed to create op %i: %r params: %r' % (i, opdef, params))
        raise
      end_points[end_point] = net
      scope = os.path.dirname(net.name)
      scopes[scope] = end_point
      if final_endpoint is not None and end_point == final_endpoint:
        break

    # Add all tensors that end with 'output' to
    # endpoints
    for t in net.graph.get_operations():
      scope = os.path.dirname(t.name)
      bn = os.path.basename(t.name)
      if scope in scopes and t.name.endswith('output'):
        end_points[scopes[scope] + '/' + bn] = t.outputs[0]
    return net, end_points


@contextlib.contextmanager
def _scope_all(scope, default_scope=None):
  with tf.variable_scope(scope, default_name=default_scope) as s,\
       tf.name_scope(s.original_name_scope):
    yield s


@slim.add_arg_scope
def mobilenet(inputs,
              num_classes=1001,
              prediction_fn=slim.softmax,
              reuse=None,
              scope='Mobilenet',
              base_only=False,
              **mobilenet_args):
  """Mobilenet model for classification, supports both V1 and V2.

  Note: default mode is inference, use mobilenet.training_scope to create
  training network.


  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    prediction_fn: a function to get predictions out of logits
      (default softmax).
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    base_only: if True will only create the base of the network (no pooling
    and no logits).
    **mobilenet_args: passed to mobilenet_base verbatim.
      - conv_defs: list of conv defs
      - multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
      - output_stride: will ensure that the last layer has at most total stride.
      If the architecture calls for more stride than that provided
      (e.g. output_stride=16, but the architecture has 5 stride=2 operators),
      it will replace output_stride with fractional convolutions using Atrous
      Convolutions.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation tensor.

  Raises:
    ValueError: Input rank is invalid.
  """
  is_training = mobilenet_args.get('is_training', False)
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

  with tf.variable_scope(scope, 'Mobilenet', reuse=reuse) as scope:
    inputs = tf.identity(inputs, 'input')
    net, end_points = mobilenet_base(inputs, scope=scope, **mobilenet_args)
    if base_only:
      return net, end_points

    net = tf.identity(net, name='embedding')

    with tf.variable_scope('Logits'):
      net = global_pool(net)
      end_points['global_pool'] = net
      if not num_classes:
        return net, end_points
      net = slim.dropout(net, scope='Dropout', is_training=is_training)
      # 1 x 1 x num_classes
      # Note: legacy scope name.
      logits = slim.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          biases_initializer=tf.zeros_initializer(),
          scope='Conv2d_1c_1x1')

      logits = tf.squeeze(logits, [1, 2])

      logits = tf.identity(logits, name='output')
    end_points['Logits'] = logits
    if prediction_fn:
      end_points['Predictions'] = prediction_fn(logits, 'Predictions')
  return logits, end_points


def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
  """Applies avg pool to produce 1x1 output.

  NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
  baked in average pool which has better support across hardware.

  Args:
    input_tensor: input tensor
    pool_op: pooling op (avg pool is default)
  Returns:
    a tensor batch_size x 1 x 1 x depth.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size = tf.convert_to_tensor(
        [1, tf.shape(input_tensor)[1],
         tf.shape(input_tensor)[2], 1])
  else:
    kernel_size = [1, shape[1], shape[2], 1]
  output = pool_op(
      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
  # Recover output shape, for unknown shape.
  output.set_shape([None, 1, 1, None])
  return output


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):
  """Defines Mobilenet training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
  Args:
    is_training: if set to False this will ensure that all customizations are
      set to non-training mode. This might be helpful for code that is reused
      across both training/evaluation, but most of the time training_scope with
      value False is not needed. If this is set to None, the parameters is not
      added to the batch_norm arg_scope.

    weight_decay: The weight decay to use for regularizing the model.
    stddev: Standard deviation for initialization, if negative uses xavier.
    dropout_keep_prob: dropout keep probability (not set if equals to None).
    bn_decay: decay for the batch norm moving averages (not set if equals to
      None).

  Returns:
    An argument scope to use via arg_scope.
  """
  # Note: do not introduce parameters that would change the inference
  # model here (for example whether to use bias), modify conv_def instead.
  batch_norm_params = {
      'decay': bn_decay,
      'is_training': is_training
  }
  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      normalizer_fn=slim.batch_norm), \
      slim.arg_scope([mobilenet_base, mobilenet], is_training=is_training),\
      safe_arg_scope([slim.batch_norm], **batch_norm_params), \
      safe_arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
    return s
