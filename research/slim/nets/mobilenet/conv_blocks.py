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
"""Convolution blocks for mobilenet."""
import contextlib
import functools

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


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
  return new_v


def _split_divisible(num, num_ways, divisible_by=8):
  """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
  assert num % divisible_by == 0
  assert num / num_ways >= divisible_by
  # Note: want to round down, we adjust each split to match the total.
  base = num // num_ways // divisible_by * divisible_by
  result = []
  accumulated = 0
  for i in range(num_ways):
    r = base
    while accumulated + r < num * (i + 1) / num_ways:
      r += divisible_by
    result.append(r)
    accumulated += r
  assert accumulated == num
  return result


@contextlib.contextmanager
def _v1_compatible_scope_naming(scope):
  """v1 compatible scope naming."""
  if scope is None:  # Create uniqified separable blocks.
    with tf.compat.v1.variable_scope(None, default_name='separable') as s, \
         tf.compat.v1.name_scope(s.original_name_scope):
      yield ''
  else:
    # We use scope_depthwise, scope_pointwise for compatibility with V1 ckpts.
    # which provide numbered scopes.
    scope += '_'
    yield scope


@slim.add_arg_scope
def split_separable_conv2d(input_tensor,
                           num_outputs,
                           scope=None,
                           normalizer_fn=None,
                           stride=1,
                           rate=1,
                           endpoints=None,
                           use_explicit_padding=False):
  """Separable mobilenet V1 style convolution.

  Depthwise convolution, with default non-linearity,
  followed by 1x1 depthwise convolution.  This is similar to
  slim.separable_conv2d, but differs in tha it applies batch
  normalization and non-linearity to depthwise. This  matches
  the basic building of Mobilenet Paper
  (https://arxiv.org/abs/1704.04861)

  Args:
    input_tensor: input
    num_outputs: number of outputs
    scope: optional name of the scope. Note if provided it will use
    scope_depthwise for deptwhise, and scope_pointwise for pointwise.
    normalizer_fn: which normalizer function to use for depthwise/pointwise
    stride: stride
    rate: output rate (also known as dilation rate)
    endpoints: optional, if provided, will export additional tensors to it.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.

  Returns:
    output tesnor
  """

  with _v1_compatible_scope_naming(scope) as scope:
    dw_scope = scope + 'depthwise'
    endpoints = endpoints if endpoints is not None else {}
    kernel_size = [3, 3]
    padding = 'SAME'
    if use_explicit_padding:
      padding = 'VALID'
      input_tensor = _fixed_padding(input_tensor, kernel_size, rate)
    net = slim.separable_conv2d(
        input_tensor,
        None,
        kernel_size,
        depth_multiplier=1,
        stride=stride,
        rate=rate,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope=dw_scope)

    endpoints[dw_scope] = net

    pw_scope = scope + 'pointwise'
    net = slim.conv2d(
        net,
        num_outputs, [1, 1],
        stride=1,
        normalizer_fn=normalizer_fn,
        scope=pw_scope)
    endpoints[pw_scope] = net
  return net


def expand_input_by_factor(n, divisible_by=8):
  return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def split_conv(input_tensor,
               num_outputs,
               num_ways,
               scope,
               divisible_by=8,
               **kwargs):
  """Creates a split convolution.

  Split convolution splits the input and output into
  'num_blocks' blocks of approximately the same size each,
  and only connects $i$-th input to $i$ output.

  Args:
    input_tensor: input tensor
    num_outputs: number of output filters
    num_ways: num blocks to split by.
    scope: scope for all the operators.
    divisible_by: make sure that every part is divisiable by this.
    **kwargs: will be passed directly into conv2d operator
  Returns:
    tensor
  """
  b = input_tensor.get_shape().as_list()[3]

  if num_ways == 1 or min(b // num_ways,
                          num_outputs // num_ways) < divisible_by:
    # Don't do any splitting if we end up with less than 8 filters
    # on either side.
    return slim.conv2d(input_tensor, num_outputs, [1, 1], scope=scope, **kwargs)

  outs = []
  input_splits = _split_divisible(b, num_ways, divisible_by=divisible_by)
  output_splits = _split_divisible(
      num_outputs, num_ways, divisible_by=divisible_by)
  inputs = tf.split(input_tensor, input_splits, axis=3, name='split_' + scope)
  base = scope
  for i, (input_tensor, out_size) in enumerate(zip(inputs, output_splits)):
    scope = base + '_part_%d' % (i,)
    n = slim.conv2d(input_tensor, out_size, [1, 1], scope=scope, **kwargs)
    n = tf.identity(n, scope + '_output')
    outs.append(n)
  return tf.concat(outs, 3, name=scope + '_concat')


@slim.add_arg_scope
def expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  split_projection=1,
                  split_expansion=1,
                  split_divisible_by=8,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  inner_activation_fn=None,
                  depthwise_activation_fn=None,
                  project_activation_fn=tf.identity,
                  depthwise_fn=slim.separable_conv2d,
                  expansion_fn=split_conv,
                  projection_fn=split_conv,
                  scope=None):
  """Depthwise Convolution Block with expansion.

  Builds a composite convolution that has the following structure
  expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

  Args:
    input_tensor: input
    num_outputs: number of outputs in the final layer.
    expansion_size: the size of expansion, could be a constant or a callable.
      If latter it will be provided 'num_inputs' as an input. For forward
      compatibility it should accept arbitrary keyword arguments.
      Default will expand the input by factor of 6.
    stride: depthwise stride
    rate: depthwise rate
    kernel_size: depthwise kernel
    residual: whether to include residual connection between input
      and output.
    normalizer_fn: batchnorm or otherwise
    split_projection: how many ways to split projection operator
      (that is conv expansion->bottleneck)
    split_expansion: how many ways to split expansion op
      (that is conv bottleneck->expansion) ops will keep depth divisible
      by this value.
    split_divisible_by: make sure every split group is divisible by this number.
    expansion_transform: Optional function that takes expansion
      as a single input and returns output.
    depthwise_location: where to put depthwise covnvolutions supported
      values None, 'input', 'output', 'expansion'
    depthwise_channel_multiplier: depthwise channel multiplier:
    each input will replicated (with different filters)
    that many times. So if input had c channels,
    output will have c x depthwise_channel_multpilier.
    endpoints: An optional dictionary into which intermediate endpoints are
      placed. The keys "expansion_output", "depthwise_output",
      "projection_output" and "expansion_transform" are always populated, even
      if the corresponding functions are not invoked.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    padding: Padding type to use if `use_explicit_padding` is not set.
    inner_activation_fn: activation function to use in all inner convolutions.
    If none, will rely on slim default scopes.
    depthwise_activation_fn: activation function to use for deptwhise only.
      If not provided will rely on slim default scopes. If both
      inner_activation_fn and depthwise_activation_fn are provided,
      depthwise_activation_fn takes precedence over inner_activation_fn.
    project_activation_fn: activation function for the project layer.
    (note this layer is not affected by inner_activation_fn)
    depthwise_fn: Depthwise convolution function.
    expansion_fn: Expansion convolution function. If use custom function then
      "split_expansion" and "split_divisible_by" will be ignored.
    projection_fn: Projection convolution function. If use custom function then
      "split_projection" and "split_divisible_by" will be ignored.

    scope: optional scope.

  Returns:
    Tensor of depth num_outputs

  Raises:
    TypeError: on inval
  """
  conv_defaults = {}
  dw_defaults = {}
  if inner_activation_fn is not None:
    conv_defaults['activation_fn'] = inner_activation_fn
    dw_defaults['activation_fn'] = inner_activation_fn
  if depthwise_activation_fn is not None:
    dw_defaults['activation_fn'] = depthwise_activation_fn
  # pylint: disable=g-backslash-continuation
  with tf.compat.v1.variable_scope(scope, default_name='expanded_conv') as s, \
       tf.compat.v1.name_scope(s.original_name_scope), \
      slim.arg_scope((slim.conv2d,), **conv_defaults), \
       slim.arg_scope((slim.separable_conv2d,), **dw_defaults):
    prev_depth = input_tensor.get_shape().as_list()[3]
    if  depthwise_location not in [None, 'input', 'output', 'expansion']:
      raise TypeError('%r is unknown value for depthwise_location' %
                      depthwise_location)
    if use_explicit_padding:
      if padding != 'SAME':
        raise TypeError('`use_explicit_padding` should only be used with '
                        '"SAME" padding.')
      padding = 'VALID'
    depthwise_func = functools.partial(
        depthwise_fn,
        num_outputs=None,
        kernel_size=kernel_size,
        depth_multiplier=depthwise_channel_multiplier,
        stride=stride,
        rate=rate,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope='depthwise')
    # b1 -> b2 * r -> b2
    #   i -> (o * r) (bottleneck) -> o
    input_tensor = tf.identity(input_tensor, 'input')
    net = input_tensor

    if depthwise_location == 'input':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net, activation_fn=None)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if callable(expansion_size):
      inner_size = expansion_size(num_inputs=prev_depth)
    else:
      inner_size = expansion_size

    if inner_size > net.shape[3]:
      if expansion_fn == split_conv:
        expansion_fn = functools.partial(
            expansion_fn,
            num_ways=split_expansion,
            divisible_by=split_divisible_by,
            stride=1)
      net = expansion_fn(
          net,
          inner_size,
          scope='expand',
          normalizer_fn=normalizer_fn)
      net = tf.identity(net, 'expansion_output')
      if endpoints is not None:
        endpoints['expansion_output'] = net

    if depthwise_location == 'expansion':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if expansion_transform:
      net = expansion_transform(expansion_tensor=net, input_tensor=input_tensor)
    # Note in contrast with expansion, we always have
    # projection to produce the desired output size.
    if projection_fn == split_conv:
      projection_fn = functools.partial(
          projection_fn,
          num_ways=split_projection,
          divisible_by=split_divisible_by,
          stride=1)
    net = projection_fn(
        net,
        num_outputs,
        scope='project',
        normalizer_fn=normalizer_fn,
        activation_fn=project_activation_fn)
    if endpoints is not None:
      endpoints['projection_output'] = net
    if depthwise_location == 'output':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net, activation_fn=None)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if callable(residual):  # custom residual
      net = residual(input_tensor=input_tensor, output_tensor=net)
    elif (residual and
          # stride check enforces that we don't add residuals when spatial
          # dimensions are None
          stride == 1 and
          # Depth matches
          net.get_shape().as_list()[3] ==
          input_tensor.get_shape().as_list()[3]):
      net += input_tensor
    return tf.identity(net, name='output')


@slim.add_arg_scope
def squeeze_excite(input_tensor,
                   divisible_by=8,
                   squeeze_factor=3,
                   inner_activation_fn=tf.nn.relu,
                   gating_fn=tf.sigmoid,
                   squeeze_input_tensor=None,
                   pool=None):
  """Squeeze excite block for Mobilenet V3.

  If the squeeze_input_tensor - or the input_tensor if squeeze_input_tensor is
  None - contains variable dimensions (Nonetype in tensor shape), perform
  average pooling (as the first step in the squeeze operation) by calling
  reduce_mean across the H/W of the input tensor.

  Args:
    input_tensor: input tensor to apply SE block to.
    divisible_by: ensures all inner dimensions are divisible by this number.
    squeeze_factor: the factor of squeezing in the inner fully connected layer
    inner_activation_fn: non-linearity to be used in inner layer.
    gating_fn: non-linearity to be used for final gating function
    squeeze_input_tensor: custom tensor to use for computing gating activation.
     If provided the result will be input_tensor * SE(squeeze_input_tensor)
     instead of input_tensor * SE(input_tensor).
    pool: if number is  provided will average pool with that kernel size
      to compute inner tensor, followed by bilinear upsampling.

  Returns:
    Gated input_tensor. (e.g. X * SE(X))
  """
  with tf.compat.v1.variable_scope('squeeze_excite'):
    if squeeze_input_tensor is None:
      squeeze_input_tensor = input_tensor
    input_size = input_tensor.shape.as_list()[1:3]
    pool_height, pool_width = squeeze_input_tensor.shape.as_list()[1:3]
    stride = 1
    if pool is not None and pool_height >= pool:
      pool_height, pool_width, stride = pool, pool, pool
    input_channels = squeeze_input_tensor.shape.as_list()[3]
    output_channels = input_tensor.shape.as_list()[3]
    squeeze_channels = _make_divisible(
        input_channels / squeeze_factor, divisor=divisible_by)

    if pool is None:
      pooled = tf.reduce_mean(squeeze_input_tensor, axis=[1, 2], keepdims=True)
    else:
      pooled = tf.nn.avg_pool(
          squeeze_input_tensor, (1, pool_height, pool_width, 1),
          strides=(1, stride, stride, 1),
          padding='VALID')
    squeeze = slim.conv2d(
        pooled,
        kernel_size=(1, 1),
        num_outputs=squeeze_channels,
        normalizer_fn=None,
        activation_fn=inner_activation_fn)
    excite_outputs = output_channels
    excite = slim.conv2d(squeeze, num_outputs=excite_outputs,
                         kernel_size=[1, 1],
                         normalizer_fn=None,
                         activation_fn=gating_fn)
    if pool is not None:
      # Note: As of 03/20/2019 only BILINEAR (the default) with
      # align_corners=True has gradients implemented in TPU.
      excite = tf.image.resize_images(
          excite, input_size,
          align_corners=True)
    result = input_tensor * excite
  return result
