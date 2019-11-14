# Lint as: python2, python3
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

r"""Xception model.

"Xception: Deep Learning with Depthwise Separable Convolutions"
Fran{\c{c}}ois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017
detection challenge submission, where the model is made deeper and has aligned
features for dense prediction tasks. See their slides for details:

"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge
2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop
http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam
https://arxiv.org/abs/1704.04861
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from deeplab.core import utils
from tensorflow.contrib.slim.nets import resnet_utils
from nets.mobilenet import conv_blocks as mobilenet_v3_ops

slim = contrib_slim


_DEFAULT_MULTI_GRID = [1, 1, 1]
# The cap for tf.clip_by_value.
_CLIP_CAP = 6


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


@slim.add_arg_scope
def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):
  """Strided 2-D separable convolution with 'SAME' padding.

  If stride > 1 and use_explicit_padding is True, then we do explicit zero-
  padding, followed by conv2d with 'VALID' padding.

  Note that

     net = separable_conv2d_same(inputs, num_outputs, 3,
       depth_multiplier=1, stride=stride)

  is equivalent to

     net = slim.separable_conv2d(inputs, num_outputs, 3,
       depth_multiplier=1, stride=1, padding='SAME')
     net = resnet_utils.subsample(net, factor=stride)

  whereas

     net = slim.separable_conv2d(inputs, num_outputs, 3, stride=stride,
       depth_multiplier=1, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function.

  Consequently, if the input feature map has even height or width, setting
  `use_explicit_padding=False` will result in feature misalignment by one pixel
  along the corresponding dimension.

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    use_explicit_padding: If True, use explicit padding to make the model fully
      compatible with the open source version, otherwise use the native
      Tensorflow 'SAME' padding.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    scope: Scope.
    **kwargs: additional keyword arguments to pass to slim.conv2d

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  def _separable_conv2d(padding):
    """Wrapper for separable conv2d."""
    return slim.separable_conv2d(inputs,
                                 num_outputs,
                                 kernel_size,
                                 depth_multiplier=depth_multiplier,
                                 stride=stride,
                                 rate=rate,
                                 padding=padding,
                                 scope=scope,
                                 **kwargs)
  def _split_separable_conv2d(padding):
    """Splits separable conv2d into depthwise and pointwise conv2d."""
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size,
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    rate=rate,
                                    padding=padding,
                                    scope=scope + '_depthwise',
                                    **kwargs)
    return slim.conv2d(outputs,
                       num_outputs,
                       1,
                       scope=scope + '_pointwise',
                       **kwargs)
  if stride == 1 or not use_explicit_padding:
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='SAME')
    else:
      outputs = _split_separable_conv2d(padding='SAME')
  else:
    inputs = fixed_padding(inputs, kernel_size, rate)
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='VALID')
    else:
      outputs = _split_separable_conv2d(padding='VALID')
  return outputs


@slim.add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    kernel_size=3,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None,
                    use_bounded_activation=False,
                    use_explicit_padding=True,
                    use_squeeze_excite=False,
                    se_pool_size=None):
  """An Xception module.

  The output of one Xception module is equal to the sum of `residual` and
  `shortcut`, where `residual` is the feature computed by three separable
  convolution. The `shortcut` is the feature computed by 1x1 convolution with
  or without striding. In some cases, the `shortcut` path could be a simple
  identity function or none (i.e, no shortcut).

  Note that we replace the max pooling operations in the Xception module with
  another separable convolution with striding, since atrous rate is not properly
  supported in current TensorFlow max pooling implementation.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth_list: A list of three integers specifying the depth values of one
      Xception module.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    stride: The block unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    kernel_size: Integer, convolution kernel size.
    unit_rate_list: A list of three integers, determining the unit rate for
      each separable convolution in the xception module.
    rate: An integer, rate for atrous convolution.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    outputs_collections: Collection to add the Xception unit output.
    scope: Optional variable_scope.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
    use_explicit_padding: If True, use explicit padding to make the model fully
      compatible with the open source version, otherwise use the native
      Tensorflow 'SAME' padding.
    use_squeeze_excite: Boolean, use squeeze-and-excitation or not.
    se_pool_size: None or integer specifying the pooling size used in SE module.

  Returns:
    The Xception module's output.

  Raises:
    ValueError: If depth_list and unit_rate_list do not contain three elements,
      or if stride != 1 for the third separable convolution operation in the
      residual path, or unsupported skip connection type.
  """
  if len(depth_list) != 3:
    raise ValueError('Expect three elements in depth_list.')
  if unit_rate_list:
    if len(unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  with tf.variable_scope(scope, 'xception_module', [inputs]) as sc:
    residual = inputs

    def _separable_conv(features, depth, kernel_size, depth_multiplier,
                        regularize_depthwise, rate, stride, scope):
      """Separable conv block."""
      if activation_fn_in_separable_conv:
        activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
      else:
        if use_bounded_activation:
          # When use_bounded_activation is True, we clip the feature values and
          # apply relu6 for activation.
          activation_fn = lambda x: tf.clip_by_value(x, -_CLIP_CAP, _CLIP_CAP)
          features = tf.nn.relu6(features)
        else:
          # Original network design.
          activation_fn = None
          features = tf.nn.relu(features)
      return separable_conv2d_same(features,
                                   depth,
                                   kernel_size,
                                   depth_multiplier=depth_multiplier,
                                   stride=stride,
                                   rate=rate,
                                   activation_fn=activation_fn,
                                   use_explicit_padding=use_explicit_padding,
                                   regularize_depthwise=regularize_depthwise,
                                   scope=scope)
    for i in range(3):
      residual = _separable_conv(residual,
                                 depth_list[i],
                                 kernel_size=kernel_size,
                                 depth_multiplier=1,
                                 regularize_depthwise=regularize_depthwise,
                                 rate=rate*unit_rate_list[i],
                                 stride=stride if i == 2 else 1,
                                 scope='separable_conv' + str(i+1))
    if use_squeeze_excite:
      residual = mobilenet_v3_ops.squeeze_excite(
          input_tensor=residual,
          squeeze_factor=16,
          inner_activation_fn=tf.nn.relu,
          gating_fn=lambda x: tf.nn.relu6(x+3)*0.16667,
          pool=se_pool_size)

    if skip_connection_type == 'conv':
      shortcut = slim.conv2d(inputs,
                             depth_list[-1],
                             [1, 1],
                             stride=stride,
                             activation_fn=None,
                             scope='shortcut')
      if use_bounded_activation:
        residual = tf.clip_by_value(residual, -_CLIP_CAP, _CLIP_CAP)
        shortcut = tf.clip_by_value(shortcut, -_CLIP_CAP, _CLIP_CAP)
      outputs = residual + shortcut
      if use_bounded_activation:
        outputs = tf.nn.relu6(outputs)
    elif skip_connection_type == 'sum':
      if use_bounded_activation:
        residual = tf.clip_by_value(residual, -_CLIP_CAP, _CLIP_CAP)
        inputs = tf.clip_by_value(inputs, -_CLIP_CAP, _CLIP_CAP)
      outputs = residual + inputs
      if use_bounded_activation:
        outputs = tf.nn.relu6(outputs)
    elif skip_connection_type == 'none':
      outputs = residual
    else:
      raise ValueError('Unsupported skip connection type.')

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            outputs)


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
  """Stacks Xception blocks and controls output feature density.

  First, this function creates scopes for the Xception in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the output
  stride, which is the ratio of the input to output spatial resolution. This
  is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A tensor of size [batch, height, width, channels].
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of Xception.
      For example, if the Xception employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the Xception block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)
          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)

      # Collect activations at the block's end before performing subsampling.
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None,
             sync_batch_norm_method='None'):
  """Generator for Xception models.

  This function generates a family of Xception models. See the xception_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce Xception of various depths.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training (see go/slim-classification-models for
      specifics).
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    keep_prob: Keep probability used in the pre-logits dropout layer.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    sync_batch_norm_method: String, sync batchnorm method. Currently only
      support `None`.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last Xception block, potentially after
      global average pooling. If num_classes is a non-zero integer, net contains
      the pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(
      scope, 'xception', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points'
    batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
    with slim.arg_scope([slim.conv2d,
                         slim.separable_conv2d,
                         xception_module,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([batch_norm], is_training=is_training):
        net = inputs
        if output_stride is not None:
          if output_stride % 2 != 0:
            raise ValueError('The output_stride needs to be a multiple of 2.')
          output_stride //= 2
        # Root block function operated on inputs.
        net = resnet_utils.conv2d_same(net, 32, 3, stride=2,
                                       scope='entry_flow/conv1_1')
        net = resnet_utils.conv2d_same(net, 64, 3, stride=1,
                                       scope='entry_flow/conv1_2')

        # Extract features for entry_flow, middle_flow, and exit_flow.
        net = stack_blocks_dense(net, blocks, output_stride)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection, clear_collection=True)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                             scope='prelogits_dropout')
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   kernel_size=3,
                   unit_rate_list=None,
                   use_squeeze_excite=False,
                   se_pool_size=None):
  """Helper function for creating a Xception block.

  Args:
    scope: The scope of the block.
    depth_list: The depth of the bottleneck layer for each unit.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
    kernel_size: Integer, convolution kernel size.
    unit_rate_list: A list of three integers, determining the unit rate in the
      corresponding xception block.
    use_squeeze_excite: Boolean, use squeeze-and-excitation or not.
    se_pool_size: None or integer specifying the pooling size used in SE module.

  Returns:
    An Xception block.
  """
  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'kernel_size': kernel_size,
      'unit_rate_list': unit_rate_list,
      'use_squeeze_excite': use_squeeze_excite,
      'se_pool_size': se_pool_size,
  }] * num_units)


def xception_41(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_41',
                sync_batch_norm_method='None'):
  """Xception-41 model."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=8,
                     stride=1),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     unit_rate_list=multi_grid),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_65_factory(inputs,
                        num_classes=None,
                        is_training=True,
                        global_pool=True,
                        keep_prob=0.5,
                        output_stride=None,
                        regularize_depthwise=False,
                        kernel_size=3,
                        multi_grid=None,
                        reuse=None,
                        use_squeeze_excite=False,
                        se_pool_size=None,
                        scope='xception_65',
                        sync_batch_norm_method='None'):
  """Xception-65 model factory."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=16,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     unit_rate_list=multi_grid,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_65',
                sync_batch_norm_method='None'):
  """Xception-65 model."""
  return xception_65_factory(
      inputs=inputs,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      keep_prob=keep_prob,
      output_stride=output_stride,
      regularize_depthwise=regularize_depthwise,
      multi_grid=multi_grid,
      reuse=reuse,
      scope=scope,
      use_squeeze_excite=False,
      se_pool_size=None,
      sync_batch_norm_method=sync_batch_norm_method)


def xception_71_factory(inputs,
                        num_classes=None,
                        is_training=True,
                        global_pool=True,
                        keep_prob=0.5,
                        output_stride=None,
                        regularize_depthwise=False,
                        kernel_size=3,
                        multi_grid=None,
                        reuse=None,
                        scope='xception_71',
                        use_squeeze_excite=False,
                        se_pool_size=None,
                        sync_batch_norm_method='None'):
  """Xception-71 model factory."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block3',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block4',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block5',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=16,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     unit_rate_list=multi_grid,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_71(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_71',
                sync_batch_norm_method='None'):
  """Xception-71 model."""
  return xception_71_factory(
      inputs=inputs,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      keep_prob=keep_prob,
      output_stride=output_stride,
      regularize_depthwise=regularize_depthwise,
      multi_grid=multi_grid,
      reuse=reuse,
      scope=scope,
      use_squeeze_excite=False,
      se_pool_size=None,
      sync_batch_norm_method=sync_batch_norm_method)


def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       regularize_depthwise=False,
                       use_batch_norm=True,
                       use_bounded_activation=False,
                       sync_batch_norm_method='None'):
  """Defines the default Xception arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    weights_initializer_stddev: The standard deviation of the trunctated normal
      weight initializer.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    use_batch_norm: Whether or not to use batch normalization.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
    sync_batch_norm_method: String, sync batchnorm method. Currently only
      support `None`. Also, it is only effective for Xception.

  Returns:
    An `arg_scope` to use for the Xception models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
  }
  if regularize_depthwise:
    depthwise_regularizer = slim.l2_regularizer(weight_decay)
  else:
    depthwise_regularizer = None
  activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
  batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_initializer=tf.truncated_normal_initializer(
          stddev=weights_initializer_stddev),
      activation_fn=activation_fn,
      normalizer_fn=batch_norm if use_batch_norm else None):
    with slim.arg_scope([batch_norm], **batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.separable_conv2d],
            weights_regularizer=depthwise_regularizer):
          with slim.arg_scope(
              [xception_module],
              use_bounded_activation=use_bounded_activation,
              use_explicit_padding=not use_bounded_activation) as arg_sc:
            return arg_sc
