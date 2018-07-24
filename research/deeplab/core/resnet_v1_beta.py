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

"""Resnet v1 model variants.

Code branched out from slim/nets/resnet_v1.py, and please refer to it for
more details.

The original version ResNets-v1 were proposed by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils

slim = tf.contrib.slim

_DEFAULT_MULTI_GRID = [1, 1, 1]


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               unit_rate=1,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    unit_rate: An integer, unit rate for atrous convolution.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth,
          [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate*unit_rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')
    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def root_block_fn_for_beta_variant(net):
  """Gets root_block_fn for beta variant.

  ResNet-v1 beta variant modifies the first original 7x7 convolution to three
  3x3 convolutions.

  Args:
    net: A tensor of size [batch, height, width, channels], input to the model.

  Returns:
    A tensor after three 3x3 convolutions.
  """
  net = resnet_utils.conv2d_same(net, 64, 3, stride=2, scope='conv1_1')
  net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
  net = resnet_utils.conv2d_same(net, 128, 3, stride=1, scope='conv1_3')

  return net


def resnet_v1_beta(inputs,
                   blocks,
                   num_classes=None,
                   is_training=None,
                   global_pool=True,
                   output_stride=None,
                   root_block_fn=None,
                   reuse=None,
                   scope=None):
  """Generator for v1 ResNet models (beta variant).

  This function generates a family of modified ResNet v1 models. In particular,
  the first original 7x7 convolution is replaced with three 3x3 convolutions.
  See the resnet_v1_*() methods for specific model instantiations, obtained by
  selecting different block instantiations that produce ResNets of various
  depths.

  The code is modified from slim/nets/resnet_v1.py, and please refer to it for
  more details.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    root_block_fn: The function consisting of convolution operations applied to
      the root input. If root_block_fn is None, use the original setting of
      RseNet-v1, which is simply one convolution with 7x7 kernel and stride=2.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  if root_block_fn is None:
    root_block_fn = functools.partial(resnet_utils.conv2d_same,
                                      num_outputs=64,
                                      kernel_size=7,
                                      stride=2,
                                      scope='conv1')
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      if is_training is not None:
        arg_scope = slim.arg_scope([slim.batch_norm], is_training=is_training)
      else:
        arg_scope = slim.arg_scope([])
      with arg_scope:
        net = inputs
        if output_stride is not None:
          if output_stride % 4 != 0:
            raise ValueError('The output_stride needs to be a multiple of 4.')
          output_stride /= 4
        net = root_block_fn(net)
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def resnet_v1_beta_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 beta variant bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'unit_rate': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'unit_rate': 1
  }])


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=None,
                 global_pool=False,
                 output_stride=None,
                 multi_grid=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """Resnet v1 50.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
  """
  if multi_grid is None:
    multi_grid = _DEFAULT_MULTI_GRID
  else:
    if len(multi_grid) != 3:
      raise ValueError('Expect multi_grid to have length 3.')

  blocks = [
      resnet_v1_beta_block(
          'block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_beta_block(
          'block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_beta_block(
          'block3', base_depth=256, num_units=6, stride=2),
      resnet_utils.Block('block4', bottleneck, [
          {'depth': 2048,
           'depth_bottleneck': 512,
           'stride': 1,
           'unit_rate': rate} for rate in multi_grid]),
  ]
  return resnet_v1_beta(
      inputs,
      blocks=blocks,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      output_stride=output_stride,
      reuse=reuse,
      scope=scope)


def resnet_v1_50_beta(inputs,
                      num_classes=None,
                      is_training=None,
                      global_pool=False,
                      output_stride=None,
                      multi_grid=None,
                      reuse=None,
                      scope='resnet_v1_50'):
  """Resnet v1 50 beta variant.

  This variant modifies the first convolution layer of ResNet-v1-50. In
  particular, it changes the original one 7x7 convolution to three 3x3
  convolutions.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
  """
  if multi_grid is None:
    multi_grid = _DEFAULT_MULTI_GRID
  else:
    if len(multi_grid) != 3:
      raise ValueError('Expect multi_grid to have length 3.')

  blocks = [
      resnet_v1_beta_block(
          'block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_beta_block(
          'block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_beta_block(
          'block3', base_depth=256, num_units=6, stride=2),
      resnet_utils.Block('block4', bottleneck, [
          {'depth': 2048,
           'depth_bottleneck': 512,
           'stride': 1,
           'unit_rate': rate} for rate in multi_grid]),
  ]
  return resnet_v1_beta(
      inputs,
      blocks=blocks,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      output_stride=output_stride,
      root_block_fn=functools.partial(root_block_fn_for_beta_variant),
      reuse=reuse,
      scope=scope)


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=False,
                  output_stride=None,
                  multi_grid=None,
                  reuse=None,
                  scope='resnet_v1_101'):
  """Resnet v1 101.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
  """
  if multi_grid is None:
    multi_grid = _DEFAULT_MULTI_GRID
  else:
    if len(multi_grid) != 3:
      raise ValueError('Expect multi_grid to have length 3.')

  blocks = [
      resnet_v1_beta_block(
          'block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_beta_block(
          'block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_beta_block(
          'block3', base_depth=256, num_units=23, stride=2),
      resnet_utils.Block('block4', bottleneck, [
          {'depth': 2048,
           'depth_bottleneck': 512,
           'stride': 1,
           'unit_rate': rate} for rate in multi_grid]),
  ]
  return resnet_v1_beta(
      inputs,
      blocks=blocks,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      output_stride=output_stride,
      reuse=reuse,
      scope=scope)


def resnet_v1_101_beta(inputs,
                       num_classes=None,
                       is_training=None,
                       global_pool=False,
                       output_stride=None,
                       multi_grid=None,
                       reuse=None,
                       scope='resnet_v1_101'):
  """Resnet v1 101 beta variant.

  This variant modifies the first convolution layer of ResNet-v1-101. In
  particular, it changes the original one 7x7 convolution to three 3x3
  convolutions.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
  """
  if multi_grid is None:
    multi_grid = _DEFAULT_MULTI_GRID
  else:
    if len(multi_grid) != 3:
      raise ValueError('Expect multi_grid to have length 3.')

  blocks = [
      resnet_v1_beta_block(
          'block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_beta_block(
          'block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_beta_block(
          'block3', base_depth=256, num_units=23, stride=2),
      resnet_utils.Block('block4', bottleneck, [
          {'depth': 2048,
           'depth_bottleneck': 512,
           'stride': 1,
           'unit_rate': rate} for rate in multi_grid]),
  ]
  return resnet_v1_beta(
      inputs,
      blocks=blocks,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      output_stride=output_stride,
      root_block_fn=functools.partial(root_block_fn_for_beta_variant),
      reuse=reuse,
      scope=scope)
