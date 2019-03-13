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

"""Network structure used by NAS.

Here we provide a few NAS backbones for semantic segmentation.
Currently, we have

1. pnasnet
"Progressive Neural Architecture Search", Chenxi Liu, Barret Zoph,
Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei,
Alan Yuille, Jonathan Huang, Kevin Murphy. In ECCV, 2018.

2. hnasnet (also called Auto-DeepLab)
"Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic
Image Segmentation", Chenxi Liu, Liang-Chieh Chen, Florian Schroff,
Hartwig Adam, Wei Hua, Alan Yuille, Li Fei-Fei. In CVPR, 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeplab.core import nas_genotypes
from deeplab.core.nas_cell import NASBaseCell
from deeplab.core.utils import resize_bilinear
from deeplab.core.utils import scale_dimension

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


def config(num_conv_filters=20,
           total_training_steps=500000,
           drop_path_keep_prob=1.0):
  return tf.contrib.training.HParams(
      # Multiplier when spatial size is reduced by 2.
      filter_scaling_rate=2.0,
      # Number of filters of the stem output tensor.
      num_conv_filters=num_conv_filters,
      # Probability to keep each path in the cell when training.
      drop_path_keep_prob=drop_path_keep_prob,
      # Total training steps to help drop path probability calculation.
      total_training_steps=total_training_steps,
  )


def nas_arg_scope(weight_decay=4e-5, batch_norm_decay=0.9997,
                  batch_norm_epsilon=0.001):
  """Default arg scope for the NAS models."""
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      'scale': True,
      'fused': True,
  }
  weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=1/3.0, mode='FAN_IN', uniform=True)
  with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                 weights_regularizer=weights_regularizer,
                 weights_initializer=weights_initializer):
    with arg_scope([slim.fully_connected],
                   activation_fn=None, scope='FC'):
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     activation_fn=None, biases_initializer=None):
        with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
          return sc


def _nas_stem(inputs):
  """Stem used for NAS models."""
  net = slim.conv2d(inputs, 64, [3, 3], stride=2,
                    scope='conv0', padding='SAME')
  net = slim.batch_norm(net, scope='conv0_bn')
  net = tf.nn.relu(net)
  net = slim.conv2d(net, 64, [3, 3], stride=1,
                    scope='conv1', padding='SAME')
  net = slim.batch_norm(net, scope='conv1_bn')
  cell_outputs = [net]
  net = tf.nn.relu(net)
  net = slim.conv2d(net, 128, [3, 3], stride=2,
                    scope='conv2', padding='SAME')
  net = slim.batch_norm(net, scope='conv2_bn')
  cell_outputs.append(net)
  return net, cell_outputs


def _build_nas_base(images,
                    cell,
                    backbone,
                    num_classes,
                    hparams,
                    global_pool=False,
                    reuse=None,
                    scope=None,
                    final_endpoint=None):
  """Constructs a NAS model.

  Args:
    images: A tensor of size [batch, height, width, channels].
    cell: Cell structure used in the network.
    backbone: Backbone structure used in the network. A list of integers in
      which value 0 means "output_stride=4", value 1 means "output_stride=8",
      value 2 means "output_stride=16", and value 3 means "output_stride=32".
    num_classes: Number of classes to predict.
    hparams: Hyperparameters needed to construct the network.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    reuse: Whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    final_endpoint: The endpoint to construct the network up to.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  with tf.variable_scope(scope, 'nas', [images], reuse=reuse):
    end_points = {}
    def add_and_check_endpoint(endpoint_name, net):
      end_points[endpoint_name] = net
      return final_endpoint and (endpoint_name == final_endpoint)

    net, cell_outputs = _nas_stem(images)
    if add_and_check_endpoint('Stem', net):
      return net, end_points

    # Run the cells
    filter_scaling = 1.0
    for cell_num in range(len(backbone)):
      stride = 1
      if cell_num == 0:
        if backbone[0] == 1:
          stride = 2
          filter_scaling *= hparams.filter_scaling_rate
      else:
        if backbone[cell_num] == backbone[cell_num - 1] + 1:
          stride = 2
          filter_scaling *= hparams.filter_scaling_rate
        elif backbone[cell_num] == backbone[cell_num - 1] - 1:
          scaled_height = scale_dimension(tf.shape(net)[1], 2)
          scaled_width = scale_dimension(tf.shape(net)[2], 2)
          net = resize_bilinear(net, [scaled_height, scaled_width], net.dtype)
          filter_scaling /= hparams.filter_scaling_rate
      net = cell(
          net,
          scope='cell_{}'.format(cell_num),
          filter_scaling=filter_scaling,
          stride=stride,
          prev_layer=cell_outputs[-2],
          cell_num=cell_num)
      if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
        return net, end_points
      cell_outputs.append(net)
    net = tf.nn.relu(net)

    if global_pool:
      # Global average pooling.
      net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
    if num_classes is not None:
      net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                        normalizer_fn=None, scope='logits')
      end_points['predictions'] = slim.softmax(net, scope='predictions')
    return net, end_points


def pnasnet(images,
            num_classes,
            is_training=True,
            global_pool=False,
            output_stride=16,
            nas_stem_output_num_conv_filters=20,
            nas_training_hyper_parameters=None,
            reuse=None,
            scope='pnasnet',
            final_endpoint=None):
  """Builds PNASNet model."""
  hparams = config(num_conv_filters=nas_stem_output_num_conv_filters)
  if nas_training_hyper_parameters:
    hparams.set_hparam('drop_path_keep_prob',
                       nas_training_hyper_parameters['drop_path_keep_prob'])
    hparams.set_hparam('total_training_steps',
                       nas_training_hyper_parameters['total_training_steps'])
  if not is_training:
    tf.logging.info('During inference, setting drop_path_keep_prob = 1.0.')
    hparams.set_hparam('drop_path_keep_prob', 1.0)
  tf.logging.info(hparams)
  if output_stride == 8:
    backbone = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  elif output_stride == 16:
    backbone = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
  elif output_stride == 32:
    backbone = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  else:
    raise ValueError('Unsupported output_stride ', output_stride)
  cell = nas_genotypes.PNASCell(hparams.num_conv_filters,
                                hparams.drop_path_keep_prob,
                                len(backbone),
                                hparams.total_training_steps)
  with arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
    return _build_nas_base(
        images,
        cell=cell,
        backbone=backbone,
        num_classes=num_classes,
        hparams=hparams,
        global_pool=global_pool,
        reuse=reuse,
        scope=scope,
        final_endpoint=final_endpoint)


# pylint: disable=unused-argument
def hnasnet(images,
            num_classes,
            is_training=True,
            global_pool=False,
            output_stride=16,
            nas_stem_output_num_conv_filters=20,
            nas_training_hyper_parameters=None,
            reuse=None,
            scope='hnasnet',
            final_endpoint=None):
  """Builds hierarchical model."""
  hparams = config(num_conv_filters=nas_stem_output_num_conv_filters)
  if nas_training_hyper_parameters:
    hparams.set_hparam('drop_path_keep_prob',
                       nas_training_hyper_parameters['drop_path_keep_prob'])
    hparams.set_hparam('total_training_steps',
                       nas_training_hyper_parameters['total_training_steps'])
  if not is_training:
    tf.logging.info('During inference, setting drop_path_keep_prob = 1.0.')
    hparams.set_hparam('drop_path_keep_prob', 1.0)
  tf.logging.info(hparams)
  operations = [
      'atrous_5x5', 'separable_3x3_2', 'separable_3x3_2', 'atrous_3x3',
      'separable_3x3_2', 'separable_3x3_2', 'separable_5x5_2',
      'separable_5x5_2', 'separable_5x5_2', 'atrous_5x5'
  ]
  used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
  hiddenstate_indices = [1, 0, 1, 0, 3, 1, 4, 2, 3, 5]
  backbone = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
  cell = NASBaseCell(hparams.num_conv_filters,
                     operations,
                     used_hiddenstates,
                     hiddenstate_indices,
                     hparams.drop_path_keep_prob,
                     len(backbone),
                     hparams.total_training_steps)
  with arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
    return _build_nas_base(
        images,
        cell=cell,
        backbone=backbone,
        num_classes=num_classes,
        hparams=hparams,
        global_pool=global_pool,
        reuse=reuse,
        scope=scope,
        final_endpoint=final_endpoint)
