# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This code implements the Fully Convolutional DenseNet described in
https://arxiv.org/abs/1611.09326
The network consist of a downsampling path, where dense blocks and transition
down are applied, followed by an upsampling path where transition up and dense
blocks are applied. Skip connections are used between the downsampling path and
the upsampling path. Each layer is a composite function of BN - ReLU - Conv and
the last layer is a softmax layer.

Typical use:

   from tensorflow.contrib.slim.nets import densenet

   with slim.arg_scope(densenet.densenet_arg_scope()):
      net, end_points = densenet.fc_densenet(inputs,
                                             n_classes, is_training=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import densenet_utils


slim = tf.contrib.slim
densenet_arg_scope = densenet_utils.densenet_arg_scope
DenseBlock = densenet_utils.DenseBlock
TransitionDown = densenet_utils.TransitionDown
TransitionUp = densenet_utils.TransitionUp


def fc_densenet(
    inputs,
    num_classes=11,
    n_filters_first_conv=48,
    n_pool=4,
    growth_rate=12,
    n_layers_per_block=5,
    dropout_p=0.2,
    is_training=False,
    reuse=None,
    scope=None):
    """
    Args:
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down =
        number of transition up
      growth_rate: number of new feature maps created by each layer in a
        dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
    """

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, 'fc_densenet', [inputs], reuse=reuse) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d, DenseBlock, TransitionUp,
                           TransitionDown],
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

          #####################
          # First Convolution #
          #####################
          # We perform a first convolution.
          stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3],
                              scope='first_conv')

          n_filters = n_filters_first_conv
          #####################
          # Downsampling path #
          #####################

          skip_connection_list = []

          for i in range(n_pool):
            # Dense Block
            stack, _ = DenseBlock(stack, n_layers_per_block[i],
                                  growth_rate, dropout_p,
                                  scope='denseblock%d' % (i+1))
            n_filters += growth_rate * n_layers_per_block[i]
            # At the end of the dense block, the current stack is stored
            # in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = TransitionDown(stack, n_filters, dropout_p,
                                   scope='transitiondown%d'%(i+1))

          skip_connection_list = skip_connection_list[::-1]

          #####################
          #     Bottleneck    #
          #####################

          # Dense Block
          # We will only upsample the new feature maps
          stack, block_to_upsample = DenseBlock(
            stack, n_layers_per_block[n_pool],
            growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))


          #######################
          #   Upsampling path   #
          #######################

          for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(block_to_upsample, skip_connection_list[i],
                                 n_filters_keep,
                                 scope='transitionup%d' % (n_pool + i + 1))

            # Dense Block
            # We will only upsample the new feature maps
            stack, block_to_upsample = DenseBlock(
              stack, n_layers_per_block[n_pool + i + 1],
              growth_rate, dropout_p,
              scope='denseblock%d' % (n_pool + i + 2))


          #####################
          #      Softmax      #
          #####################
          net = slim.conv2d(stack, num_classes, [1, 1], scope='logits')
          # Convert end_points_collection into a dictionary of end_points.
          end_points = slim.utils.convert_collection_to_dict(
              end_points_collection)
          end_points['predictions'] = slim.softmax(net, scope='predictions')
          return net, end_points
