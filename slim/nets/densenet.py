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
DenseNet from  arXiv:1608.06993v3
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import densenet_utils

slim = tf.contrib.slim
densenet_arg_scope = densenet_utils.densenet_arg_scope
DenseBlock = densenet_utils.DenseBlock
TransitionLayer = densenet_utils.TransitionLayer



def densenet(
    inputs,
    num_classes=1000,
    n_filters_first_conv=16,
    n_dense=4,
    growth_rate=12,
    n_layers_per_block=[6, 12, 24, 16],
    dropout_p=0.2,
    bottleneck=False,
    compression=1.0,
    is_training=False,
    dense_prediction=False,
    reuse=None,
    scope=None):
    """
    DenseNet as described for ImageNet use. Supports B (bottleneck) and
    C (compression) variants.
    Args:
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_dense: number of dense_blocks
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_dense + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
          is_training: whether is training or not.
      dense_prediction: Bool, defaults to False
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      if
    end_points: A dictionary from components of the network to the corresponding
      activation.
    """
    # check n_layers_per_block argument
    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == n_dense)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] *  n_dense
    else:
        raise ValueError


    with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d, DenseBlock, TransitionLayer],
                        outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

          #####################
          # First Convolution #
          #####################
          # We perform a first convolution.
          # If DenseNet BC, first convolution has 2*growth_rate output channels
          if bottleneck and compression < 1.0:
            n_filters_first_conv = 2 * growth_rate
          net = slim.conv2d(inputs, n_filters_first_conv, [7, 7],
                              stride = [2, 2], scope='first_conv')
          net = slim.pool(net, [2, 2], stride= [2, 2], pooling_type='MAX')
          n_filters = n_filters_first_conv

          #####################
          #    Dense blocks   #
          #####################

          for i in range(n_dense-1):
            # Dense Block
            net, _ = DenseBlock(net, n_layers_per_block[i],
                                growth_rate, dropout_p,
                                bottleneck=bottleneck,
                                scope='denseblock%d' % (i+1))
            n_filters += n_layers_per_block[i] * growth_rate

            # Transition layer
            net = TransitionLayer(net, n_filters, dropout_p,
                                  compression=compression,
                                  scope='transition%d'%(i+1))


          # Final dense block (no transition layer afterwards)
          net, _ = DenseBlock(net, n_layers_per_block[n_dense-1],
                              growth_rate, dropout_p,
                              scope='denseblock%d' % (n_dense))

          #####################
          #      Outputs      #
          #####################
          pool_name = 'pool%d' % (n_dense + 1)
          if dense_prediction:
            net = slim.pool(net, [7, 7], pooling_type='AVG', scope=pool_name)
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

          else:
            net = tf.reduce_mean(net, [1, 2], name=pool_name, keep_dims=True)
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='4Dlogits')
            net = tf.squeeze(net, [1, 2], name='logits')

          # Convert end_points_collection into a dictionary of end_points.
          end_points = slim.utils.convert_collection_to_dict(
              end_points_collection)

          end_points['predictions'] = slim.softmax(net, scope='predictions')
          return net, end_points
