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

"""Utility functions for setting up the CMP graph.
"""

import os, numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope
import logging
from src import utils
import src.file_utils as fu
from tfcode import tf_utils

resnet_v2 = tf_utils.resnet_v2
custom_residual_block = tf_utils.custom_residual_block

def value_iteration_network(
    fr, num_iters, val_neurons, action_neurons, kernel_size, share_wts=False,
    name='vin', wt_decay=0.0001, activation_fn=None, shape_aware=False):
  """
  Constructs a Value Iteration Network, convolutions and max pooling across
  channels.
  Input:
    fr:             NxWxHxC
    val_neurons:    Number of channels for maintaining the value.
    action_neurons: Computes action_neurons * val_neurons at each iteration to
                    max pool over.
  Output:
    value image:  NxHxWx(val_neurons)
  """
  init_var = np.sqrt(2.0/(kernel_size**2)/(val_neurons*action_neurons))
  vals = []
  with tf.variable_scope(name) as varscope:
    if shape_aware == False:
      fr_shape = tf.unstack(tf.shape(fr))
      val_shape = tf.stack(fr_shape[:-1] + [val_neurons])
      val = tf.zeros(val_shape, name='val_init')
    else:
      val = tf.expand_dims(tf.zeros_like(fr[:,:,:,0]), dim=-1) * \
          tf.constant(0., dtype=tf.float32, shape=[1,1,1,val_neurons])
      val_shape = tf.shape(val)
    vals.append(val)
    for i in range(num_iters):
      if share_wts:
        # The first Value Iteration maybe special, so it can have its own
        # paramterss.
        scope = 'conv'
        if i == 0: scope = 'conv_0'
        if i > 1: varscope.reuse_variables()
      else:
        scope = 'conv_{:d}'.format(i)
      val = slim.conv2d(tf.concat([val, fr], 3, name='concat_{:d}'.format(i)),
                        num_outputs=action_neurons*val_neurons,
                        kernel_size=kernel_size, stride=1, activation_fn=activation_fn,
                        scope=scope, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(wt_decay),
                        weights_initializer=tf.random_normal_initializer(stddev=init_var),
                        biases_initializer=tf.zeros_initializer())
      val = tf.reshape(val, [-1, action_neurons*val_neurons, 1, 1],
                       name='re_{:d}'.format(i))
      val = slim.max_pool2d(val, kernel_size=[action_neurons,1],
                            stride=[action_neurons,1], padding='VALID',
                            scope='val_{:d}'.format(i))
      val = tf.reshape(val, val_shape, name='unre_{:d}'.format(i))
      vals.append(val)
  return val, vals


def rotate_preds(loc_on_map, relative_theta, map_size, preds,
                 output_valid_mask):
  with tf.name_scope('rotate'):
    flow_op = tf_utils.get_flow(loc_on_map, relative_theta, map_size=map_size)
    if type(preds) != list:
      rotated_preds, valid_mask_warps = tf_utils.dense_resample(preds, flow_op,
                                                                output_valid_mask)
    else:
      rotated_preds = [] ;valid_mask_warps = []
      for pred in preds:
        rotated_pred, valid_mask_warp = tf_utils.dense_resample(pred, flow_op,
                                                                output_valid_mask)
        rotated_preds.append(rotated_pred)
        valid_mask_warps.append(valid_mask_warp)
  return rotated_preds, valid_mask_warps

def get_visual_frustum(map_size, shape_like, expand_dims=[0,0]):
  with tf.name_scope('visual_frustum'):
    l = np.tril(np.ones(map_size)) ;l = l + l[:,::-1]
    l = (l == 2).astype(np.float32)
    for e in expand_dims:
      l = np.expand_dims(l, axis=e)
    confs_probs = tf.constant(l, dtype=tf.float32)
    confs_probs = tf.ones_like(shape_like, dtype=tf.float32) * confs_probs
  return confs_probs

def deconv(x, is_training, wt_decay, neurons, strides, layers_per_block,
            kernel_size, conv_fn, name, offset=0):
  """Generates a up sampling network with residual connections. 
  """
  batch_norm_param = {'center': True, 'scale': True,
                      'activation_fn': tf.nn.relu,
                      'is_training': is_training}
  outs = []
  for i, (neuron, stride) in enumerate(zip(neurons, strides)):
    for s in range(layers_per_block):
      scope = '{:s}_{:d}_{:d}'.format(name, i+1+offset,s+1)
      x = custom_residual_block(x, neuron, kernel_size, stride, scope,
                                is_training, wt_decay, use_residual=True,
                                residual_stride_conv=True, conv_fn=conv_fn,
                                batch_norm_param=batch_norm_param)
      stride = 1
    outs.append((x,True))
  return x, outs

def fr_v2(x, output_neurons, inside_neurons, is_training, name='fr',
          wt_decay=0.0001, stride=1, updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Performs fusion of information between the map and the reward map.
  Inputs
    x:   NxHxWxC1

  Outputs
    fr map:     NxHxWx(output_neurons)
  """
  if type(stride) != list:
    stride = [stride]
  with slim.arg_scope(resnet_v2.resnet_utils.resnet_arg_scope(
      is_training=is_training, weight_decay=wt_decay)):
    with slim.arg_scope([slim.batch_norm], updates_collections=updates_collections) as arg_sc:
      # Change the updates_collections for the conv normalizer_params to None
      for i in range(len(arg_sc.keys())):
        if 'convolution' in arg_sc.keys()[i]:
          arg_sc.values()[i]['normalizer_params']['updates_collections'] = updates_collections
      with slim.arg_scope(arg_sc):
        bottleneck = resnet_v2.bottleneck
        blocks = []
        for i, s in enumerate(stride):
          b = resnet_v2.resnet_utils.Block(
              'block{:d}'.format(i + 1), bottleneck, [{
                  'depth': output_neurons,
                  'depth_bottleneck': inside_neurons,
                  'stride': stride[i]
              }])
          blocks.append(b)
        x, outs = resnet_v2.resnet_v2(x, blocks, num_classes=None, global_pool=False,
                                     output_stride=None, include_root_block=False,
                                     reuse=False, scope=name)
  return x, outs
