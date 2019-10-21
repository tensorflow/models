# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import i3d
from nets import inception
from nets import lenet
from nets import mobilenet_v1
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import s3dg
from nets import vgg
from nets.mobilenet import mobilenet_v2
from nets.nasnet import nasnet
from nets.nasnet import pnasnet


slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'overfeat': overfeat.overfeat,
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v1': inception.inception_v1,
                'inception_v2': inception.inception_v2,
                'inception_v3': inception.inception_v3,
                'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'i3d': i3d.i3d,
                's3dg': s3dg.s3dg,
                'lenet': lenet.lenet,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                'resnet_v1_200': resnet_v1.resnet_v1_200,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
                'mobilenet_v2': mobilenet_v2.mobilenet,
                'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
                'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,
                'nasnet_cifar': nasnet.build_nasnet_cifar,
                'nasnet_mobile': nasnet.build_nasnet_mobile,
                'nasnet_large': nasnet.build_nasnet_large,
                'pnasnet_large': pnasnet.build_pnasnet_large,
                'pnasnet_mobile': pnasnet.build_pnasnet_mobile,
               }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v1': inception.inception_v3_arg_scope,
                  'inception_v2': inception.inception_v3_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2':
                  inception.inception_resnet_v2_arg_scope,
                  'i3d': i3d.i3d_arg_scope,
                  's3dg': s3dg.s3dg_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v2': mobilenet_v2.training_scope,
                  'mobilenet_v2_035': mobilenet_v2.training_scope,
                  'mobilenet_v2_140': mobilenet_v2.training_scope,
                  'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
                  'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
                  'nasnet_large': nasnet.nasnet_large_arg_scope,
                  'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
                  'pnasnet_mobile': pnasnet.pnasnet_mobile_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification. If 0 or None,
      the logits layer is omitted and its input features are returned instead.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
          net, end_points = network_fn(images)
      The `images` input is a tensor of shape [batch_size, height, width, 3 or
       1] with height = width = network_fn.default_image_size. (The
      permissibility and treatment of other sizes depends on the network_fn.)
      The returned `end_points` are a dictionary of intermediate activations.
      The returned `net` is the topmost layer, depending on `num_classes`:
      If `num_classes` was a non-zero integer, `net` is a logits tensor
      of shape [batch_size, num_classes].
      If `num_classes` was 0 or `None`, `net` is a tensor with the input
      to the logits layer of shape [batch_size, 1, 1, num_features] or
      [batch_size, num_features]. Dropout has not been applied to this
      (even if the network's original classification does); it remains for
      the caller to do this or not.

  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes=num_classes, is_training=is_training,
                  **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
