# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

WORKSPACE_PATH = '/home/wew57/dataset/'

def clip_gradients(grads_and_vars, clip_factor = 2.5):
    """ Clip gradients to [-clip_factor*stddev, clip_factor*stddev]."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = []
    for gradient in gradients:
        if gradient is None:
            clipped_gradients.append(None)
            continue

        mean_gradient = tf.reduce_mean(gradient)
        stddev_gradient = tf.sqrt(tf.reduce_mean(tf.square(gradient - mean_gradient)))
        clipped_gradient = tf.clip_by_value(gradient, -clip_factor * stddev_gradient, clip_factor * stddev_gradient)

        clipped_gradients.append(clipped_gradient)
    return list(zip(clipped_gradients, variables))

def stochastical_binarize_gradients(grads_and_vars):
  """Stochastically binarize gradients."""
  gradients, variables = zip(*grads_and_vars)
  binarized_gradients = []
  for gradient in gradients:
    if gradient is None:
      binarized_gradients.append(None)
      continue
    if isinstance(gradient, tf.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()

    #mean_gradient = tf.reduce_mean(gradient)
    #stddev_gradient = tf.sqrt(tf.reduce_mean(tf.square(gradient - mean_gradient)))
    #clipped_gradient = tf.clip_by_value(gradient,-clip_factor*stddev_gradient,clip_factor*stddev_gradient)
    zeros = tf.zeros(gradient_shape)
    abs_gradient = tf.abs(gradient)
    #tf.summary.tensor_summary(gradient.op.name + '/abs_gradients', abs_gradient)
    max_abs_gradient = tf.reduce_max( abs_gradient )
    #tf.summary.scalar(gradient.op.name + '/max_abs_gradients', max_abs_gradient)
    sign_gradient = tf.sign( gradient )
    rnd_sample = tf.random_uniform(gradient_shape,0,max_abs_gradient)
    where_cond = tf.less(rnd_sample, abs_gradient)
    binarized_gradient = tf.where(where_cond, sign_gradient * max_abs_gradient, zeros)

    #debug_op = tf.Print(gradient, [gradient, rnd_sample,binarized_gradient],
    #                    first_n=1, summarize=64,
    #                    message=gradient.op.name)
    #with tf.control_dependencies([debug_op]):
    #  binarized_gradient = tf.negative(tf.negative(binarized_gradient))

    binarized_gradients.append(binarized_gradient)
  return list(zip(binarized_gradients, variables))