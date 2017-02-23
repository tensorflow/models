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

def clip_gradients_by_stddev(grads_and_vars, clip_factor = 2.5):
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

def clip_gradients_by_thresholds(grads_and_vars, thresholds):
    """ Clip gradients to [-threshold, threshold]."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = []
    for gradient,threshold in zip(gradients,thresholds):
        if gradient is None:
            clipped_gradients.append(None)
            continue

        clipped_gradient = tf.clip_by_value(gradient, -threshold, threshold)

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

def gradient_binarizing_scalers(grads_and_vars, clip_factor):
    """ Get the scalers."""
    gradients, variables = zip(*grads_and_vars)
    scalers = []
    for gradient in gradients:
        if gradient is None:
            scalers.append(None)
            continue

        if(clip_factor > 1.0e-5):
            mean_gradient = tf.reduce_mean(gradient)
            stddev_gradient = tf.sqrt(tf.reduce_mean(tf.square(gradient - mean_gradient)))
            scalers.append(clip_factor * stddev_gradient)
        else:
            scalers.append(tf.reduce_max(tf.abs(gradient)))

    return list(zip(scalers, variables))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def average_gradients2(tower_grads):
  """This is identical to average_gradients() but returns pairs of (shared gradient, unshared variable) across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of Lists of pairs of (gradient, variable) where the gradient has been averaged
     across all towers and variable is the one in each tower.
  """
  res = []
  mean_grads = average_gradients(tower_grads)
  for grad_and_vars in tower_grads:
      _grads = []
      for _grad1, _grad2 in zip(mean_grads, grad_and_vars):
          _grads.append( (_grad1[0],_grad2[1]) )
      res.append(_grads)

  return res

def average_scalers(tower_scalers):
  """Calculate the average scalers for gradients across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_scalers: List of lists of (scaler, variable) tuples. The outer list
      is over individual scaler. The inner list is over the scaler
      calculation for each tower.
  Returns:
     List of pairs of scaler where the scaler has been averaged
     across all towers.
  """
  average_scalers = []
  for scale_and_vars in zip(*tower_scalers):
    # Note that each scale_and_vars looks like the following:
    #   ((scale0_gpu0, var0_gpu0), ... , (scale0_gpuN, var0_gpuN))
    scalers = []
    for s, _ in scale_and_vars:
      # Add 0 dimension to the scalers to represent the tower.
      expanded_s = tf.expand_dims(s, 0)

      # Append on a 'tower' dimension which we will average over below.
      scalers.append(expanded_s)

    # Average over the 'tower' dimension.
    scaler = tf.concat(scalers, 0)
    scaler = tf.reduce_mean(scaler, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    #v = scale_and_vars[0][1]
    #scale_and_var = (scale, v)
    average_scalers.append(scaler)
  return average_scalers