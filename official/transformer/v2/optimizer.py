# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Optimizer from addons and learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
K = tf.keras.backend


class LazyAdam(tf.keras.optimizers.Adam):
  """Variant of the Adam optimizer that handles sparse updates more efficiently.

  The original Adam algorithm maintains two moving-average accumulators for
  each trainable variable; the accumulators are updated at every step.
  This class provides lazier handling of gradient updates for sparse
  variables.  It only updates moving-average accumulators for sparse variable
  indices that appear in the current batch, rather than updating the
  accumulators for all indices. Compared with the original Adam optimizer,
  it can provide large improvements in model training throughput for some
  applications. However, it provides slightly different semantics than the
  original Adam algorithm, and may lead to different empirical results.
  Note, amsgrad is currently not supported and the argument can only be
  False.

  This class is borrowed from:
  https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lazy_adam.py
  """

  def _resource_apply_sparse(self, grad, var, indices):
    """Applies grad for one step."""
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
    lr = (lr_t * tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power))

    # \\(m := beta1 * m + (1 - beta1) * g_t\\)
    m = self.get_slot(var, 'm')
    m_t_slice = beta_1_t * tf.gather(m, indices) + (1 - beta_1_t) * grad

    m_update_kwargs = {
        'resource': m.handle,
        'indices': indices,
        'updates': m_t_slice
    }
    m_update_op = tf.raw_ops.ResourceScatterUpdate(**m_update_kwargs)

    # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
    v = self.get_slot(var, 'v')
    v_t_slice = (beta_2_t * tf.gather(v, indices) +
                 (1 - beta_2_t) * tf.math.square(grad))

    v_update_kwargs = {
        'resource': v.handle,
        'indices': indices,
        'updates': v_t_slice
    }
    v_update_op = tf.raw_ops.ResourceScatterUpdate(**v_update_kwargs)

    # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
    var_slice = lr * m_t_slice / (tf.math.sqrt(v_t_slice) + epsilon_t)

    var_update_kwargs = {
        'resource': var.handle,
        'indices': indices,
        'updates': var_slice
    }
    var_update_op = tf.raw_ops.ResourceScatterSub(**var_update_kwargs)

    return tf.group(*[var_update_op, m_update_op, v_update_op])


class LearningRateFn(object):
  """Creates learning rate function."""

  def __init__(self, learning_rate, hidden_size, warmup_steps):
    self.learning_rate = learning_rate
    self.hidden_size = hidden_size
    self.warmup_steps = float(warmup_steps)

  def __call__(self, global_step):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    step = float(global_step)
    learning_rate = self.learning_rate
    learning_rate *= (self.hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= np.minimum(1.0, step / self.warmup_steps)
    # Apply rsqrt decay
    learning_rate /= np.sqrt(np.maximum(step, self.warmup_steps))
    return learning_rate


class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Keras callback to schedule learning rate.

  TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
  official/resnet/keras/keras_common.py.
  """

  def __init__(self, schedule, init_steps=None, verbose=False):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose
    if init_steps is None:
      init_steps = 0.0
    self.steps = float(init_steps)   # Total steps during training.

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    if not hasattr(self.model.optimizer, 'iterations'):
      raise ValueError('Optimizer must have a "iterations" attribute.')

  def on_train_batch_begin(self, batch, logs=None):
    """Adjusts learning rate for each train batch."""
    if self.verbose > 0:
      iterations = K.get_value(self.model.optimizer.iterations)
      print('Original iteration %d' % iterations)

    self.steps += 1.0
    try:  # new API
      lr = float(K.get_value(self.model.optimizer.lr))
      lr = self.schedule(self.steps, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(self.steps)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    K.set_value(self.model.optimizer.lr, lr)
    K.set_value(self.model.optimizer.iterations, self.steps)

    if self.verbose > 0:
      print('Batch %05d Step %05d: LearningRateScheduler setting learning '
            'rate to %s.' % (batch + 1, self.steps, lr))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    logs['steps'] = self.steps
