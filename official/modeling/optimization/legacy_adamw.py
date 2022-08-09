# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Adam optimizer with weight decay that exactly matches the original BERT."""

import re

from absl import logging
import tensorflow as tf


class AdamWeightDecay(tf.keras.optimizers.legacy.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  [Warning!]: Keras optimizer supports gradient clipping and has an AdamW
  implementation. Please consider evaluating the choice in Keras package.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want to decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               include_in_weight_decay=None,
               exclude_from_weight_decay=None,
               gradient_clip_norm=1.0,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                          epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self.gradient_clip_norm = gradient_clip_norm
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay
    logging.info('AdamWeightDecay gradient_clip_norm=%f', gradient_clip_norm)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,  # pytype: disable=attribute-error  # typed-keras
                                                apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
        self.weight_decay_rate, name='adam_weight_decay_rate')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    grads, tvars = list(zip(*grads_and_vars))
    if experimental_aggregate_gradients and self.gradient_clip_norm > 0.0:
      # when experimental_aggregate_gradients = False, apply_gradients() no
      # longer implicitly allreduce gradients, users manually allreduce gradient
      # and passed the allreduced grads_and_vars. For now, the
      # clip_by_global_norm will be moved to before the explicit allreduce to
      # keep the math the same as TF 1 and pre TF 2.2 implementation.
      (grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=self.gradient_clip_norm)
    return super(AdamWeightDecay, self).apply_gradients(
        zip(grads, tvars),
        name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['lr_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_dense(grad, var, **kwargs)  # pytype: disable=attribute-error  # typed-keras

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_sparse(grad, var, indices, **kwargs)  # pytype: disable=attribute-error  # typed-keras

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate': self.weight_decay_rate,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False

    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True
