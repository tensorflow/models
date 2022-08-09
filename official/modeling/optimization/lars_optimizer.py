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

"""Layer-wise adaptive rate scaling optimizer."""
import re
from typing import Text, List, Optional

import tensorflow as tf


# pylint: disable=protected-access


class LARS(tf.keras.optimizers.legacy.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate: float = 0.01,
               momentum: float = 0.9,
               weight_decay_rate: float = 0.0,
               eeta: float = 0.001,
               nesterov: bool = False,
               classic_momentum: bool = True,
               exclude_from_weight_decay: Optional[List[Text]] = None,
               exclude_from_layer_adaptation: Optional[List[Text]] = None,
               name: Text = "LARS",
               **kwargs):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: `float` for learning rate. Defaults to 0.01.
      momentum: `float` hyperparameter >= 0 that accelerates gradient descent
          in the relevant direction and dampens oscillations. Defaults to 0.9.
      weight_decay_rate: `float` for weight decay.
      eeta: `float` LARS coefficient as used in the paper. Default set to LARS
          coefficient from the paper. (eeta / weight_decay) determines the
          highest scaling factor in LARS..
      nesterov: 'boolean' for whether to use nesterov momentum.
      classic_momentum: `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momentum update in
          classic momentum, but after momentum for popular momentum.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      name: `Text` as optional name for the operations created when applying
        gradients. Defaults to "LARS".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for
        backward compatibility, recommended to use `learning_rate` instead.
    """
    super(LARS, self).__init__(name, **kwargs)

    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("decay", self._initial_decay)
    self.momentum = momentum
    self.weight_decay_rate = weight_decay_rate
    self.eeta = eeta
    self.nesterov = nesterov
    self.classic_momentum = classic_momentum
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "momentum")

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    v = self.get_slot(param, "momentum")

    if self._use_weight_decay(param_name):
      grad += self.weight_decay_rate * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    raise NotImplementedError("Applying sparse gradients is not implemented.")

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def get_config(self):
    config = super(LARS, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay_rate": self.weight_decay_rate,
        "eeta": self.eeta,
        "nesterov": self.nesterov,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
