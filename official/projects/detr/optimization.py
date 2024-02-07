# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Customized optimizer to match paper results."""

import dataclasses
import tensorflow as tf, tf_keras
from official.modeling import optimization
from official.nlp import optimization as nlp_optimization


@dataclasses.dataclass
class DETRAdamWConfig(optimization.AdamWeightDecayConfig):
  pass


@dataclasses.dataclass
class OptimizerConfig(optimization.OptimizerConfig):
  detr_adamw: DETRAdamWConfig = dataclasses.field(
      default_factory=DETRAdamWConfig
  )


@dataclasses.dataclass
class OptimizationConfig(optimization.OptimizationConfig):
  """Configuration for optimizer and learning rate schedule.

  Attributes:
    optimizer: optimizer oneof config.
    ema: optional exponential moving average optimizer config, if specified, ema
      optimizer will be used.
    learning_rate: learning rate oneof config.
    warmup: warmup oneof config.
  """
  optimizer: OptimizerConfig = dataclasses.field(
      default_factory=OptimizerConfig
  )


# TODO(frederickliu): figure out how to make this configuable.
# TODO(frederickliu): Study if this is needed.
class _DETRAdamW(nlp_optimization.AdamWeightDecay):
  """Custom AdamW to support different lr scaling for backbone.

  The code is copied from AdamWeightDecay and Adam with learning scaling.
  """

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    apply_state = kwargs['apply_state']
    if 'detr' not in var.name:
      lr_t *= 0.1
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = ((apply_state or {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      m = self.get_slot(var, 'm')
      v = self.get_slot(var, 'v')
      lr = coefficients[
          'lr_t'] * 0.1 if 'detr' not in var.name else coefficients['lr_t']

      if not self.amsgrad:
        return tf.raw_ops.ResourceApplyAdam(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=coefficients['beta_1_power'],
            beta2_power=coefficients['beta_2_power'],
            lr=lr,
            beta1=coefficients['beta_1_t'],
            beta2=coefficients['beta_2_t'],
            epsilon=coefficients['epsilon'],
            grad=grad,
            use_locking=self._use_locking)
      else:
        vhat = self.get_slot(var, 'vhat')
        return tf.raw_ops.ResourceApplyAdamWithAmsgrad(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            vhat=vhat.handle,
            beta1_power=coefficients['beta_1_power'],
            beta2_power=coefficients['beta_2_power'],
            lr=lr,
            beta1=coefficients['beta_1_t'],
            beta2=coefficients['beta_2_t'],
            epsilon=coefficients['epsilon'],
            grad=grad,
            use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    apply_state = kwargs['apply_state']
    if 'detr' not in var.name:
      lr_t *= 0.1
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = ((apply_state or {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, 'm')
      m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
      m_t = tf.compat.v1.assign(m, m * coefficients['beta_1_t'],
                                use_locking=self._use_locking)
      with tf.control_dependencies([m_t]):
        m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

      # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
      v = self.get_slot(var, 'v')
      v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
      v_t = tf.compat.v1.assign(v, v * coefficients['beta_2_t'],
                                use_locking=self._use_locking)
      with tf.control_dependencies([v_t]):
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
      lr = coefficients[
          'lr_t'] * 0.1 if 'detr' not in var.name else coefficients['lr_t']
      if not self.amsgrad:
        v_sqrt = tf.sqrt(v_t)
        var_update = tf.compat.v1.assign_sub(
            var, lr * m_t / (v_sqrt + coefficients['epsilon']),
            use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])
      else:
        v_hat = self.get_slot(var, 'vhat')
        v_hat_t = tf.maximum(v_hat, v_t)
        with tf.control_dependencies([v_hat_t]):
          v_hat_t = tf.compat.v1.assign(
              v_hat, v_hat_t, use_locking=self._use_locking)
        v_hat_sqrt = tf.sqrt(v_hat_t)
        var_update = tf.compat.v1.assign_sub(
            var,
            lr* m_t / (v_hat_sqrt + coefficients['epsilon']),
            use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t, v_hat_t])

optimization.register_optimizer_cls('detr_adamw', _DETRAdamW)
