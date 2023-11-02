# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
from typing import List, Optional

from absl import logging

import tensorflow as tf, tf_keras

from official.modeling import optimization
from official.nlp import optimization as nlp_optimization


@dataclasses.dataclass
class ViTAdamWConfig(optimization.AdamWeightDecayConfig):
  layer_decay: Optional[float] = 1.0
  vars_substr: Optional[List[str]] = None
  layers_idx: Optional[List[int]] = None


@dataclasses.dataclass
class OptimizerConfig(optimization.OptimizerConfig):
  vit_adamw: ViTAdamWConfig = dataclasses.field(default_factory=ViTAdamWConfig)


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
class _ViTAdamW(nlp_optimization.AdamWeightDecay):
  """Custom AdamW to support different lr scaling for backbone.

  The code is copied from AdamWeightDecay and Adam with learning scaling.
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
               layer_decay=1.0,
               vars_substr=None,
               layers_idx=None,
               name='ViTAdamWeightDecay',
               **kwargs):
    super(_ViTAdamW,
          self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad,
                         weight_decay_rate, include_in_weight_decay,
                         exclude_from_weight_decay, gradient_clip_norm, name,
                         **kwargs)
    self._layer_decay = layer_decay
    self._vars_substr = vars_substr
    self._layers_idx = layers_idx
    self._max_idx = max(layers_idx) + 1 if layers_idx is not None else 1

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    apply_state = kwargs['apply_state']
    if (
        self._layer_decay != 1.0
        and self._vars_substr is not None
        and self._layers_idx is not None
    ):
      is_decayed = False
      for var_substr, idx in zip(self._vars_substr, self._layers_idx):
        if var_substr in var.name:
          decay_factor = self._layer_decay ** (self._max_idx - idx)
          lr_t = lr_t * decay_factor
          is_decayed = True
          logging.debug(
              'Applying layer-wise lr decay: %s: %f', var.name, decay_factor)
          break
      if not is_decayed:
        logging.debug('Ignore layer-wise lr decay: %s', var.name)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = ((apply_state or {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      m = self.get_slot(var, 'm')
      v = self.get_slot(var, 'v')
      lr = coefficients['lr_t']
      if (
          self._layer_decay != 1.0
          and self._vars_substr is not None
          and self._layers_idx is not None
      ):
        for var_substr, idx in zip(self._vars_substr, self._layers_idx):
          if var_substr in var.name:
            lr = lr * (self._layer_decay ** (self._max_idx - idx))
            break

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
    if (
        self._layer_decay != 1.0
        and self._vars_substr is not None
        and self._layers_idx is not None
    ):
      is_decayed = False
      for var_substr, idx in zip(self._vars_substr, self._layers_idx):
        if var_substr in var.name:
          decay_factor = self._layer_decay ** (self._max_idx - idx)
          lr_t = lr_t * decay_factor
          is_decayed = True
          logging.debug(
              'Applying layer-wise lr decay: %s: %f', var.name, decay_factor)
          break
      if not is_decayed:
        logging.debug('Ignore layer-wise lr decay: %s', var.name)
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
      lr = coefficients['lr_t']
      if (
          self._layer_decay != 1.0
          and self._vars_substr is not None
          and self._layers_idx is not None
      ):
        for var_substr, idx in zip(self._vars_substr, self._layers_idx):
          if var_substr in var.name:
            lr = lr * (self._layer_decay ** (self._max_idx - idx))
            break
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

optimization.register_optimizer_cls('vit_adamw', _ViTAdamW)
