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

"""Optimizer factory for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras

from official.legacy.image_classification import learning_rate
from official.legacy.image_classification.configs import base_configs
from official.modeling import optimization
from official.modeling.optimization import legacy_adamw

# pylint: disable=protected-access

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


class Lookahead(tf_keras.optimizers.legacy.Optimizer):
  """This class allows to extend optimizers with the lookahead mechanism.

  The mechanism is proposed by Michael R. Zhang et.al in the paper [Lookahead
  Optimizer: k steps forward, 1 step back] (https://arxiv.org/abs/1907.08610v1).
  The optimizer iteratively updates two sets of weights: the search directions
  for weights are chosen by the inner optimizer, while the "slow weights" are
  updated each `k` steps based on the directions of the "fast weights" and the
  two sets of weights are synchronized. This method improves the learning
  stability and lowers the variance of its inner optimizer.

  Example of usage:

  ```python
  opt = tf_keras.optimizers.SGD(learning_rate) opt =
  tfa.optimizers.Lookahead(opt)
  ```
  """

  def __init__(
      self,
      optimizer: tf_keras.optimizers.Optimizer,
      sync_period: int = 6,
      slow_step_size: FloatTensorLike = 0.5,
      name: str = 'Lookahead',
      **kwargs,
  ):
    """Wrap optimizer with the lookahead mechanism.

    Args:
      optimizer: The original optimizer that will be used to compute and apply
        the gradients.
      sync_period: An integer. The synchronization period of lookahead. Enable
        lookahead mechanism by setting it with a positive value.
      slow_step_size: A floating point value. The ratio for updating the slow
        weights.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Lookahead".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(name, **kwargs)

    if isinstance(optimizer, str):
      optimizer = tf_keras.optimizers.get(optimizer)
    if not isinstance(
        optimizer,
        (tf_keras.optimizers.Optimizer, tf_keras.optimizers.legacy.Optimizer),
    ):
      raise TypeError(
          'optimizer is not an object of tf_keras.optimizers.Optimizer'
      )

    self._optimizer = optimizer
    self._set_hyper('sync_period', sync_period)
    self._set_hyper('slow_step_size', slow_step_size)
    self._initialized = False
    self._track_trackable(self._optimizer, 'lh_base_optimizer')

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access
    for var in var_list:
      self.add_slot(var, 'slow', initializer=var)

  def _create_hypers(self):
    self._optimizer._create_hypers()  # pylint: disable=protected-access

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

  def apply_gradients(
      self, grads_and_vars, name=None, skip_gradients_aggregation=None, **kwargs
  ):
    self._optimizer._iterations = self.iterations  # pylint: disable=protected-access
    return super().apply_gradients(grads_and_vars, name, **kwargs)

  def _look_ahead_op(self, var):
    var_dtype = var.dtype.base_dtype
    slow_var = self.get_slot(var, 'slow')
    local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
    sync_period = self._get_hyper('sync_period', tf.dtypes.int64)
    slow_step_size = self._get_hyper('slow_step_size', var_dtype)
    step_back = slow_var + slow_step_size * (var - slow_var)
    sync_cond = tf.equal(
        tf.math.floordiv(local_step, sync_period) * sync_period, local_step
    )
    with tf.control_dependencies([step_back]):
      slow_update = slow_var.assign(
          tf.where(sync_cond, step_back, slow_var),
          use_locking=self._use_locking,
      )
      var_update = var.assign(
          tf.where(sync_cond, step_back, var), use_locking=self._use_locking
      )
    return tf.group(slow_update, var_update)

  @property
  def weights(self):
    return self._weights + self._optimizer.weights

  def _resource_apply_dense(self, grad, var):
    train_op = self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access
    with tf.control_dependencies([train_op]):
      look_ahead_op = self._look_ahead_op(var)
    return tf.group(train_op, look_ahead_op)

  def _resource_apply_sparse(self, grad, var, indices):
    train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
        grad, var, indices
    )
    with tf.control_dependencies([train_op]):
      look_ahead_op = self._look_ahead_op(var)
    return tf.group(train_op, look_ahead_op)

  def get_config(self):
    config = {
        'optimizer': tf_keras.optimizers.serialize(self._optimizer),
        'sync_period': self._serialize_hyperparameter('sync_period'),
        'slow_step_size': self._serialize_hyperparameter('slow_step_size'),
    }
    base_config = super().get_config()
    return {**base_config, **config}

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')

  @learning_rate.setter
  def learning_rate(self, value):
    self._optimizer._set_hyper('learning_rate', value)

  @property
  def lr(self):
    return self.learning_rate

  @lr.setter
  def lr(self, lr):
    self.learning_rate = lr

  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf_keras.optimizers.deserialize(
        config.pop('optimizer'), custom_objects=custom_objects
    )
    return cls(optimizer, **config)


def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf_keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any],
    model: Optional[tf_keras.Model] = None):
  """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples: sgd,
      momentum, rmsprop.
    base_learning_rate: `tf_keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params. This
      should contain optimizer specific parameters such as `base_learning_rate`,
      `decay`, etc.
    model: The `tf_keras.Model`. This is used for the shadow copy if using
      `ExponentialMovingAverage`.

  Returns:
    A tf_keras.optimizers.legacy.Optimizer.

  Raises:
    ValueError if the provided optimizer_name is not supported.

  """
  optimizer_name = optimizer_name.lower()
  logging.info('Building %s optimizer with params %s', optimizer_name, params)

  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf_keras.optimizers.legacy.SGD(
        learning_rate=base_learning_rate, nesterov=nesterov)
  elif optimizer_name == 'momentum':
    logging.info('Using momentum optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf_keras.optimizers.legacy.SGD(
        learning_rate=base_learning_rate,
        momentum=params['momentum'],
        nesterov=nesterov)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp')
    rho = params.get('decay', None) or params.get('rho', 0.9)
    momentum = params.get('momentum', 0.9)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf_keras.optimizers.legacy.RMSprop(
        learning_rate=base_learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam')
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf_keras.optimizers.legacy.Adam(
        learning_rate=base_learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)
  elif optimizer_name == 'adamw':
    logging.info('Using AdamW')
    weight_decay = params.get('weight_decay', 0.01)
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = legacy_adamw.AdamWeightDecay(
        learning_rate=base_learning_rate,
        weight_decay_rate=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
  else:
    raise ValueError('Unknown optimizer %s' % optimizer_name)

  if params.get('lookahead', None):
    logging.info('Using lookahead optimizer.')
    optimizer = Lookahead(optimizer)

  # Moving average should be applied last, as it's applied at test time
  moving_average_decay = params.get('moving_average_decay', 0.)
  if moving_average_decay is not None and moving_average_decay > 0.:
    if model is None:
      raise ValueError(
          '`model` must be provided if using `ExponentialMovingAverage`.')
    logging.info('Including moving average decay.')
    optimizer = optimization.ExponentialMovingAverage(
        optimizer=optimizer, average_decay=moving_average_decay)
    optimizer.shadow_copy(model)
  return optimizer


def build_learning_rate(params: base_configs.LearningRateConfig,
                        batch_size: Optional[int] = None,
                        train_epochs: Optional[int] = None,
                        train_steps: Optional[int] = None):
  """Build the learning rate given the provided configuration."""
  decay_type = params.name
  base_lr = params.initial_lr
  decay_rate = params.decay_rate
  if params.decay_epochs is not None:
    decay_steps = params.decay_epochs * train_steps
  else:
    decay_steps = 0
  if params.warmup_epochs is not None:
    warmup_steps = params.warmup_epochs * train_steps
  else:
    warmup_steps = 0

  lr_multiplier = params.scale_by_batch_size

  if lr_multiplier and lr_multiplier > 0:
    # Scale the learning rate based on the batch size and a multiplier
    base_lr *= lr_multiplier * batch_size
    logging.info(
        'Scaling the learning rate based on the batch size '
        'multiplier. New base_lr: %f', base_lr)

  if decay_type == 'exponential':
    logging.info(
        'Using exponential learning rate with: '
        'initial_learning_rate: %f, decay_steps: %d, '
        'decay_rate: %f', base_lr, decay_steps, decay_rate)
    lr = tf_keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=params.staircase)
  elif decay_type == 'stepwise':
    steps_per_epoch = params.examples_per_epoch // batch_size
    boundaries = [boundary * steps_per_epoch for boundary in params.boundaries]
    multipliers = [batch_size * multiplier for multiplier in params.multipliers]
    logging.info(
        'Using stepwise learning rate. Parameters: '
        'boundaries: %s, values: %s', boundaries, multipliers)
    lr = tf_keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=multipliers)
  elif decay_type == 'cosine_with_warmup':
    lr = learning_rate.CosineDecayWithWarmup(
        batch_size=batch_size,
        total_steps=train_epochs * train_steps,
        warmup_steps=warmup_steps)
  if warmup_steps > 0:
    if decay_type not in ['cosine_with_warmup']:
      logging.info('Applying %d warmup steps to the learning rate',
                   warmup_steps)
      lr = learning_rate.WarmupDecaySchedule(
          lr, warmup_steps, warmup_lr=base_lr)
  return lr
