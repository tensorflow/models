# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Optimizer factory for vision tasks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from typing import Any, Dict, Text, List

from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa

from official.vision.image_classification import learning_rate
from official.vision.image_classification.configs import base_configs

# pylint: disable=protected-access


class MovingAverage(tf.keras.optimizers.Optimizer):
  """Optimizer that computes a moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate)
  opt = MovingAverage(opt)

  opt.shadow_copy(model)
  ```

  At test time, swap the shadow variables to evaluate on the averaged weights:
  ```python
  opt.swap_weights()
  # Test eval the model here
  opt.swap_weights()
  ```
  """

  def __init__(self,
               optimizer: tf.keras.optimizers.Optimizer,
               average_decay: float = 0.99,
               start_step: int = 0,
               dynamic_decay: bool = True,
               name: Text = 'moving_average',
               **kwargs):
    """Construct a new MovingAverage optimizer.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      average_decay: float. Decay to use to maintain the moving averages
        of trained variables.
      start_step: int. What step to start the moving average.
      dynamic_decay: bool. Whether to change the decay based on the number
        of optimizer updates. Decay will start at 0.1 and gradually increase
        up to `average_decay` after each optimizer update. This behavior is
        similar to `tf.train.ExponentialMovingAverage` in TF 1.x.
      name: Optional name for the operations created when applying
        gradients. Defaults to "moving_average".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}.
    """
    super(MovingAverage, self).__init__(name, **kwargs)
    self._optimizer = optimizer
    self._average_decay = average_decay
    self._start_step = tf.constant(start_step, tf.float32)
    self._dynamic_decay = dynamic_decay

  def shadow_copy(self, model: tf.keras.Model):
    """Creates shadow variables for the given model weights."""
    for var in model.weights:
      self.add_slot(var, 'average', initializer='zeros')
    self._average_weights = [
        self.get_slot(var, 'average') for var in model.weights
    ]
    self._model_weights = model.weights

  @property
  def has_shadow_copy(self):
    """Whether this optimizer has created shadow variables."""
    return self._model_weights is not None

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access

  def apply_gradients(self, grads_and_vars, name: Text = None):
    result = self._optimizer.apply_gradients(grads_and_vars, name)
    self.update_average(self._optimizer.iterations)
    return result

  @tf.function
  def update_average(self, step: tf.Tensor):
    step = tf.cast(step, tf.float32)
    if step < self._start_step:
      decay = tf.constant(0., tf.float32)
    elif self._dynamic_decay:
      decay = step - self._start_step
      decay = tf.minimum(self._average_decay, (1. + decay) / (10. + decay))
    else:
      decay = self._average_decay

    def _apply_moving(v_moving, v_normal):
      diff = v_moving - v_normal
      v_moving.assign_sub(tf.cast(1. - decay, v_moving.dtype) * diff)
      return v_moving

    def _update(strategy, v_moving_and_v_normal):
      for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(_update, args=(zip(self._average_weights,
                                             self._model_weights),))

  def swap_weights(self):
    """Swap the average and moving weights.

    This is a convenience method to allow one to evaluate the averaged weights
    at test time. Loads the weights stored in `self._average` into the model,
    keeping a copy of the original model weights. Swapping twice will return
    the original weights.
    """
    if tf.distribute.in_cross_replica_context():
      strategy = tf.distribute.get_strategy()
      strategy.run(self._swap_weights, args=())
    else:
      raise ValueError('Swapping weights must occur under a '
                       'tf.distribute.Strategy')

  @tf.function
  def _swap_weights(self):
    def fn_0(a, b):
      a.assign_add(b)
      return a
    def fn_1(b, a):
      b.assign(a - b)
      return b
    def fn_2(a, b):
      a.assign_sub(b)
      return a

    def swap(strategy, a_and_b):
      """Swap `a` and `b` and mirror to all devices."""
      for a, b in a_and_b:
        strategy.extended.update(a, fn_0, args=(b,))  # a = a + b
        strategy.extended.update(b, fn_1, args=(a,))  # b = a - b
        strategy.extended.update(a, fn_2, args=(b,))  # a = a - b

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(
        swap, args=(zip(self._average_weights, self._model_weights),))

  def assign_average_vars(self, var_list: List[tf.Variable]):
    """Assign variables in var_list with their respective averages.

    Args:
      var_list: List of model variables to be assigned to their average.
    Returns:
      assign_op: The op corresponding to the assignment operation of
        variables to their average.
    """
    assign_op = tf.group([
        var.assign(self.get_slot(var, 'average')) for var in var_list
        if var.trainable
    ])
    return assign_op

  def _create_hypers(self):
    self._optimizer._create_hypers()  # pylint: disable=protected-access

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  @property
  def weights(self):
    # return self._weights + self._optimizer.weights
    return self._optimizer.weights

  @property
  def lr(self):
    return self._optimizer._get_hyper('learning_rate')

  @lr.setter
  def lr(self, lr):
    self._optimizer._set_hyper('learning_rate', lr)

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')

  @learning_rate.setter
  def learning_rate(self, learning_rate):  # pylint: disable=redefined-outer-name
    self._optimizer._set_hyper('learning_rate', learning_rate)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse(grad, var, indices)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse_duplicate_indices(
        grad, var, indices)

  def get_config(self):
    config = {
        'optimizer': tf.keras.optimizers.serialize(self._optimizer),
        'average_decay': self._average_decay,
        'start_step': self._start_step,
        'dynamic_decay': self._dynamic_decay,
    }
    base_config = super(MovingAverage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    return cls(optimizer, **config)


def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any],
    model: tf.keras.Model = None):
  """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples:
      sgd, momentum, rmsprop.
    base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params.
      This should contain optimizer specific parameters such as
      `base_learning_rate`, `decay`, etc.
    model: The `tf.keras.Model`. This is used for the shadow copy if using
      `MovingAverage`.

  Returns:
    A tf.keras.Optimizer.

  Raises:
    ValueError if the provided optimizer_name is not supported.

  """
  optimizer_name = optimizer_name.lower()
  logging.info('Building %s optimizer with params %s', optimizer_name, params)

  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        nesterov=nesterov)
  elif optimizer_name == 'momentum':
    logging.info('Using momentum optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        momentum=params['momentum'],
                                        nesterov=nesterov)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp')
    rho = params.get('decay', None) or params.get('rho', 0.9)
    momentum = params.get('momentum', 0.9)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate,
                                            rho=rho,
                                            momentum=momentum,
                                            epsilon=epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam')
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon)
  elif optimizer_name == 'adamw':
    logging.info('Using AdamW')
    weight_decay = params.get('weight_decay', 0.01)
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
                                     learning_rate=base_learning_rate,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)
  else:
    raise ValueError('Unknown optimizer %s' % optimizer_name)

  if params.get('lookahead', None):
    logging.info('Using lookahead optimizer.')
    optimizer = tfa.optimizers.Lookahead(optimizer)

  # Moving average should be applied last, as it's applied at test time
  moving_average_decay = params.get('moving_average_decay', 0.)
  if moving_average_decay is not None and moving_average_decay > 0.:
    if model is None:
      raise ValueError('`model` must be provided if using `MovingAverage`.')
    logging.info('Including moving average decay.')
    optimizer = MovingAverage(
        optimizer=optimizer,
        average_decay=moving_average_decay)
    optimizer.shadow_copy(model)
  return optimizer


def build_learning_rate(params: base_configs.LearningRateConfig,
                        batch_size: int = None,
                        train_epochs: int = None,
                        train_steps: int = None):
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
    logging.info('Scaling the learning rate based on the batch size '
                 'multiplier. New base_lr: %f', base_lr)

  if decay_type == 'exponential':
    logging.info('Using exponential learning rate with: '
                 'initial_learning_rate: %f, decay_steps: %d, '
                 'decay_rate: %f', base_lr, decay_steps, decay_rate)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=params.staircase)
  elif decay_type == 'piecewise_constant_with_warmup':
    logging.info('Using Piecewise constant decay with warmup. '
                 'Parameters: batch_size: %d, epoch_size: %d, '
                 'warmup_epochs: %d, boundaries: %s, multipliers: %s',
                 batch_size, params.examples_per_epoch,
                 params.warmup_epochs, params.boundaries,
                 params.multipliers)
    lr = learning_rate.PiecewiseConstantDecayWithWarmup(
        batch_size=batch_size,
        epoch_size=params.examples_per_epoch,
        warmup_epochs=params.warmup_epochs,
        boundaries=params.boundaries,
        multipliers=params.multipliers)
  elif decay_type == 'cosine_with_warmup':
    lr = learning_rate.CosineDecayWithWarmup(
        batch_size=batch_size,
        total_steps=train_epochs * train_steps,
        warmup_steps=warmup_steps)
  if warmup_steps > 0:
    if decay_type not in [
        'piecewise_constant_with_warmup', 'cosine_with_warmup'
    ]:
      logging.info('Applying %d warmup steps to the learning rate',
                   warmup_steps)
      lr = learning_rate.WarmupDecaySchedule(lr, warmup_steps)
  return lr
