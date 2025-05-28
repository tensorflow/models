# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Exponential moving average optimizer."""

from typing import List, Optional

import tensorflow as tf, tf_keras

# pylint: disable=protected-access


def maybe_merge_call(fn, strategy, *args, **kwargs):
  """Maybe invoke `fn` via `merge_call` which may or may not be fulfilled.

  The caller of this utility function requests to invoke `fn` via `merge_call`
  at `tf.distribute.Strategy`'s best efforts. It is `tf.distribute`'s internal
  whether the request is honored, depending on the `Strategy`. See
  `tf.distribute.ReplicaContext.merge_call()` for more information.

  This is adapted from tensorflow/python/distribute/merge_call_interim.py.

  Args:
    fn: the function to be invoked.
    strategy: the `tf.distribute.Strategy` to call `fn` with.
    *args: the positional arguments to be passed in to `fn`.
    **kwargs: the keyword arguments to be passed in to `fn`.

  Returns:
    The return value of the `fn` call.
  """
  if strategy.extended._use_merge_call():
    return tf.distribute.get_replica_context().merge_call(
        fn, args=args, kwargs=kwargs
    )
  else:
    return fn(strategy, *args, **kwargs)


class ExponentialMovingAverage(tf_keras.optimizers.legacy.Optimizer):
  """Optimizer that computes an exponential moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf_keras.optimizers.SGD(learning_rate)
  opt = ExponentialMovingAverage(opt)

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
               optimizer: tf_keras.optimizers.Optimizer,
               trainable_weights_only: bool = True,
               average_decay: float = 0.99,
               start_step: int = 0,
               dynamic_decay: bool = True,
               name: str = 'ExponentialMovingAverage',
               **kwargs):
    """Construct a new ExponentialMovingAverage optimizer.

    Args:
      optimizer: `tf_keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      trainable_weights_only: 'bool', if True, only model trainable weights will
        be updated. Otherwise, all model weights will be updated. This mainly
        affects batch normalization parameters.
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
    super().__init__(name, **kwargs)
    self._average_decay = average_decay
    self._trainable_weights_only = trainable_weights_only
    self._start_step = tf.constant(start_step, tf.float32)
    self._dynamic_decay = dynamic_decay
    self._optimizer = optimizer
    self._track_trackable(self._optimizer, 'ema_base_optimizer')
    self._average_weights = None
    self._model_weights = None

  def shadow_copy(self, model: tf_keras.Model):
    """Creates shadow variables for the given model weights."""

    if self._trainable_weights_only:
      self._model_weights = model.trainable_variables
    else:
      self._model_weights = model.variables
    for var in self._model_weights:
      self.add_slot(var, 'average', initializer='zeros')

    self._average_weights = [
        self.get_slot(var, 'average') for var in self._model_weights
    ]

  @property
  def has_shadow_copy(self):
    """Whether this optimizer has created shadow variables."""
    return self._model_weights is not None and self._average_weights is not None

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access

  def apply_gradients(self, grads_and_vars, name: Optional[str] = None):
    result = self._optimizer.apply_gradients(grads_and_vars, name)
    maybe_merge_call(self.update_average, tf.distribute.get_strategy())
    return result

  @tf.function
  def update_average(self, strategy):
    # Compute current decay value.
    step = tf.cast(self.iterations, tf.float32)
    if step < self._start_step:
      decay = tf.constant(0., tf.float32)
    elif self._dynamic_decay:
      decay = step - self._start_step
      decay = tf.minimum(self._average_decay, (1. + decay) / (10. + decay))
    else:
      decay = self._average_decay

    def _apply_moving(average, normal):
      diff = average - normal
      average.assign_sub(tf.cast(1.0 - decay, average.dtype) * diff)
      return average

    # Update moving average with the latest value.
    for average, normal in zip(self._average_weights, self._model_weights):
      strategy.extended.update(
          average, _apply_moving, args=(normal,), group=False
      )

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
      raise ValueError(
          'Swapping weights must occur under a tf.distribute.Strategy.'
      )

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

    def _swap(strategy, a_and_b):
      """Swap `a` and `b` and mirror to all devices."""
      for a, b in a_and_b:
        strategy.extended.update(a, fn_0, args=(b,))  # a = a + b
        strategy.extended.update(b, fn_1, args=(a,))  # b = a - b
        strategy.extended.update(a, fn_2, args=(b,))  # a = a - b

    # Use merge_call if requested by strategy and always for TPUStrategy as
    # the use of merge_call is not recommended and deprecated for other
    # strategies such as mirrored strategy (MS) and multi-worker mirrored
    # strategy (MWMS) if nccl/collective_ops are used, which can operate in
    # pure replica context.
    strategy = tf.distribute.get_strategy()
    if isinstance(strategy, tf.distribute.TPUStrategy):
      maybe_merge_call(
          _swap,
          strategy,
          zip(self._average_weights, self._model_weights),
      )
    else:
      _swap(
          strategy,
          zip(self._average_weights, self._model_weights),
      )

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

  def variables(self):
    return self._weights + [self.iterations]

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
        'optimizer': tf_keras.optimizers.serialize(self._optimizer),
        'average_decay': self._average_decay,
        'start_step': self._start_step,
        'dynamic_decay': self._dynamic_decay,
    }
    base_config = super(ExponentialMovingAverage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf_keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    return cls(optimizer, **config)
