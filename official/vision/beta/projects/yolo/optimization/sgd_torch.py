from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.training import gen_training_ops

import tensorflow as tf
import logging

__all__ = ['SGDTorch']


def _var_key(var):
  """Key for representing a primary variable, for looking up slots.
  In graph mode the name is derived from the var shared name.
  In eager mode the name is derived from the var unique id.
  If distribution strategy exists, get the primary variable first.
  Args:
    var: the variable.
  Returns:
    the unique name of the variable.
  """

  # pylint: disable=protected-access
  # Get the distributed variable if it exists.
  if hasattr(var, "_distributed_container"):
    var = var._distributed_container()
  if var._in_graph_mode:
    return var._shared_name
  return var._unique_id


class SGDTorch(tf.keras.optimizers.Optimizer):
  """Optimizer that computes an exponential moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate)
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

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               weight_decay=0.0,
               learning_rate=0.01,
               momentum=0.0,
               momentum_start=0.0,
               warmup_steps=1000,
               nesterov=False,
               sim_torch=False,
               name="SGD",
               **kwargs):
    super(SGDTorch, self).__init__(name, **kwargs)

    # Create Hyper Params for each group of the LR
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("bias_learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("other_learning_rate", kwargs.get("lr", learning_rate))

    # SGD decay param
    self._set_hyper("decay", self._initial_decay)

    # Weight decay param
    self._weight_decay = weight_decay != 0.0
    self._set_hyper("weight_decay", weight_decay)

    # Enable Momentum
    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)
    self._set_hyper("momentum_start", momentum_start)
    self._set_hyper("warmup_steps", tf.cast(warmup_steps, tf.int32))

    # Enable Nesterov Momentum
    self.nesterov = nesterov

    # Simulate Pytorch Optimizer
    self.sim_torch = sim_torch

    # weights, biases, other
    self._wset = set()
    self._bset = set()
    self._oset = set()
    logging.info(f"Pytorch SGD simulation: ")
    logging.info(f"Weight Decay: {weight_decay}")

  def set_bias_lr(self, lr):
    self._set_hyper("bias_learning_rate", lr)

  def set_other_lr(self, lr):
    self._set_hyper("other_learning_rate", lr)

  def set_params(self, weights, biases, others):
    self._wset = set([_var_key(w) for w in weights])
    self._bset = set([_var_key(b) for b in biases])
    self._oset = set([_var_key(o) for o in others])

    logging.info(
        f"Weights: {len(weights)} Biases: {len(biases)} Others: {len(others)}")
    return

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _get_momentum(self, iteration):
    momentum = self._get_hyper("momentum")
    momentum_start = self._get_hyper("momentum_start")
    momentum_warm_up_steps = tf.cast(
        self._get_hyper("warmup_steps"), iteration.dtype)
    value = tf.cond(
        (iteration - momentum_warm_up_steps) <= 0,
        true_fn=lambda: (momentum_start +
                         (tf.cast(iteration, momentum.dtype) *
                          (momentum - momentum_start) / tf.cast(
                              momentum_warm_up_steps, momentum.dtype))),
        false_fn=lambda: momentum)
    return value

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDTorch, self)._prepare_local(var_device, var_dtype,
                                                   apply_state)
    weight_decay = self._get_hyper("weight_decay")
    apply_state[(var_device,
                 var_dtype)]["weight_decay"] = tf.cast(weight_decay, var_dtype)

    if self._momentum:
      momentum = self._get_momentum(self.iterations)
      momentum = tf.cast(momentum, var_dtype)
      apply_state[(var_device,
                   var_dtype)]["momentum"] = array_ops.identity(momentum)

    bias_lr = self._get_hyper("bias_learning_rate")
    if isinstance(bias_lr, LearningRateSchedule):
      bias_lr = bias_lr(self.iterations)
    bias_lr = tf.cast(bias_lr, var_dtype)
    apply_state[(var_device,
                 var_dtype)]["bias_lr_t"] = array_ops.identity(bias_lr)

    other_lr = self._get_hyper("other_learning_rate")
    if isinstance(other_lr, LearningRateSchedule):
      other_lr = other_lr(self.iterations)
    other_lr = tf.cast(other_lr, var_dtype)
    apply_state[(var_device,
                 var_dtype)]["other_lr_t"] = array_ops.identity(other_lr)

    return apply_state[(var_device, var_dtype)]

  def _apply_tf(self, grad, var, weight_decay, momentum, lr):

    def decay_op(var, learning_rate, wd):
      if self._weight_decay and wd > 0:
        return var.assign_sub(
            learning_rate * var * wd, use_locking=self._use_locking)
      return tf.no_op()

    decay = decay_op(var, lr, weight_decay)
    with tf.control_dependencies([decay]):
      if self._momentum:
        momentum_var = self.get_slot(var, "momentum")
        return gen_training_ops.ResourceApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=lr,
            grad=grad,
            momentum=momentum,
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)
      else:
        return gen_training_ops.ResourceApplyGradientDescent(
            var=var.handle, alpha=lr, delta=grad, use_locking=self._use_locking)

  def _apply(self, grad, var, weight_decay, momentum, lr):
    dparams = grad
    groups = []

    # do not update non-trainable weights
    if not var.trainable:
      return tf.group(*groups)

    if self._weight_decay:
      dparams += (weight_decay * var)

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      momentum_update = momentum_var.assign(
          momentum * momentum_var + dparams, use_locking=self._use_locking)
      groups.append(momentum_update)

      if self.nesterov:
        dparams += (momentum * momentum_update)
      else:
        dparams = momentum_update

    weight_update = var.assign_add(-lr * dparams, use_locking=self._use_locking)
    groups.append(weight_update)
    return tf.group(*groups)

  def _get_vartype(self, var, coefficients):
    if (_var_key(var) in self._wset):
      return True, False, False
    elif (_var_key(var) in self._bset):
      return False, True, False
    return False, False, True

  def _run_sgd(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    weights, bias, others = self._get_vartype(var, coefficients)
    weight_decay = tf.zeros_like(coefficients["weight_decay"])
    lr = coefficients["lr_t"]
    if weights:
      weight_decay = coefficients["weight_decay"]
      lr = coefficients["lr_t"]
    elif bias:
      weight_decay = tf.zeros_like(coefficients["weight_decay"])
      lr = coefficients["bias_lr_t"]
    elif others:
      weight_decay = tf.zeros_like(coefficients["weight_decay"])
      lr = coefficients["other_lr_t"]
    momentum = coefficients["momentum"]

    if self.sim_torch:
      return self._apply(grad, var, weight_decay, momentum, lr)
    else:
      return self._apply_tf(grad, var, weight_decay, momentum, lr)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    return self._run_sgd(grad, var, apply_state=apply_state)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    holder = tf.tensor_scatter_nd_add(
        tf.zeros_like(var), tf.expand_dims(indices, axis = -1), grad) 
    return self._run_sgd(holder, var, apply_state=apply_state)

  def get_config(self):
    config = super(SGDTorch, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "momentum": self._serialize_hyperparameter("momentum"),
        "momentum_start": self._serialize_hyperparameter("momentum_start"),
        "warmup_steps": self._serialize_hyperparameter("warmup_steps"),
        "nesterov": self.nesterov,
    })
    return config

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')
