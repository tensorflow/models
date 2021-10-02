from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.training import gen_training_ops

import tensorflow as tf
import re
import logging

try:
  from keras.optimizer_v2.optimizer_v2 import _var_key
except:
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
  """Optimizer that simulates the SGD module used in pytorch. 
  
  
  For details on the differences between the original SGD implemention and the 
  one in pytorch: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.
  This optimizer also allow for the usage of a momentum warmup along side a 
  learning rate warm up, though using this is not required. 

  Example of usage for training:
  ```python
  opt = SGDTorch(learning_rate)

  # Models must implement a method to iterate all model.trainable_variables
  # and split the variables by key into the weights, biases, and others.
  # Weight decay wil be applied to all variables in the weights group. Bias 
  # and others are included as a way to proved alternate LR scedules to various 
  # paramter groups. An example of this variable search can be found in 
  # official/vision/beta/projects/yolo/modeling/yolo_model.py.

  optimizer.search_and_set_variable_groups(model.trainable_variables)

  # if the learning rate schedule on the biases are different. if lr is not set 
  # the default schedule used for weights will be used on the biases. 
  opt.set_bias_lr(<lr schedule>)

  # if the learning rate schedule on the others are different. if lr is not set 
  # the default schedule used for weights will be used on the biases. 
  opt.set_other_lr(<lr schedule>)
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
               weight_keys=["kernel", "weight"],
               bias_keys=["bias", "beta"],
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
    self._weight_keys = weight_keys
    self._bias_keys = bias_keys
    self._variables_set = False
    self._wset = set()
    self._bset = set()
    self._oset = set()

    if self.sim_torch:
      logging.info(f"Pytorch SGD simulation: ")
    logging.info(f"Weight Decay: {weight_decay}")

  def set_bias_lr(self, lr):
    self._set_hyper("bias_learning_rate", lr)

  def set_other_lr(self, lr):
    self._set_hyper("other_learning_rate", lr)

  def _search(self, var, keys):
    """Search all all keys for matches. Return True on match."""
    for r in keys:
      if re.search(r, var.name) is not None:
        return True
    return False 

  def search_and_set_variable_groups(self, variables):
    """Search all variable for matches at each group. 

    Args:
      variables: List[tf.Variable] from model.trainable_variables
    """
    weights = []
    biases = []
    others = []

    for var in variables:
      # search for weights
      if self.search(var, self._weight_keys):
        weights.append(var)
        continue
      # search for biases
      if self.search(var, self._bias_keys):
        biases.append(var)
        continue
      # if all searches fail, add to other group
      others.append(var)

    self.set_variable_groups(weights, biases, others)
    return weights, biases, others

  def set_variable_groups(self, weights, biases, others):
    """Alterantive to search and set allowing user to manually set each group.

    This method is allows the user to bypass the weights, biases and others 
    search by key, and manually set the values for each group. This is the 
    safest alternative in cases where the variables cannot be grouped by 
    searching their names. 

    Args:
      weights: List[tf.Variable] from model.trainable_variables
      biases: List[tf.Variable] from model.trainable_variables
      others: List[tf.Variable] from model.trainable_variables
    """
    if self._variables_set:
      logging.warning("set_variable_groups has been called again indicating"
                      "that the variable groups have already been set, they" 
                      "will be updated.")
    self._wset.update(set([_var_key(w) for w in weights]))
    self._bset.update(set([_var_key(b) for b in biases]))
    self._oset.update(set([_var_key(o) for o in others]))

    # Log the number of objects in each group. 
    logging.info(
        f"Weights: {len(self._wset)} Biases: {len(self._bset)} Others: {len(self._oset)}")
    self._variables_set = True
    return

  def _get_variable_group(self, var, coefficients):
    if self._variables_set:
      # check which groups hold which varaibles, preset.
      if (_var_key(var) in self._wset):
        return True, False, False
      elif (_var_key(var) in self._bset):
        return False, True, False
    else:
      # search the variables at run time.
      if self._search(var, self._weight_keys):
        return True, False, False
      elif self._search(var, self._bias_keys):
        return False, True, False
    return False, False, True

  def _create_slots(self, var_list):
    """Create a momentum variable for each variable."""
    if self._momentum:
      for var in var_list:
        # check if trainable to support GPU EMA. 
        if var.trainable: 
          self.add_slot(var, "momentum")

  def _get_momentum(self, iteration):
    """Get the momentum value."""
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
    """Uses Tensorflow Optimizer with Weight decay SGDW."""
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
    """Uses Pytorch Optimizer with Weight decay SGDW."""
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

  def _run_sgd(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    weights, bias, others = self._get_variable_group(var, coefficients)
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
