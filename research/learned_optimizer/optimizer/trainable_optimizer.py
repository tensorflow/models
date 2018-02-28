# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""A base class definition for trainable optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import tensorflow as tf

from tensorflow.python.framework import tensor_shape

OPTIMIZER_SCOPE = "LOL"
_LOCAL_VARIABLE_PREFIX = "local_state_"
_LOCAL_STATE_VARIABLE_COLLECTION = "local_state_collection"
EPSILON = 1e-6


class TrainableOptimizer(tf.train.Optimizer):
  """Base class for trainable optimizers.

  A trainable optimizer is an optimizer that has parameters that can themselves
  be learned (meta-optimized).

  Subclasses must implement:
      _compute_update(self, param, grad, state)
  """

  def __init__(self, name, state_keys, use_attention=False,
               use_log_objective=False, obj_train_max_multiplier=-1,
               use_second_derivatives=True, use_numerator_epsilon=False,
               **kwargs):
    """Initializes the optimizer with the given name and settings.

    Args:
      name: The name string for this optimizer.
      state_keys: The names of any required state variables (list)
      use_attention: Whether this optimizer uses attention (Default: True)
      use_log_objective: Whether this optimizer uses the logarithm of the
          objective when computing the loss (Default: False)
      obj_train_max_multiplier: The maximum multiplier for the increase in the
          objective before meta-training is stopped. If <= 0, meta-training is
          not stopped early. (Default: -1)
      use_second_derivatives: Whether this optimizer uses second derivatives in
          meta-training. This should be set to False if some second derivatives
          in the meta-training problem set are not defined in Tensorflow.
          (Default: True)
      use_numerator_epsilon: Whether to use epsilon in the numerator when
          scaling the problem objective during meta-training. (Default: False)
      **kwargs: Any additional keyword arguments.
    """
    self.use_second_derivatives = use_second_derivatives
    self.state_keys = sorted(state_keys)
    self.use_attention = use_attention
    self.use_log_objective = use_log_objective
    self.obj_train_max_multiplier = obj_train_max_multiplier
    self.use_numerator_epsilon = use_numerator_epsilon

    use_locking = False
    super(TrainableOptimizer, self).__init__(use_locking, name)

  def _create_slots(self, var_list):
    """Creates all slots needed by the variables.

    Args:
      var_list: A list of `Variable` objects.
    """
    for var in var_list:
      init_states = self._initialize_state(var)
      for slot_name in sorted(init_states):
        slot_var_name = "{}_{}".format(self.get_name(), slot_name)
        value = init_states[slot_name]
        self._get_or_make_slot(var, value, slot_name, slot_var_name)

  def _initialize_state(self, var):
    """Initializes any state required for this variable.

    Args:
      var: a tensor containing parameters to be optimized

    Returns:
      state: a dictionary mapping state keys to initial state values (tensors)
    """
    return {}

  def _initialize_global_state(self):
    """Initializes any global state values."""
    return []

  def _apply_common(self, grad, var):
    """Applies the optimizer updates to the variables.

    Note: this should only get called via _apply_dense or _apply_sparse when
    using the optimizer via optimizer.minimize or optimizer.apply_gradients.
    During meta-training, the optimizer.train function should be used to
    construct an optimization path that is differentiable.

    Args:
      grad: A tensor representing the gradient.
      var: A tf.Variable with the same shape as grad.

    Returns:
      update_op: A tensorflow op that assigns new values to the variable, and
          also defines dependencies that update the state variables for the
          optimizer.
    """
    state = {key: self.get_slot(var, key) for key in self.get_slot_names()}
    new_var, new_state = self._compute_update(var, grad, state)
    state_assign_ops = [tf.assign(state_var, new_state[key])
                        for key, state_var in state.items()]
    with tf.control_dependencies(state_assign_ops):
      update_op = var.assign(new_var)

    return update_op

  def _apply_dense(self, grad, var):
    """Adds ops to apply dense gradients to 'var'."""
    return self._apply_common(grad, var)

  def _apply_sparse(self, grad, var):
    """Adds ops to apply sparse gradients to 'var'."""
    return self._apply_common(grad, var)

  def _compute_update(self, param, grad, state):
    """Computes the update step for optimization.

    Args:
      param: A tensor of parameters to optimize.
      grad: The gradient tensor of the objective with respect to the parameters.
          (It has the same shape as param.)
      state: A dictionary containing any extra state required by the optimizer.

    Returns:
      updated_params: The updated parameters.
      updated_state: The dictionary of updated state variable(s).
    """
    raise NotImplementedError

  def _compute_updates(self, params, grads, states, global_state):
    """Maps the compute update functions for each parameter.

    This function can be overriden by a subclass if the subclass wants to
    combine information across the different parameters in the list.

    Args:
      params: A list of parameter tensors.
      grads: A list of gradients corresponding to each parameter.
      states: A list of state variables corresponding to each parameter.
      global_state: A list of global state variables for the problem.

    Returns:
      new_params: The updated parameters.
      new_states: The updated states.
      new_global_state: The updated global state.
      attention_params: A list of attention parameters. This is the same as
          new_params if the optimizer does not use attention.
    """
    # Zip up the arguments to _compute_update.
    args = zip(params, grads, states)

    # Call compute_update on each set of parameter/gradient/state args.
    new_params, new_states = zip(*list(
        itertools.starmap(self._compute_update, args)))

    # Global state is unused in the basic case, just pass it through.
    return list(new_params), list(new_states), global_state, list(new_params)

  def train(self, problem, dataset):
    """Creates graph operations to train the optimizer.

    Args:
      problem: A problem_generator.Problem instance to train on.
      dataset: A datasets.Dataset tuple to use when training.

    Returns:
      meta_objective: A tensorflow operation for computing the meta-objective
      obj_weights: A tensor placeholder for feeding in the objective weights
      obj_values: The subproblem objective values during optimization
      batches: The batch indexes tensor for overriding with feed_dict
      first_unroll: A placeholder signifying if this is a first unroll
        (this will propagate the gradients slightly differently).
      reset_state: A placeholder signifying that the rnn state should be reset.
      output_state: The final state of the optimizer
      init_loop_vars_to_override: Local variables that can be assigned to
        propagate the optimizer and problem state for unrolling
      final_loop_vals: Final values of the loop variables that can be
        assigned to init_loop_vars_to_override.
    """

    # Placeholder for the objective weights
    obj_weights = tf.placeholder(tf.float32)
    num_iter = tf.shape(obj_weights)[0]

    # Unpack the dataset and generate the minibatches for training
    data, labels = dataset
    # Convert the ndarrays to tensors so we can pass them back in via feed_dict
    data = tf.constant(data)
    labels = tf.constant(labels)
    batches = tf.placeholder(tf.int32)
    first_unroll = tf.placeholder_with_default(False, [])
    reset_state = tf.placeholder_with_default(False, [])

    training_output = collections.namedtuple("TrainingOutput",
                                             ["metaobj",
                                              "obj_weights",
                                              "problem_objectives",
                                              "initial_obj",
                                              "batches",
                                              "first_unroll",
                                              "reset_state",
                                              "output_state",
                                              "init_loop_vars",
                                              "output_loop_vars"])

    def loop_body(itr, obj_accum, params, attend_params, flattened_states,
                  global_state, all_obj, unused_init_obj, data,
                  labels, batches):
      """Body of the meta-training while loop for optimizing a sub-problem.

      Args:
        itr: The current meta-training iteration.
        obj_accum: The accumulated objective over all training steps so far.
        params: The parameters of the sub-problem.
        attend_params: The parameters of the sub-problems at the attended
            location.
        flattened_states: The states of the trainable optimizer, sorted and
            flattened into a list (since a while loop can't handle nested lists
            or dictionaries).
        global_state: The global state of the optimizer.
        all_obj: The list of all objective values in the training process.
        unused_init_obj: The initial objective (unused here, but needed in the
            variable list because it's used in a stopping condition in the
            loop_cond.)
        data: The data for this problem.
        labels: The labels corresponding to the data.
        batches: The batch indexes needed for shuffled minibatch creation.

      Returns:
        itr: The updated meta-training iteration.
        obj_accum: The updated accumulated objective.
        params: The new parameters of the sub-problem.
        attend_params: The new parameters of the sub-problems at the attended
            location.
        flattened_states: The new states of the trainable optimizer.
        global_state: The updated global state.
        all_obj: The updates list of all objective values.
        unused_init_obj: The initial objective.
        data: The data for this problem.
        labels: The labels corresponding to the data.
        batches: The batch indexes needed for shuffled minibatch creation.
      """
      batch_indices = tf.gather(batches, itr)
      batch_data = tf.gather(data, batch_indices)
      batch_labels = tf.gather(labels, batch_indices)

      # Compute the objective over the entire dataset (full batch).
      obj = problem.objective(params, data, labels)

      # Compute the gradients on just the current batch
      if self.use_attention:
        current_obj = problem.objective(attend_params, batch_data, batch_labels)
        grads = problem.gradients(current_obj, attend_params)
      else:
        current_obj = problem.objective(params, batch_data, batch_labels)
        grads = problem.gradients(current_obj, params)

      if not self.use_second_derivatives:
        new_grads = []
        for grad in grads:
          if isinstance(grad, tf.IndexedSlices):
            new_grads.append(
                tf.IndexedSlices(tf.stop_gradient(grad.values), grad.indices))
          else:
            new_grads.append(tf.stop_gradient(grad))
        grads = new_grads

      # store the objective value for the entire problem at each iteration
      all_obj = tf.concat([all_obj, tf.reshape(obj, (1,))], 0)

      # accumulate the weighted objective for the entire dataset
      acc = tf.gather(obj_weights, itr) * obj

      obj_accum = tf.add(obj_accum, acc)
      # Set the shape to keep the shape invariant for obj_accum. Without this,
      # the graph builder thinks the tensor shape is unknown on the 2nd iter.
      obj_accum.set_shape([])

      # convert flattened_states to dictionaries
      dict_states = [dict(zip(self.state_keys, flat_state))
                     for flat_state in flattened_states]

      # compute the new parameters and states
      args = (params, grads, dict_states, global_state)
      updates = self._compute_updates(*args)
      new_params, new_states, new_global_state, new_attend_params = updates

      # flatten the states
      new_flattened_states = map(flatten_and_sort, new_states)

      return [itr + 1, obj_accum, new_params, new_attend_params,
              new_flattened_states, new_global_state, all_obj, unused_init_obj,
              data, labels, batches]

    def loop_cond(itr, obj_accum, unused_params, unused_attend_params,
                  unused_flattened_states, unused_global_state, all_obj,
                  init_obj, *args):
      """Termination conditions of the sub-problem optimization loop."""
      del args  # unused

      cond1 = tf.less(itr, num_iter)  # We've run < num_iter times
      cond2 = tf.is_finite(obj_accum)  # The objective is still finite

      if self.obj_train_max_multiplier > 0:
        current_obj = tf.gather(all_obj, itr)
        # Account for negative init_obj too
        max_diff = (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
        max_obj = init_obj + max_diff
        # The objective is a reasonable multiplier of the original objective
        cond3 = tf.less(current_obj, max_obj)

        return tf.logical_and(tf.logical_and(cond1, cond2), cond3,
                              name="training_loop_cond")
      else:
        return tf.logical_and(cond1, cond2, name="training_loop_cond")

    init = self._initialize_training_loop_parameters(
        problem, data, labels, batches, first_unroll, reset_state)
    loop_vars, invariants, initial_obj, init_loop_vars_to_override = init

    loop_output = tf.while_loop(loop_cond, loop_body, loop_vars,
                                swap_memory=True, shape_invariants=invariants)
    meta_obj, problem_objectives = loop_output[1], loop_output[6]

    # The meta objective is normalized by the initial objective at the start of
    # the series of partial unrolls.
    scaled_meta_objective = self.scale_objective(
        meta_obj, problem_objectives, initial_obj)

    final_loop_vals = (
        [initial_obj] + loop_output[2] + loop_output[3] + loop_output[5])
    final_loop_vals.extend(itertools.chain(*loop_output[4]))

    return training_output(scaled_meta_objective,
                           obj_weights,
                           problem_objectives,
                           initial_obj,
                           batches,
                           first_unroll,
                           reset_state,
                           loop_output[4],
                           init_loop_vars_to_override,
                           final_loop_vals)

  def _initialize_training_loop_parameters(
      self, problem, data, labels, batches, first_unroll, reset_state):
    """Initializes the vars and params needed for the training process.

    Args:
      problem: The problem being optimized.
      data: The data for the problem.
      labels: The corresponding labels for the data.
      batches: The indexes needed to create shuffled batches of the data.
      first_unroll: Whether this is the first unroll in a partial unrolling.
      reset_state: Whether RNN state variables should be reset.

    Returns:
      loop_vars: The while loop variables for training.
      invariants: The corresponding variable shapes (required by while loop).
      initial_obj: The initial objective (used later for scaling).
      init_loop_vars_to_override: The loop vars that can be overridden when
          performing training via partial unrolls.
    """
    # Extract these separately so we don't have to make inter-variable
    # dependencies.
    initial_tensors = problem.init_tensors()

    return_initial_tensor_values = first_unroll
    initial_params_vars, initial_params = local_state_variables(
        initial_tensors, return_initial_tensor_values)
    initial_attend_params_vars, initial_attend_params = local_state_variables(
        initial_tensors, return_initial_tensor_values)
    # Recalculate the initial objective for the list on each partial unroll with
    # the new initial_params. initial_obj holds the value from the very first
    # unroll.
    initial_obj_init = problem.objective(initial_params, data, labels)
    return_initial_obj_init = first_unroll
    [initial_obj_var], [initial_obj] = local_state_variables(
        [initial_obj_init], return_initial_obj_init)

    # Initialize the loop variables.
    initial_itr = tf.constant(0, dtype=tf.int32)
    initial_meta_obj = tf.constant(0, dtype=tf.float32)
    # N.B. the use of initial_obj_init here rather than initial_obj
    initial_problem_objectives = tf.reshape(initial_obj_init, (1,))

    # Initialize the extra state.
    initial_state_vars = []
    initial_state = []
    state_shapes = []
    return_initial_state_values = reset_state
    for param in initial_tensors:
      param_state_vars, param_state = local_state_variables(
          flatten_and_sort(self._initialize_state(param)),
          return_initial_state_values)

      initial_state_vars.append(param_state_vars)
      initial_state.append(param_state)
      state_shapes.append([f.get_shape() for f in param_state])

    # Initialize any global (problem-level) state.
    initial_global_state_vars, initial_global_state = local_state_variables(
        self._initialize_global_state(), return_initial_state_values)

    global_shapes = []
    for item in initial_global_state:
      global_shapes.append(item.get_shape())

    # build the list of loop variables:
    loop_vars = [
        initial_itr,
        initial_meta_obj,
        initial_params,         # Local variables.
        initial_attend_params,  # Local variables.
        initial_state,          # Local variables.
        initial_global_state,   # Local variables.
        initial_problem_objectives,
        initial_obj,            # Local variable.
        data,
        labels,
        batches,
    ]

    invariants = [
        initial_itr.get_shape(),
        initial_meta_obj.get_shape(),
        [t.get_shape() for t in initial_params],
        [t.get_shape() for t in initial_attend_params],
        state_shapes,
        global_shapes,
        tensor_shape.TensorShape([None]),   # The problem objectives list grows
        initial_obj.get_shape(),
        tensor_shape.unknown_shape(),  # Placeholder shapes are unknown
        tensor_shape.unknown_shape(),
        tensor_shape.unknown_shape(),
    ]

    # Initialize local variables that we will override with final tensors at the
    # next iter.
    init_loop_vars_to_override = (
        [initial_obj_var] + initial_params_vars + initial_attend_params_vars +
        initial_global_state_vars)
    init_loop_vars_to_override.extend(itertools.chain(*initial_state_vars))

    return loop_vars, invariants, initial_obj, init_loop_vars_to_override

  def scale_objective(self, total_obj, all_objs, initial_obj,
                      obj_scale_eps=1e-6):
    """Normalizes the objective based on the initial objective value.

    Args:
      total_obj: The total accumulated objective over the training run.
      all_objs: A list of all the individual objectives over the training run.
      initial_obj: The initial objective value.
      obj_scale_eps: The epsilon value to use in computations for stability.

    Returns:
      The scaled objective as a single value.
    """
    if self.use_log_objective:
      if self.use_numerator_epsilon:
        scaled_problem_obj = ((all_objs + obj_scale_eps) /
                              (initial_obj + obj_scale_eps))
        log_scaled_problem_obj = tf.log(scaled_problem_obj)
      else:
        scaled_problem_obj = all_objs / (initial_obj + obj_scale_eps)
        log_scaled_problem_obj = tf.log(scaled_problem_obj + obj_scale_eps)
      return tf.reduce_mean(log_scaled_problem_obj)
    else:
      return total_obj / (initial_obj + obj_scale_eps)


def local_state_variables(init_values, return_init_values):
  """Create local variables initialized from init_values.

  This will create local variables from a list of init_values. Each variable
  will be named based on the value's shape and dtype.

  As a convenience, a boolean tensor allows you to return value from
  the created local variable or from the original init value.

  Args:
    init_values: iterable of tensors
    return_init_values: boolean tensor

  Returns:
    local_vars: list of the created local variables.
    vals: if return_init_values is true, then this returns the values of
      init_values. Otherwise it returns the values of the local_vars.
  """
  if not init_values:
    return [], []

  # This generates a harmless warning when saving the metagraph.
  variable_use_count = tf.get_collection_ref(_LOCAL_STATE_VARIABLE_COLLECTION)
  if not variable_use_count:
    variable_use_count.append(collections.defaultdict(int))
  variable_use_count = variable_use_count[0]

  local_vars = []
  with tf.variable_scope(OPTIMIZER_SCOPE):
    # We can't use the init_value as an initializer as init_value may
    # itself depend on some problem variables. This would produce
    # inter-variable initialization order dependence which TensorFlow
    # sucks at making easy.
    for init_value in init_values:
      name = create_local_state_variable_name(init_value)
      unique_name = name + "_" + str(variable_use_count[name])
      variable_use_count[name] += 1
      # The overarching idea here is to be able to reuse variables between
      # different sessions on the same TensorFlow master without errors. By
      # uniquifying based on the type and name we mirror the checks made inside
      # TensorFlow, while still allowing some memory reuse. Ultimately this is a
      # hack due to the broken Session.reset().
      local_vars.append(
          tf.get_local_variable(
              unique_name,
              initializer=tf.zeros(
                  init_value.get_shape(), dtype=init_value.dtype)))

  # It makes things a lot simpler if we use the init_value the first
  # iteration, instead of the variable itself. It allows us to propagate
  # gradients through it as well as simplifying initialization. The variable
  # ends up assigned to after the first iteration.
  vals = tf.cond(return_init_values, lambda: init_values, lambda: local_vars)
  if len(init_values) == 1:
    # tf.cond extracts elements from singleton lists.
    vals = [vals]
  return local_vars, vals


def create_local_state_variable_name(tensor):
  """Create a name of the variable based on its type and shape."""
  if not tensor.get_shape().is_fully_defined():
    raise ValueError("Need a fully specified shape to create a local variable.")

  return (_LOCAL_VARIABLE_PREFIX + "_".join(
      map(str, tensor.get_shape().as_list())) + "_" + tensor.dtype.name)


def is_local_state_variable(op):
  """Returns if this op is a local state variable created for training."""
  return op.node_def.op in ["Variable", "VariableV2"] and op.name.startswith(
      OPTIMIZER_SCOPE + "/" + _LOCAL_VARIABLE_PREFIX)


def flatten_and_sort(dictionary):
  """Flattens a dictionary into a list of values sorted by the keys."""
  return [dictionary[k] for k in sorted(dictionary.keys())]
