# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Builds a DRAGNN graph for local training."""

import collections
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.platform import tf_logging as logging

from dragnn.protos import spec_pb2
from dragnn.python import component
from dragnn.python import composite_optimizer
from dragnn.python import dragnn_ops
from syntaxnet.util import check

try:
  tf.NotDifferentiable('ExtractFixedFeatures')
except KeyError as e:
  logging.info(str(e))


def _validate_grid_point(hyperparams, is_sub_optimizer=False):
  """Validates that a grid point's configuration is reasonable.

  Args:
    hyperparams (spec_pb2.GridPoint): Grid point to validate.
    is_sub_optimizer (bool): Whether this optimizer is a sub-optimizer of
      a composite optimizer.

  Raises:
    ValueError: If the grid point is not valid.
  """
  valid_methods = ('gradient_descent', 'adam', 'lazyadam', 'momentum',
                   'composite')
  if hyperparams.learning_method not in valid_methods:
    raise ValueError('Unknown learning method (optimizer)')

  if is_sub_optimizer:
    for base_only_field in ('decay_steps', 'decay_base', 'decay_staircase'):
      if hyperparams.HasField(base_only_field):
        raise ValueError('Field {} is not valid for sub-optimizers of a '
                         'composite optimizer.'.format(base_only_field))

  if hyperparams.learning_method == 'composite':
    spec = hyperparams.composite_optimizer_spec
    if spec.switch_after_steps < 1:
      raise ValueError('switch_after_steps {} not valid for composite '
                       'optimizer!'.format(spec.switch_after_steps))
    for sub_optimizer in (spec.method1, spec.method2):
      _validate_grid_point(sub_optimizer, is_sub_optimizer=True)


def _create_learning_rate(hyperparams, step_var):
  """Creates learning rate var, with decay and switching for CompositeOptimizer.

  Args:
    hyperparams: a GridPoint proto containing optimizer spec, particularly
      learning_method to determine optimizer class to use.
    step_var: tf.Variable, global training step.

  Raises:
    ValueError: If the composite optimizer is set, but not correctly configured.

  Returns:
    a scalar `Tensor`, the learning rate based on current step and hyperparams.
  """
  if hyperparams.learning_method != 'composite':
    base_rate = hyperparams.learning_rate
    adjusted_steps = step_var
  else:
    spec = hyperparams.composite_optimizer_spec
    switch = tf.less(step_var, spec.switch_after_steps)
    base_rate = tf.cond(switch, lambda: tf.constant(spec.method1.learning_rate),
                        lambda: tf.constant(spec.method2.learning_rate))
    if spec.reset_learning_rate:
      adjusted_steps = tf.cond(switch, lambda: step_var,
                               lambda: step_var - spec.switch_after_steps)
    else:
      adjusted_steps = step_var

  return tf.train.exponential_decay(
      learning_rate=base_rate,
      global_step=adjusted_steps,
      decay_steps=hyperparams.decay_steps,
      decay_rate=hyperparams.decay_base,
      staircase=hyperparams.decay_staircase)


def _create_optimizer(hyperparams, learning_rate_var, step_var=None):
  """Creates an optimizer object for a given spec, learning rate and step var.

  Args:
    hyperparams: a GridPoint proto containing optimizer spec, particularly
      learning_method to determine optimizer class to use.
    learning_rate_var: a `tf.Tensor`, the learning rate.
    step_var: a `tf.Variable`, global training step.

  Returns:
    a `tf.train.Optimizer` object that was built.
  """
  if hyperparams.learning_method == 'gradient_descent':
    return tf.train.GradientDescentOptimizer(
        learning_rate_var, use_locking=True)
  elif hyperparams.learning_method == 'adam':
    return tf.train.AdamOptimizer(
        learning_rate_var,
        beta1=hyperparams.adam_beta1,
        beta2=hyperparams.adam_beta2,
        epsilon=hyperparams.adam_eps,
        use_locking=True)
  elif hyperparams.learning_method == 'lazyadam':
    return tf.contrib.opt.LazyAdamOptimizer(
        learning_rate_var,
        beta1=hyperparams.adam_beta1,
        beta2=hyperparams.adam_beta2,
        epsilon=hyperparams.adam_eps,
        use_locking=True)
  elif hyperparams.learning_method == 'momentum':
    return tf.train.MomentumOptimizer(
        learning_rate_var, hyperparams.momentum, use_locking=True)
  elif hyperparams.learning_method == 'composite':
    spec = hyperparams.composite_optimizer_spec
    optimizer1 = _create_optimizer(spec.method1, learning_rate_var, step_var)
    optimizer2 = _create_optimizer(spec.method2, learning_rate_var, step_var)
    if step_var is None:
      logging.fatal('step_var is required for CompositeOptimizer')
    switch = tf.less(step_var, spec.switch_after_steps)
    return composite_optimizer.CompositeOptimizer(
        optimizer1, optimizer2, switch, use_locking=True)
  else:
    logging.fatal('Unknown learning method (optimizer)')


class MasterBuilder(object):
  """A builder for a DRAGNN stack of models.

  This class is the major factory for all DRAGNN models. It provides
  common hooks to build training and evaluation targets from a single
  MasterSpec and hyperparameter configuration.

  The key concept is as follows: to execute a DRAGNN graph, one needs
  two stateful pieces:

    1. A handle to a C++ dragnn state, managed outside of TensorFlow and
       accesssed via the custom dragnn ops.
    2. A set of StoredActivations, one for each component, that contain network
       activations that can be used across components.

  TODO(googleuser): Update these comments to be accurate.
  Both of these can be handled automatically "under-the-hood" by the
  MasterBuilder API. For #1, the key consideration is that each C++
  ComputeSession is allocated statically, meaning memory is shared
  across different tensorflow::Session invocations. ComputeSessions are
  allocated from pools. The `pool_scope` identifies the pool, unique to this
  MasterBuilder, from which the ComputeSession is allocated. From there,
  GetSession takes care of handing out ComputeSessions with unique handles.
  Each ComputeSession can then be run concurrently.

  Attributes:
    spec: the MasterSpec proto.
    hyperparams: the GridPoint proto containing hyperparameters.
    pool_scope: string identifier for the ComputeSession pool to use.
    components: a list of ComponentBuilders in the order they are defined
      in the MasterSpec.
    lookup_component: a dictionary to lookup ComponentBuilders by name.
    optimizer: handle to the tf.train Optimizer object used to train this model.
    master_vars: dictionary of globally shared tf.Variable objects (e.g.
      the global training step and learning rate.)
  """

  def __init__(self, master_spec, hyperparam_config=None, pool_scope='shared'):
    """Initializes the MasterBuilder from specifications.

    During construction, all components are initialized along with their
    parameter tf.Variables.

    Args:
      master_spec: dragnn.MasterSpec proto.
      hyperparam_config: dragnn.GridPoint proto specifying hyperparameters.
        Defaults to empty specification.
      pool_scope: string identifier for the compute session pool to use.

    Raises:
      ValueError: if a component is not found in the registry.
    """
    self.spec = master_spec
    self.hyperparams = (spec_pb2.GridPoint()
                        if hyperparam_config is None else hyperparam_config)
    _validate_grid_point(self.hyperparams)
    self.pool_scope = pool_scope

    # Set the graph-level random seed before creating the Components so the ops
    # they create will use this seed.
    tf.set_random_seed(hyperparam_config.seed)

    # Construct all utility class and variables for each Component.
    self.components = []
    self.lookup_component = {}
    for component_spec in master_spec.component:
      component_type = component_spec.component_builder.registered_name

      # Raises ValueError if not found.
      comp = component.ComponentBuilderBase.Create(component_type, self,
                                                   component_spec)

      self.lookup_component[comp.name] = comp
      self.components.append(comp)

    # Add global step variable.
    self.master_vars = {}
    with tf.variable_scope('master', reuse=False):
      self.master_vars['step'] = tf.get_variable(
          'step', [], initializer=tf.zeros_initializer(), dtype=tf.int32)
      self.master_vars['learning_rate'] = _create_learning_rate(
          self.hyperparams, self.master_vars['step'])

    # Construct optimizer.
    self.optimizer = _create_optimizer(self.hyperparams,
                                       self.master_vars['learning_rate'],
                                       self.master_vars['step'])

  @property
  def component_names(self):
    return tuple(c.name for c in self.components)

  def _get_compute_session(self):
    """Returns a new ComputeSession handle."""
    return dragnn_ops.get_session(
        self.pool_scope,
        master_spec=self.spec.SerializeToString(),
        grid_point=self.hyperparams.SerializeToString(),
        name='GetSession')

  def _get_session_with_reader(self, enable_tracing):
    """Utility to create ComputeSession management ops.

    Creates a new ComputeSession handle and provides the following
    named nodes:

    ComputeSession/InputBatch -- a placeholder for attaching a string
      specification for AttachReader.
    ComputeSession/AttachReader -- the AttachReader op.

    Args:
      enable_tracing: bool, whether to enable tracing before attaching the data.

    Returns:
      handle: handle to a new ComputeSession returned by the AttachReader op.
      input_batch: InputBatch placeholder.
    """
    with tf.name_scope('ComputeSession'):
      input_batch = tf.placeholder(
          dtype=tf.string, shape=[None], name='InputBatch')

      # Get the ComputeSession and chain some essential ops.
      handle = self._get_compute_session()
      if enable_tracing:
        handle = dragnn_ops.set_tracing(handle, True)
      handle = dragnn_ops.attach_data_reader(
          handle, input_batch, name='AttachReader')

    return handle, input_batch

  def _outputs_with_release(self, handle, inputs, outputs):
    """Ensures ComputeSession is released before outputs are returned.

    Args:
      handle: Handle to ComputeSession on which all computation until now has
          depended. It will be released and assigned to the output 'run'.
      inputs: list of nodes we want to pass through without any dependencies.
      outputs: list of nodes whose access should ensure the ComputeSession is
          safely released.

    Returns:
      A dictionary of both input and output nodes.
    """
    with tf.control_dependencies(outputs.values()):
      with tf.name_scope('ComputeSession'):
        release_op = dragnn_ops.release_session(handle)
      run_op = tf.group(release_op, name='run')
      for output in outputs:
        with tf.control_dependencies([release_op]):
          outputs[output] = tf.identity(outputs[output], name=output)
    all_nodes = inputs.copy()
    all_nodes.update(outputs)

    # Add an alias for simply running without collecting outputs.
    # Common, for instance, with training.
    all_nodes['run'] = run_op
    return all_nodes

  def build_warmup_graph(self, asset_dir):
    """Builds a warmup graph.

    This graph performs a MasterSpec asset location rewrite via
    SetAssetDirectory, then grabs a ComputeSession and immediately returns it.
    By grabbing a session, we cause the underlying transition systems to cache
    their static data reads.

    Args:
      asset_dir: The base directory to append to all resources.

    Returns:
      A single op suitable for passing to the legacy_init_op of the ModelSaver.
    """
    with tf.control_dependencies([dragnn_ops.set_asset_directory(asset_dir)]):
      session = self._get_compute_session()
      release_op = dragnn_ops.release_session(session)
    return tf.group(release_op, name='run')

  def build_training(self,
                     handle,
                     compute_gradients=True,
                     use_moving_average=False,
                     advance_counters=True,
                     component_weights=None,
                     unroll_using_oracle=None,
                     max_index=-1):
    """Builds a training pipeline.

    Args:
      handle: Handle tensor for the ComputeSession.
      compute_gradients: Whether to generate gradients and an optimizer op.
        When False, build_training will return a 'dry run' training op,
        used normally only for oracle tracing.
      use_moving_average: Whether or not to read from the moving
        average variables instead of the true parameters. Note: it is not
        possible to make gradient updates when this is True.
      advance_counters: Whether or not this loop should increment the
        per-component step counters.
      component_weights: If set, this is a list of relative weights
        each component's cost should get in the pipeline. Defaults to 1.0 for
        each component.
      unroll_using_oracle: If set, this is a list of booleans indicating
        whether or not to use the gold decodings for each component. Defaults
        to True for each component.
      max_index: Training will use only the first max_index components,
        or -1 for all components.

    Returns:
      handle: to the ComputeSession, conditioned on completing training step.
      outputs: a dictionary of useful training tensors.

    Raises:
      IndexError: if max_index is positive but out of bounds.
    """
    check.IsFalse(compute_gradients and use_moving_average,
                  'It is not possible to make gradient updates when reading '
                  'from the moving average variables.')

    self.read_from_avg = use_moving_average
    if max_index < 0:
      max_index = len(self.components)
    else:
      if not 0 < max_index <= len(self.components):
        raise IndexError('Invalid max_index {} for components {}; handle {}'.
                         format(max_index, self.component_names, handle.name))

    # By default, we train every component supervised.
    if not component_weights:
      component_weights = [1] * max_index
    if not unroll_using_oracle:
      unroll_using_oracle = [True] * max_index

    component_weights = component_weights[:max_index]
    total_weight = (float)(sum(component_weights))
    component_weights = [w / total_weight for w in component_weights]

    unroll_using_oracle = unroll_using_oracle[:max_index]

    logging.info('Creating training target:')
    logging.info('\tWeights: %s', component_weights)
    logging.info('\tOracle: %s', unroll_using_oracle)

    metrics_list = []
    cost = tf.constant(0.)
    effective_batch = tf.constant(0)

    avg_ops = []
    params_to_train = []

    network_states = {}
    for component_index in range(0, max_index):
      comp = self.components[component_index]
      network_states[comp.name] = component.NetworkState()

      logging.info('Initializing data for component "%s"', comp.name)
      handle = dragnn_ops.init_component_data(
          handle, beam_size=comp.training_beam_size, component=comp.name)
      # TODO(googleuser): Phase out component.MasterState.
      master_state = component.MasterState(handle,
                                           dragnn_ops.batch_size(
                                               handle, component=comp.name))
      with tf.control_dependencies([handle, cost]):
        args = (master_state, network_states)
        if unroll_using_oracle[component_index]:

          handle, component_cost, component_correct, component_total = (tf.cond(
              comp.training_beam_size > 1,
              lambda: comp.build_structured_training(*args),
              lambda: comp.build_greedy_training(*args)))

        else:
          handle = comp.build_greedy_inference(*args, during_training=True)
          component_cost = tf.constant(0.)
          component_correct, component_total = tf.constant(0), tf.constant(0)

        weighted_component_cost = tf.multiply(
            component_cost,
            tf.constant((float)(component_weights[component_index])),
            name='weighted_component_cost')

        cost += weighted_component_cost
        effective_batch += component_total
        metrics_list += [[component_total], [component_correct]]

        if advance_counters:
          with tf.control_dependencies(
              [comp.advance_counters(component_total)]):
            cost = tf.identity(cost)

        # Keep track of which parameters will be trained, and any moving
        # average updates to apply for these parameters.
        params_to_train += comp.network.params
        if self.hyperparams.use_moving_average:
          avg_ops += comp.avg_ops

    # Concatenate evaluation results
    metrics = tf.concat(metrics_list, 0)

    # If gradient computation is requested, then:
    # 1. compute the gradients,
    # 2. add an optimizer to update the parameters using the gradients,
    # 3. make the ComputeSession handle depend on the optimizer.
    if compute_gradients:
      logging.info('Creating train op with %d variables:\n\t%s',
                   len(params_to_train),
                   '\n\t'.join([x.name for x in params_to_train]))

      grads_and_vars = self.optimizer.compute_gradients(
          cost, var_list=params_to_train)
      clipped_gradients = [(self._clip_gradients(g), v)
                           for g, v in grads_and_vars]
      minimize_op = self.optimizer.apply_gradients(
          clipped_gradients, global_step=self.master_vars['step'])

      if self.hyperparams.use_moving_average:
        with tf.control_dependencies([minimize_op]):
          minimize_op = tf.group(*avg_ops)

      # Make sure all the side-effectful minimizations ops finish before
      # proceeding.
      with tf.control_dependencies([minimize_op]):
        handle = tf.identity(handle)

    # Restore that subsequent builds don't use average by default.
    self.read_from_avg = False

    cost = tf.check_numerics(cost, message='Cost is not finite.')

    # Returns named access to common outputs.
    outputs = {
        'cost': cost,
        'batch': effective_batch,
        'metrics': metrics,
    }
    return handle, outputs

  def _clip_gradients(self, grad):
    """Clips gradients if the hyperparameter `gradient_clip_norm` requires it.

    Sparse tensors, in the form of IndexedSlices returned for the
    gradients of embeddings, require special handling.

    Args:
      grad: Gradient Tensor, IndexedSlices, or None.

    Returns:
      Optionally clipped gradient.
    """
    if grad is not None and self.hyperparams.gradient_clip_norm > 0:
      logging.info('Clipping gradient %s', grad)
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, self.hyperparams.gradient_clip_norm)
        return tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        return tf.clip_by_norm(grad, self.hyperparams.gradient_clip_norm)
    else:
      return grad

  def build_post_restore_hook(self):
    """Builds a graph that should be executed after the restore op.

    This graph is intended to be run once, before the inference pipeline is
    run.

    Returns:
      setup_op - An op that, when run, guarantees all setup ops will run.
    """
    control_ops = []
    for comp in self.components:
      hook = comp.build_post_restore_hook()
      if isinstance(hook, collections.Iterable):
        control_ops.extend(hook)
      else:
        control_ops.append(hook)
    with tf.control_dependencies(control_ops):
      return tf.no_op(name='post_restore_hook_master')

  def build_inference(self, handle, use_moving_average=False):
    """Builds an inference pipeline.

    This always uses the whole pipeline.

    Args:
      handle: Handle tensor for the ComputeSession.
      use_moving_average: Whether or not to read from the moving
        average variables instead of the true parameters. Note: it is not
        possible to make gradient updates when this is True.

    Returns:
      handle: Handle after annotation.
    """
    self.read_from_avg = use_moving_average
    network_states = {}

    for comp in self.components:
      network_states[comp.name] = component.NetworkState()
      handle = dragnn_ops.init_component_data(
          handle, beam_size=comp.inference_beam_size, component=comp.name)
      master_state = component.MasterState(handle,
                                           dragnn_ops.batch_size(
                                               handle, component=comp.name))
      with tf.control_dependencies([handle]):
        handle = comp.build_greedy_inference(master_state, network_states)
      handle = dragnn_ops.write_annotations(handle, component=comp.name)

    self.read_from_avg = False
    return handle

  def add_training_from_config(self,
                               target_config,
                               prefix='train-',
                               trace_only=False,
                               **kwargs):
    """Constructs a training pipeline from a TrainTarget proto.

    This constructs a separately managed pipeline for a given target:
    it has its own ComputeSession, InputSpec placeholder, etc. The ops
    are given standardized names to allow access from the C++ API. It
    passes the values in target_config to build_training() above.

    For the default prefix ('train-'), and a target named 'target', this will
    construct the following targets in the graph:

      train-target/ComputeSession/* (the standard ComputeSession controls)
      train-target/run (handle to a completed training step)
      train-target/metrics (per-decision metrics from gold oracles)
      train-target/cost (total cost across all components)

    Enabling `trace_only` effectively creates a graph that is a 'dry run'.
    There will be no side affects. In addition, the gradients won't be computed
    and the model parameters will not be updated.

    Args:
      target_config: the TrainTarget proto.
      prefix: Preprends target_config.name with this to construct
        a unique identifier.
      trace_only: Enabling this will result in:
          1. Tracing will be enabled for the ComputeSession..
          2. A 'traces' node will be added to the outputs.
          3. Gradients will not be computed.
      **kwargs: Passed on to build_training() above.

    Returns:
      Dictionary of training targets.
    """
    logging.info('Creating new training target '
                 '%s'
                 ' from config: %s', target_config.name, str(target_config))
    scope_id = prefix + target_config.name
    with tf.name_scope(scope_id):
      # Construct training targets. Disable tracing during training.
      handle, input_batch = self._get_session_with_reader(trace_only)

      # If `trace_only` is True, the training graph shouldn't have any
      # side effects. Otherwise, the standard training scenario should
      # generate gradients and update counters.
      handle, outputs = self.build_training(
          handle,
          compute_gradients=not trace_only,
          advance_counters=not trace_only,
          component_weights=target_config.component_weights,
          unroll_using_oracle=target_config.unroll_using_oracle,
          max_index=target_config.max_index,
          **kwargs)
      if trace_only:
        outputs['traces'] = dragnn_ops.get_component_trace(
            handle, component=self.spec.component[-1].name)
      else:
        # Standard training keeps track of the number of training steps.
        outputs['target_step'] = tf.get_variable(
            scope_id + '/TargetStep', [],
            initializer=tf.zeros_initializer(),
            dtype=tf.int32)
        increment_target_step = tf.assign_add(
            outputs['target_step'], 1, use_locking=True)

        with tf.control_dependencies([increment_target_step]):
          handle = tf.identity(handle)

      return self._outputs_with_release(handle, {'input_batch': input_batch},
                                        outputs)

  def add_annotation(self, name_scope='annotation', enable_tracing=False):
    """Adds an annotation pipeline to the graph.

    This will create the following additional named targets by default, for use
    in C++ annotation code (as well as regular ComputeSession targets):
      annotation/ComputeSession/session_id (placeholder for giving unique id)
      annotation/EmitAnnotations (get annotated data)
      annotation/GetComponentTrace (get trace data)
      annotation/SetTracing (sets tracing based on annotation/tracing_on)

    Args:
      name_scope: Scope for the annotation pipeline.
      enable_tracing: Enabling this will result in two things:
          1. Tracing will be enabled during inference.
          2. A 'traces' node will be added to the outputs.

    Returns:
      A dictionary of input and output nodes.
    """
    with tf.name_scope(name_scope):
      handle, input_batch = self._get_session_with_reader(enable_tracing)
      handle = self.build_inference(handle, use_moving_average=True)

      annotations = dragnn_ops.emit_annotations(
          handle, component=self.spec.component[-1].name)
      outputs = {'annotations': annotations}

      if enable_tracing:
        outputs['traces'] = dragnn_ops.get_component_trace(
            handle, component=self.spec.component[-1].name)

      return self._outputs_with_release(handle, {'input_batch': input_batch},
                                        outputs)

  def add_post_restore_hook(self, name_scope):
    """Adds the post restore ops."""
    with tf.name_scope(name_scope):
      return self.build_post_restore_hook()

  def add_saver(self):
    """Adds a Saver for all variables in the graph."""
    logging.info('Saving variables:\n\t%s',
                 '\n\t'.join([x.name for x in tf.global_variables()]))
    self.saver = tf.train.Saver(
        var_list=[x for x in tf.global_variables()],
        write_version=saver_pb2.SaverDef.V1)
