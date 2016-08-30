# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy Slim models across multiple clones and replicas.

# TODO(sguada) docstring paragraph by (a) motivating the need for the file and
# (b) defining clones.

# TODO(sguada) describe the high-level components of model deployment.
# E.g. "each model deployment is composed of several parts: a DeploymentConfig,
# which captures A, B and C, an input_fn which loads data.. etc

To easily train a model on multiple GPUs or across multiple machines this
module provides a set of helper functions: `create_clones`,
`optimize_clones` and `deploy`.

Usage:

  g = tf.Graph()

  # Set up DeploymentConfig
  config = model_deploy.DeploymentConfig(num_clones=2, clone_on_cpu=True)

  # Create the global step on the device storing the variables.
  with tf.device(config.variables_device()):
    global_step = slim.create_global_step()

  # Define the inputs
  with tf.device(config.inputs_device()):
    images, labels = LoadData(...)
    inputs_queue = slim.data.prefetch_queue((images, labels))

  # Define the optimizer.
  with tf.device(config.optimizer_device()):
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)

  # Define the model including the loss.
  def model_fn(inputs_queue):
    images, labels = inputs_queue.dequeue()
    predictions = CreateNetwork(images)
    slim.losses.log_loss(predictions, labels)

  model_dp = model_deploy.deploy(config, model_fn, [inputs_queue],
                                 optimizer=optimizer)

  # Run training.
  slim.learning.train(model_dp.train_op, my_log_dir,
                      summary_op=model_dp.summary_op)

The Clone namedtuple holds together the values associated with each call to
model_fn:
  * outputs: The return values of the calls to `model_fn()`.
  * scope: The scope used to create the clone.
  * device: The device used to create the clone.

DeployedModel namedtuple, holds together the values needed to train multiple
clones:
  * train_op: An operation that run the optimizer training op and include
    all the update ops created by `model_fn`. Present only if an optimizer
    was specified.
  * summary_op: An operation that run the summaries created by `model_fn`
    and process_gradients.
  * total_loss: A `Tensor` that contains the sum of all losses created by
    `model_fn` plus the regularization losses.
  * clones: List of `Clone` tuples returned by `create_clones()`.

DeploymentConfig parameters:
  * num_clones: Number of model clones to deploy in each replica.
  * clone_on_cpu: True if clones should be placed on CPU.
  * replica_id: Integer.  Index of the replica for which the model is
      deployed.  Usually 0 for the chief replica.
  * num_replicas: Number of replicas to use.
  * num_ps_tasks: Number of tasks for the `ps` job. 0 to not use replicas.
  * worker_job_name: A name for the worker job.
  * ps_job_name: A name for the parameter server job.

TODO(sguada):
  - describe side effect to the graph.
  - what happens to summaries and update_ops.
  - which graph collections are altered.
  - write a tutorial on how to use this.
  - analyze the possibility of calling deploy more than once.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


__all__ = ['create_clones',
           'deploy',
           'optimize_clones',
           'DeployedModel',
           'DeploymentConfig',
           'Clone',
          ]


# Namedtuple used to represent a clone during deployment.
Clone = collections.namedtuple('Clone',
                               ['outputs',  # Whatever model_fn() returned.
                                'scope',  # The scope used to create it.
                                'device',  # The device used to create.
                               ])

# Namedtuple used to represent a DeployedModel, returned by deploy().
DeployedModel = collections.namedtuple('DeployedModel',
                                       ['train_op',  # The `train_op`
                                        'summary_op',  # The `summary_op`
                                        'total_loss',  # The loss `Tensor`
                                        'clones',  # A list of `Clones` tuples.
                                       ])

# Default parameters for DeploymentConfig
_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'worker',
                      'ps_job_name': 'ps'}


def create_clones(config, model_fn, args=None, kwargs=None):
  """Creates multiple clones according to config using a `model_fn`.

  The returned values of `model_fn(*args, **kwargs)` are collected along with
  the scope and device used to created it in a namedtuple
  `Clone(outputs, scope, device)`

  Note: it is assumed that any loss created by `model_fn` is collected at
  the tf.GraphKeys.LOSSES collection.

  To recover the losses, summaries or update_ops created by the clone use:
  ```python
    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, clone.scope)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone.scope)
  ```

  The deployment options are specified by the config object and support
  deploying one or several clones on different GPUs and one or several replicas
  of such clones.

  The argument `model_fn` is called `config.num_clones` times to create the
  model clones as `model_fn(*args, **kwargs)`.

  If `config` specifies deployment on multiple replicas then the default
  tensorflow device is set appropriatly for each call to `model_fn` and for the
  slim variable creation functions: model and global variables will be created
  on the `ps` device, the clone operations will be on the `worker` device.

  Args:
    config: A DeploymentConfig object.
    model_fn: A callable. Called as `model_fn(*args, **kwargs)`
    args: Optional list of arguments to pass to `model_fn`.
    kwargs: Optional list of keyword arguments to pass to `model_fn`.

  Returns:
    A list of namedtuples `Clone`.
  """
  clones = []
  args = args or []
  kwargs = kwargs or {}
  with slim.arg_scope([slim.model_variable, slim.variable],
                      device=config.variables_device()):
    # Create clones.
    for i in range(0, config.num_clones):
      with tf.name_scope(config.clone_scope(i)) as clone_scope:
        clone_device = config.clone_device(i)
        with tf.device(clone_device):
          with tf.variable_scope(tf.get_variable_scope(),
                                 reuse=True if i > 0 else None):
            outputs = model_fn(*args, **kwargs)
          clones.append(Clone(outputs, clone_scope, clone_device))
  return clones


def _gather_clone_loss(clone, num_clones, regularization_losses):
  """Gather the loss for a single clone.

  Args:
    clone: A Clone namedtuple.
    num_clones: The number of clones being deployed.
    regularization_losses: Possibly empty list of regularization_losses
      to add to the clone losses.

  Returns:
    A tensor for the total loss for the clone.  Can be None.
  """
  # The return value.
  sum_loss = None
  # Individual components of the loss that will need summaries.
  clone_loss = None
  regularization_loss = None
  # Compute and aggregate losses on the clone device.
  with tf.device(clone.device):
    all_losses = []
    clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
    if clone_losses:
      clone_loss = tf.add_n(clone_losses, name='clone_loss')
      if num_clones > 1:
        clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                            name='scaled_clone_loss')
      all_losses.append(clone_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)
  # Add the summaries out of the clone device block.
  if clone_loss is not None:
    tf.scalar_summary(clone.scope + '/clone_loss', clone_loss,
                      name='clone_loss')
  if regularization_loss is not None:
    tf.scalar_summary('regularization_loss', regularization_loss,
                      name='regularization_loss')
  return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):
  """Compute losses and gradients for a single clone.

  Args:
    optimizer: A tf.Optimizer  object.
    clone: A Clone namedtuple.
    num_clones: The number of clones being deployed.
    regularization_losses: Possibly empty list of regularization_losses
      to add to the clone losses.
    **kwargs: Dict of kwarg to pass to compute_gradients().

  Returns:
    A tuple (clone_loss, clone_grads_and_vars).
      - clone_loss: A tensor for the total loss for the clone.  Can be None.
      - clone_grads_and_vars: List of (gradient, variable) for the clone.
        Can be empty.
  """
  sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
  clone_grad = None
  if sum_loss is not None:
    with tf.device(clone.device):
      clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
  return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):
  """Compute clone losses and gradients for the given list of `Clones`.

  Note: The regularization_losses are added to the first clone losses.

  Args:
   clones: List of `Clones` created by `create_clones()`.
   optimizer: An `Optimizer` object.
   regularization_losses: Optional list of regularization losses. If None it
     will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
     exclude them.
   **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

  Returns:
   A tuple (total_loss, grads_and_vars).
     - total_loss: A Tensor containing the average of the clone losses including
       the regularization loss.
     - grads_and_vars: A List of tuples (gradient, variable) containing the sum
       of the gradients for each variable.

  """
  grads_and_vars = []
  clones_losses = []
  num_clones = len(clones)
  if regularization_losses is None:
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
  for clone in clones:
    with tf.name_scope(clone.scope):
      clone_loss, clone_grad = _optimize_clone(
          optimizer, clone, num_clones, regularization_losses, **kwargs)
      if clone_loss is not None:
        clones_losses.append(clone_loss)
        grads_and_vars.append(clone_grad)
      # Only use regularization_losses for the first clone
      regularization_losses = None
  # Compute the total_loss summing all the clones_losses.
  total_loss = tf.add_n(clones_losses, name='total_loss')
  # Sum the gradients accross clones.
  grads_and_vars = _sum_clones_gradients(grads_and_vars)
  return total_loss, grads_and_vars


def deploy(config,
           model_fn,
           args=None,
           kwargs=None,
           optimizer=None,
           summarize_gradients=False):
  """Deploys a Slim-constructed model across multiple clones.

  The deployment options are specified by the config object and support
  deploying one or several clones on different GPUs and one or several replicas
  of such clones.

  The argument `model_fn` is called `config.num_clones` times to create the
  model clones as `model_fn(*args, **kwargs)`.

  The optional argument `optimizer` is an `Optimizer` object.  If not `None`,
  the deployed model is configured for training with that optimizer.

  If `config` specifies deployment on multiple replicas then the default
  tensorflow device is set appropriatly for each call to `model_fn` and for the
  slim variable creation functions: model and global variables will be created
  on the `ps` device, the clone operations will be on the `worker` device.

  Args:
    config: A `DeploymentConfig` object.
    model_fn: A callable. Called as `model_fn(*args, **kwargs)`
    args: Optional list of arguments to pass to `model_fn`.
    kwargs: Optional list of keyword arguments to pass to `model_fn`.
    optimizer: Optional `Optimizer` object.  If passed the model is deployed
      for training with that optimizer.
    summarize_gradients: Whether or not add summaries to the gradients.

  Returns:
    A `DeployedModel` namedtuple.

  """
  # Gather initial summaries.
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  # Create Clones.
  clones = create_clones(config, model_fn, args, kwargs)
  first_clone = clones[0]

  # Gather update_ops from the first clone. These contain, for example,
  # the updates for the batch_norm variables created by model_fn.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone.scope)

  train_op = None
  total_loss = None
  with tf.device(config.optimizer_device()):
    if optimizer:
      # Place the global step on the device storing the variables.
      with tf.device(config.variables_device()):
        global_step = slim.get_or_create_global_step()

      # Compute the gradients for the clones.
      total_loss, clones_gradients = optimize_clones(clones, optimizer)

      if clones_gradients:
        if summarize_gradients:
          # Add summaries to the gradients.
          summaries |= set(_add_gradients_summaries(clones_gradients))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')
    else:
      clones_losses = []
      regularization_losses = tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES)
      for clone in clones:
        with tf.name_scope(clone.scope):
          clone_loss = _gather_clone_loss(clone, len(clones),
                                          regularization_losses)
          if clone_loss is not None:
            clones_losses.append(clone_loss)
          # Only use regularization_losses for the first clone
          regularization_losses = None
      if clones_losses:
        total_loss = tf.add_n(clones_losses, name='total_loss')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone.scope))

    if total_loss is not None:
      # Add total_loss to summary.
      summaries.add(tf.scalar_summary('total_loss', total_loss,
                                      name='total_loss'))

    if summaries:
      # Merge all summaries together.
      summary_op = tf.merge_summary(list(summaries), name='summary_op')
    else:
      summary_op = None

  return DeployedModel(train_op, summary_op, total_loss, clones)


def _sum_clones_gradients(clone_grads):
  """Calculate the sum gradient for each shared variable across all clones.

  This function assumes that the clone_grads has been scaled appropriately by
  1 / num_clones.

  Args:
    clone_grads: A List of List of tuples (gradient, variable), one list per
    `Clone`.

  Returns:
     List of tuples of (gradient, variable) where the gradient has been summed
     across all clones.
  """
  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      sum_grads.append((sum_grad, var))
  return sum_grads


def _add_gradients_summaries(grads_and_vars):
  """Add histogram summaries to gradients.

  Note: The summaries are also added to the SUMMARIES collection.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The _list_ of the added summaries for grads_and_vars.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(tf.histogram_summary(var.op.name + ':gradient',
                                            grad_values))
      summaries.append(tf.histogram_summary(var.op.name + ':gradient_norm',
                                            tf.global_norm([grad_values])))
    else:
      tf.logging.info('Var %s has no gradient', var.op.name)
  return summaries


class DeploymentConfig(object):
  """Configuration for deploying a model with `deploy()`.

  You can pass an instance of this class to `deploy()` to specify exactly
  how to deploy the model to build.  If you do not pass one, an instance built
  from the default deployment_hparams will be used.
  """

  def __init__(self,
               num_clones=1,
               clone_on_cpu=False,
               replica_id=0,
               num_replicas=1,
               num_ps_tasks=0,
               worker_job_name='worker',
               ps_job_name='ps'):
    """Create a DeploymentConfig.

    The config describes how to deploy a model across multiple clones and
    replicas.  The model will be replicated `num_clones` times in each replica.
    If `clone_on_cpu` is True, each clone will placed on CPU.

    If `num_replicas` is 1, the model is deployed via a single process.  In that
    case `worker_device`, `num_ps_tasks`, and `ps_device` are ignored.

    If `num_replicas` is greater than 1, then `worker_device` and `ps_device`
    must specify TensorFlow devices for the `worker` and `ps` jobs and
    `num_ps_tasks` must be positive.

    Args:
      num_clones: Number of model clones to deploy in each replica.
      clone_on_cpu: If True clones would be placed on CPU.
      replica_id: Integer.  Index of the replica for which the model is
        deployed.  Usually 0 for the chief replica.
      num_replicas: Number of replicas to use.
      num_ps_tasks: Number of tasks for the `ps` job. 0 to not use replicas.
      worker_job_name: A name for the worker job.
      ps_job_name: A name for the parameter server job.

    Raises:
      ValueError: If the arguments are invalid.
    """
    if num_replicas > 1:
      if num_ps_tasks < 1:
        raise ValueError('When using replicas num_ps_tasks must be positive')
    if num_replicas > 1 or num_ps_tasks > 0:
      if not worker_job_name:
        raise ValueError('Must specify worker_job_name when using replicas')
      if not ps_job_name:
        raise ValueError('Must specify ps_job_name when using parameter server')
    if replica_id >= num_replicas:
      raise ValueError('replica_id must be less than num_replicas')
    self._num_clones = num_clones
    self._clone_on_cpu = clone_on_cpu
    self._replica_id = replica_id
    self._num_replicas = num_replicas
    self._num_ps_tasks = num_ps_tasks
    self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
    self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

  @property
  def num_clones(self):
    return self._num_clones

  @property
  def clone_on_cpu(self):
    return self._clone_on_cpu

  @property
  def replica_id(self):
    return self._replica_id

  @property
  def num_replicas(self):
    return self._num_replicas

  @property
  def num_ps_tasks(self):
    return self._num_ps_tasks

  @property
  def ps_device(self):
    return self._ps_device

  @property
  def worker_device(self):
    return self._worker_device

  def caching_device(self):
    """Returns the device to use for caching variables.

    Variables are cached on the worker CPU when using replicas.

    Returns:
      A device string or None if the variables do not need to be cached.
    """
    if self._num_ps_tasks > 0:
      return lambda op: op.device
    else:
      return None

  def clone_device(self, clone_index):
    """Device used to create the clone and all the ops inside the clone.

    Args:
      clone_index: Int, representing the clone_index.

    Returns:
      A value suitable for `tf.device()`.

    Raises:
      ValueError: if `clone_index` is greater or equal to the number of clones".
    """
    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    if self._clone_on_cpu:
      device += '/device:CPU:0'
    else:
      if self._num_clones > 1:
        device += '/device:GPU:%d' % clone_index
    return device

  def clone_scope(self, clone_index):
    """Name scope to create the clone.

    Args:
      clone_index: Int, representing the clone_index.

    Returns:
      A name_scope suitable for `tf.name_scope()`.

    Raises:
      ValueError: if `clone_index` is greater or equal to the number of clones".
    """
    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    scope = ''
    if self._num_clones > 1:
      scope = 'clone_%d' % clone_index
    return scope

  def optimizer_device(self):
    """Device to use with the optimizer.

    Returns:
      A value suitable for `tf.device()`.
    """
    if self._num_ps_tasks > 0 or self._num_clones > 0:
      return self._worker_device + '/device:CPU:0'
    else:
      return ''

  def inputs_device(self):
    """Device to use to build the inputs.

    Returns:
      A value suitable for `tf.device()`.
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    device += '/device:CPU:0'
    return device

  def variables_device(self):
    """Returns the device to use for variables created inside the clone.

    Returns:
      A value suitable for `tf.device()`.
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._ps_device
    device += '/device:CPU:0'

    class _PSDeviceChooser(object):
      """Slim device chooser for variables when using PS."""

      def __init__(self, device, tasks):
        self._device = device
        self._tasks = tasks
        self._task = 0

      def choose(self, op):
        if op.device:
          return op.device
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == 'Variable':
          t = self._task
          self._task = (self._task + 1) % self._tasks
          d = '%s/task:%d' % (self._device, t)
          return d
        else:
          return op.device

    if not self._num_ps_tasks:
      return device
    else:
      chooser = _PSDeviceChooser(device, self._num_ps_tasks)
      return chooser.choose
