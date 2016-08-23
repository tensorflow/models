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
"""Deploy Slim models on multiple towers and replicas.

# TODO(sguada) docstring paragraph by (a) motivating the need for the file and
# (b) defining clones.

# TODO(sguada) describe the high-level components of model deployment.
# E.g. "each model deployment is composed of several parts: a DeploymentConfig,
# which captures A, B and C, an input_fn which loads data.. etc

To easily train a model on multiple GPUs or across multiple machines this
module provides a set of helper functions: `create_towers`,
`optimize_towers` and `deploy`.

Usage:

  g = tf.Graph()

  # Set up DeploymentConfig
  config = slim.DeploymentConfig(num_towers=2, use_gpu=True)

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
  def tower_fn(inputs_queue):
    images, labels = inputs_queue.dequeue()
    predictions = CreateNetwork(images)
    slim.losses.log_loss(predictions, labels)

  tower_dp = slim.deploy(config, tower_fn, [inputs_queue], optimizer=optimizer)

  # Run training.
  slim.learning.train(tower_dp.train_op, my_log_dir,
                      summary_op=tower_dp.summary_op)

Tower namedtuple, holds together the values associated with each call to
tower_fn:
  * outputs: The return values of the calls to `tower_fn()`.
  * scope: The scope used to create the tower.
  * device: The device used to create the tower.

DeployedTower namedtuple, holds together the values needed to train multiple
towers:
  * train_op: An operation that run the optimizer training op and include
    all the update ops created by `tower_fn`. Present only if an optimizer
    was specified.
  * summary_op: An operation that run the summaries created by `tower_fn`
    and process_gradients.
  * total_loss: A `Tensor` that contains the sum of all losses created by
    `tower_fn` plus the regularization losses.
  * towers: List of `Tower` tuples returned by `create_towers()`.

DeploymentConfig parameters:
  * num_towers: Number of model towers to deploy in each replica.
  * use_gpu: True if towers should be deployed on GPUS.
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

import google3
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


__all__ = ['create_towers',
           'deploy',
           'optimize_towers',
           'DeployedTower',
           'DeploymentConfig',
           'Tower',
          ]


# Namedtuple used to represent a tower during deployment.
Tower = collections.namedtuple('Tower',
                               ['outputs',  # Whatever tower_fn() returned.
                                'scope',  # The scope used to create it.
                                'device',  # The device used to create.
                               ])

# Namedtuple used to represent a Deployed Tower, returned by deploy().
DeployedTower = collections.namedtuple('DeployedTower',
                                       ['train_op',  # The `train_op`
                                        'summary_op',  # The `summary_op`
                                        'total_loss',  # The loss `Tensor`
                                        'towers',  # A list of `Towers` tuples.
                                       ])

# Default parameters for DeploymentConfig
_deployment_params = {'num_towers': 1,
                      'use_gpu': True,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'worker',
                      'ps_job_name': 'ps'}


def create_towers(config, tower_fn, args=None, kwargs=None):
  """Creates multiple towers according to config using a `tower_fn`.

  The returned values of `tower_fn(*args, **kwargs)` are collected along with
  the scope and device used to created it in a namedtuple
  `Tower(outputs, scope, device)`

  Note: it is assumed that any loss created by `tower_fn` is collected at
  the tf.GraphKeys.LOSSES collection.

  To recover the losses, summaries or update_ops created by the tower use:
  ```python
    losses = tf.get_collection(tf.GraphKeys.LOSSES, tower.scope)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, tower.scope)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, tower.scope)
  ```

  The deployment options are specified by the config object and support
  deploying one or several towers on different GPUs and one or several replicas
  of such towers.

  The argument `tower_fn` is called `config.num_towers` times to create the
  model towers as `tower_fn(*args, **kwargs)`.

  If `config` specifies deployment on multiple replicas then the default
  tensorflow device is set appropriatly for each call to `tower_fn` and for the
  slim variable creation functions: model and global variables will be created
  on the `ps` device, the tower operations will be on the `worker` device.

  Args:
    config: A DeploymentConfig object.
    tower_fn: A callable. Called as `tower_fn(*args, **kwargs)`
    args: Optional list of arguments to pass to `tower_fn`.
    kwargs: Optional list of keyword arguments to pass to `tower_fn`.

  Returns:
    A list of namedtuples `Tower`.
  """
  towers = []
  args = args or []
  kwargs = kwargs or {}
  with slim.arg_scope([slim.model_variable, slim.variable],
                      device=config.variables_device()):
    # Create towers.
    for i in range(0, config.num_towers):
      with tf.name_scope(config.tower_scope(i)) as tower_scope:
        tower_device = config.tower_device(i)
        with tf.device(tower_device):
          with tf.variable_scope(tf.get_variable_scope(),
                                 reuse=True if i > 0 else None):
            outputs = tower_fn(*args, **kwargs)
          towers.append(Tower(outputs, tower_scope, tower_device))
  return towers


def _gather_tower_loss(tower, num_towers, regularization_losses):
  """Gather the loss for a single tower.

  Args:
    tower: A Tower namedtuple.
    num_towers: The number of towers being deployed.
    regularization_losses: Possibly empty list of regularization_losses
      to add to the tower losses.

  Returns:
    A tensor for the total loss for the tower.  Can be None.
  """
  # The return value.
  sum_loss = None
  # Individual components of the loss that will need summaries.
  tower_loss = None
  regularization_loss = None
  # Compute and aggregate losses on the tower device.
  with tf.device(tower.device):
    all_losses = []
    tower_losses = tf.get_collection(tf.GraphKeys.LOSSES, tower.scope)
    if tower_losses:
      tower_loss = tf.add_n(tower_losses, name='tower_loss')
      if num_towers > 1:
        tower_loss = tf.div(tower_loss, 1.0 * num_towers,
                            name='scaled_tower_loss')
      all_losses.append(tower_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)
  # Add the summaries out of the tower device block.
  if tower_loss is not None:
    tf.scalar_summary(tower.scope + '/tower_loss', tower_loss,
                      name='tower_loss')
  if regularization_loss is not None:
    tf.scalar_summary('regularization_loss', regularization_loss,
                      name='regularization_loss')
  return sum_loss


def _optimize_tower(optimizer, tower, num_towers, regularization_losses,
                    kwargs=None):
  """Compute losses and gradients for a single tower.

  Args:
    optimizer: A tf.Optimizer  object.
    tower: A Tower namedtuple.
    num_towers: The number of towers being deployed.
    regularization_losses: Possibly empty list of regularization_losses
      to add to the tower losses.
    kwargs: Dict of kwarg to pass to compute_gradients().

  Returns:
    A tuple (tower_loss, tower_grads_and_vars).
      - tower_loss: A tensor for the total loss for the tower.  Can be None.
      - tower_grads_and_vars: List of (gradient, variable) for the tower.
        Can be empty.
  """
  sum_loss = _gather_tower_loss(tower, num_towers, regularization_losses)
  tower_grad = None
  if sum_loss is not None:
    with tf.device(tower.device):
      tower_grad = optimizer.compute_gradients(sum_loss, **kwargs)
  return sum_loss, tower_grad


def optimize_towers(towers, optimizer,
                    regularization_losses=None,
                    kwargs=None):
  """Compute tower losses and gradients for the given list of `Towers`.

  Note: The regularization_losses are added to the first tower losses.

  Args:
   towers: List of `Towers` created by `create_towers()`.
   optimizer: An `Optimizer` object.
   regularization_losses: Optional list of regularization losses. If None it
     will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
     exclude them.
   kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

  Returns:
   A tuple (total_loss, grads_and_vars).
     - total_loss: A Tensor containing the average of the tower losses including
       the regularization loss.
     - grads_and_vars: A List of tuples (gradient, variable) containing the sum
       of the gradients for each variable.

  """
  grads_and_vars = []
  towers_losses = []
  kwargs = kwargs or {}
  num_towers = len(towers)
  if regularization_losses is None:
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
  for tower in towers:
    with tf.name_scope(tower.scope):
      tower_loss, tower_grad = _optimize_tower(
          optimizer, tower, num_towers, regularization_losses, kwargs)
      if tower_loss is not None:
        towers_losses.append(tower_loss)
        grads_and_vars.append(tower_grad)
      # Only use regularization_losses for the first tower
      regularization_losses = None
  # Compute the total_loss summing all the towers_losses.
  total_loss = tf.add_n(towers_losses, name='total_loss')
  # Sum the gradients accross towers.
  grads_and_vars = _sum_towers_gradients(grads_and_vars)
  return total_loss, grads_and_vars


def deploy(config,
           tower_fn,
           args=None,
           kwargs=None,
           optimizer=None,
           summarize_gradients=False):
  """Deploys a Slim-constructed model on multiple towers and replicas.

  The deployment options are specified by the config object and support
  deploying one or several towers on different GPUs and one or several replicas
  of such towers.

  The argument `tower_fn` is called `config.num_towers` times to create the
  model towers as `tower_fn(*args, **kwargs)`.

  The optional argument `optimizer` is an `Optimizer` object.  If not `None`,
  the deployed model is configured for training with that optimizer.

  If `config` specifies deployment on multiple replicas then the default
  tensorflow device is set appropriatly for each call to `tower_fn` and for the
  slim variable creation functions: model and global variables will be created
  on the `ps` device, the tower operations will be on the `worker` device.

  Args:
    config: A `DeploymentConfig` object.
    tower_fn: A callable. Called as `tower_fn(*args, **kwargs)`
    args: Optional list of arguments to pass to `tower_fn`.
    kwargs: Optional list of keyword arguments to pass to `tower_fn`.
    optimizer: Optional `Optimizer` object.  If passed the model is deployed
      for training with that optimizer.
    summarize_gradients: Whether or not add summaries to the gradients.

  Returns:
    A `DeployedTower` namedtuple.

  """
  # Gather initial summaries.
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  # Create Towers.
  towers = create_towers(config, tower_fn, args, kwargs)
  first_tower = towers[0]

  # Gather update_ops from the first tower. These contain, for example,
  # the updates for the batch_norm variables created by tower_fn.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_tower.scope)

  train_op = None
  total_loss = None
  with tf.device(config.optimizer_device()):
    if optimizer:
      # Place the global step on the device storing the variables.
      with tf.device(config.variables_device()):
        global_step = slim.get_or_create_global_step()

      # Compute the gradients for the towers.
      total_loss, towers_gradients = optimize_towers(towers, optimizer)

      if towers_gradients:
        if summarize_gradients:
          # Add summaries to the gradients.
          summaries |= set(_add_gradients_summaries(towers_gradients))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(towers_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')
    else:
      towers_losses = []
      regularization_losses = tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES)
      for tower in towers:
        with tf.name_scope(tower.scope):
          tower_loss = _gather_tower_loss(tower, len(towers),
                                          regularization_losses)
          if tower_loss is not None:
            towers_losses.append(tower_loss)
          # Only use regularization_losses for the first tower
          regularization_losses = None
      if towers_losses:
        total_loss = tf.add_n(towers_losses, name='total_loss')

    # Add the summaries from the first tower. These contain the summaries
    # created by tower_fn and either optimize_towers() or _gather_tower_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_tower.scope))

    if total_loss is not None:
      # Add total_loss to summary.
      summaries.add(tf.scalar_summary('total_loss', total_loss,
                                      name='total_loss'))

    if summaries:
      # Merge all summaries together.
      summary_op = tf.merge_summary(list(summaries), name='summary_op')
    else:
      summary_op = None

  return DeployedTower(train_op, summary_op, total_loss, towers)


def _sum_towers_gradients(tower_grads):
  """Calculate the sum gradient for each shared variable across all towers.

  This function assumes that the tower_grads has been scaled appropriately by
  1 / num_towers.

  Args:
    tower_grads: A List of List of tuples (gradient, variable), one list per
    `Tower`.

  Returns:
     List of tuples of (gradient, variable) where the gradient has been summed
     across all towers.
  """
  sum_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_tower0, var0), ... (grad_varN_towerN, varN))
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
               num_towers=1,
               use_gpu=True,
               replica_id=0,
               num_replicas=1,
               num_ps_tasks=0,
               worker_job_name='worker',
               ps_job_name='ps'):
    """Create a DeploymentConfig.

    The config describes how to deploy a model across multiple towers and
    replicas.  The model will be replicated `num_towers` times in each replica.
    If `use_gpu` is True, each tower will use a different GPU.

    If `num_replicas` is 1, the model is deployed via a single process.  In that
    case `worker_device`, `num_ps_tasks`, and `ps_device` are ignored.

    If `num_replicas` is greater than 1, then `worker_device` and `ps_device`
    must specify TensorFlow devices for the `worker` and `ps` jobs and
    `num_ps_tasks` must be positive.

    Args:
      num_towers: Number of model towers to deploy in each replica.
      use_gpu: True if towers should be deployed on GPUS.
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
    self._num_towers = num_towers
    self._use_gpu = use_gpu
    self._replica_id = replica_id
    self._num_replicas = num_replicas
    self._num_ps_tasks = num_ps_tasks
    self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
    self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

  @property
  def num_towers(self):
    return self._num_towers

  @property
  def use_gpu(self):
    return self._use_gpu

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

  def tower_device(self, tower_index):
    """Device used to create the tower and all the ops inside the tower.

    Args:
      tower_index: Int, representing the tower_index.

    Returns:
      A value suitable for `tf.device()`.

    Raises:
      ValueError: if `tower_index` is greater or equal to the number of towers".
    """
    if tower_index >= self._num_towers:
      raise ValueError('tower_index must be less than num_towers')
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    if self._use_gpu:
      device += '/device:GPU:%d' % tower_index
    else:
      device += '/device:CPU:0'
    return device

  def tower_scope(self, tower_index):
    """Name scope to create the tower.

    Args:
      tower_index: Int, representing the tower_index.

    Returns:
      A name_scope suitable for `tf.name_scope()`.

    Raises:
      ValueError: if `tower_index` is greater or equal to the number of towers".
    """
    if tower_index >= self._num_towers:
      raise ValueError('tower_index must be less than num_towers')
    scope = ''
    if self._num_towers > 1:
      scope = 'tower_%d' % tower_index
    return scope

  def optimizer_device(self):
    """Device to use with the optimizer.

    Returns:
      A value suitable for `tf.device()`.
    """
    if self._num_ps_tasks > 0 or self._num_towers > 0:
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
    """Returns the device to use for variables created inside the tower.

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
