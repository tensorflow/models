# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Custom training loop for running TensorFlow 2.0 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Optional, Dict, List, Text, Callable, Union, Iterator, Any

from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

# pylint: disable=unused-import,g-import-not-at-top,redefined-outer-name,reimported
from official.common import distribute_utils
from official.modeling.hyperparams import params_dict
from official.utils import hyperparams_flags
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS

strategy_flags_dict = hyperparams_flags.strategy_flags_dict
hparam_flags_dict = hyperparams_flags.hparam_flags_dict


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
  """Saves model to model_dir with provided checkpoint prefix."""

  checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
  saved_path = checkpoint.save(checkpoint_path)
  logging.info('Saving model as TF checkpoint: %s', saved_path)


def _steps_to_run(current_step, total_steps, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  return min(total_steps - current_step, steps_per_loop)


def _no_metric():
  return None


def metrics_as_dict(metric):
  """Puts input metric(s) into a list.

  Args:
    metric: metric(s) to be put into the list. `metric` could be an object, a
      list, or a dict of tf.keras.metrics.Metric or has the `required_method`.

  Returns:
    A dictionary of valid metrics.
  """
  if isinstance(metric, tf.keras.metrics.Metric):
    metrics = {metric.name: metric}
  elif isinstance(metric, list):
    metrics = {m.name: m for m in metric}
  elif isinstance(metric, dict):
    metrics = metric
  elif not metric:
    return {}
  else:
    metrics = {'metric': metric}
  return metrics


def metric_results(metric):
  """Collects results from the given metric(s)."""
  metrics = metrics_as_dict(metric)
  metric_result = {
      name: m.result().numpy().astype(float) for name, m in metrics.items()
  }
  return metric_result


def reset_states(metric):
  """Resets states of the given metric(s)."""
  metrics = metrics_as_dict(metric)
  for m in metrics.values():
    m.reset_states()


class SummaryWriter(object):
  """Simple SummaryWriter for writing dictionary of metrics.

  Attributes:
    writer: The tf.SummaryWriter.
  """

  def __init__(self, model_dir: Text, name: Text):
    """Inits SummaryWriter with paths.

    Args:
      model_dir: the model folder path.
      name: the summary subfolder name.
    """
    self.writer = tf.summary.create_file_writer(os.path.join(model_dir, name))

  def __call__(self, metrics: Union[Dict[Text, float], float], step: int):
    """Write metrics to summary with the given writer.

    Args:
      metrics: a dictionary of metrics values. Prefer dictionary.
      step: integer. The training step.
    """
    if not isinstance(metrics, dict):
      # Support scalar metric without name.
      logging.warning('Warning: summary writer prefer metrics as dictionary.')
      metrics = {'metric': metrics}

    with self.writer.as_default():
      for k, v in metrics.items():
        tf.summary.scalar(k, v, step=step)
      self.writer.flush()


class DistributedExecutor(object):
  """Interface to train and eval models with tf.distribute.Strategy."""

  def __init__(self, strategy, params, model_fn, loss_fn, is_multi_host=False):
    """Constructor.

    Args:
      strategy: an instance of tf.distribute.Strategy.
      params: Model configuration needed to run distribution strategy.
      model_fn: Keras model function. Signature:
        (params: ParamsDict) -> tf.keras.models.Model.
      loss_fn: loss function. Signature:
        (y_true: Tensor, y_pred: Tensor) -> Tensor
      is_multi_host: Set to True when using multi hosts for training, like multi
        worker GPU or TPU pod (slice). Otherwise, False.
    """

    self._params = params
    self._model_fn = model_fn
    self._loss_fn = loss_fn
    self._strategy = strategy
    self._checkpoint_name = 'ctl_step_{step}.ckpt'
    self._is_multi_host = is_multi_host
    self.train_summary_writer = None
    self.eval_summary_writer = None
    self.global_train_step = None

  @property
  def checkpoint_name(self):
    """Returns default checkpoint name."""
    return self._checkpoint_name

  @checkpoint_name.setter
  def checkpoint_name(self, name):
    """Sets default summary writer for the current thread."""
    self._checkpoint_name = name

  def loss_fn(self):
    return self._loss_fn()

  def model_fn(self, params):
    return self._model_fn(params)

  def _save_config(self, model_dir):
    """Save parameters to config files if model_dir is defined."""

    logging.info('Save config to model_dir %s.', model_dir)
    if model_dir:
      if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)
      self._params.lock()
      params_dict.save_params_dict_to_yaml(self._params,
                                           model_dir + '/params.yaml')
    else:
      logging.warning('model_dir is empty, so skip the save config.')

  def _get_input_iterator(
      self, input_fn: Callable[..., tf.data.Dataset],
      strategy: tf.distribute.Strategy) -> Optional[Iterator[Any]]:
    """Returns distributed dataset iterator.

    Args:
      input_fn: (params: dict) -> tf.data.Dataset.
      strategy: an instance of tf.distribute.Strategy.

    Returns:
      An iterator that yields input tensors.
    """

    if input_fn is None:
      return None
    # When training with multiple TPU workers, datasets needs to be cloned
    # across workers. Since Dataset instance cannot be cloned in eager mode,
    # we instead pass callable that returns a dataset.
    if self._is_multi_host:
      return iter(strategy.distribute_datasets_from_function(input_fn))
    else:
      input_data = input_fn()
      return iter(strategy.experimental_distribute_dataset(input_data))

  def _create_replicated_step(self,
                              strategy,
                              model,
                              loss_fn,
                              optimizer,
                              metric=None):
    """Creates a single training step.

    Args:
      strategy: an instance of tf.distribute.Strategy.
      model: (Tensor, bool) -> Tensor. model function.
      loss_fn: (y_true: Tensor, y_pred: Tensor) -> Tensor.
      optimizer: tf.keras.optimizers.Optimizer.
      metric: tf.keras.metrics.Metric subclass.

    Returns:
      The training step callable.
    """
    metrics = metrics_as_dict(metric)

    def _replicated_step(inputs):
      """Replicated training step."""
      inputs, labels = inputs

      with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        prediction_loss = loss_fn(labels, outputs)
        loss = tf.reduce_mean(prediction_loss)
        loss = loss / strategy.num_replicas_in_sync
        for m in metrics.values():
          m.update_state(labels, outputs)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    return _replicated_step

  def _create_train_step(self,
                         strategy,
                         model,
                         loss_fn,
                         optimizer,
                         metric=None):
    """Creates a distributed training step.

    Args:
      strategy: an instance of tf.distribute.Strategy.
      model: (Tensor, bool) -> Tensor. model function.
      loss_fn: (y_true: Tensor, y_pred: Tensor) -> Tensor.
      optimizer: tf.keras.optimizers.Optimizer.
      metric: tf.keras.metrics.Metric subclass.

    Returns:
      The training step callable.
    """
    replicated_step = self._create_replicated_step(strategy, model, loss_fn,
                                                   optimizer, metric)

    @tf.function
    def train_step(iterator, num_steps):
      """Performs a distributed training step.

      Args:
        iterator: an iterator that yields input tensors.
        num_steps: the number of steps in the loop.

      Returns:
        The loss tensor.
      """
      if not isinstance(num_steps, tf.Tensor):
        raise ValueError('steps should be an Tensor. Python object may cause '
                         'retracing.')

      per_replica_losses = strategy.run(replicated_step, args=(next(iterator),))
      for _ in tf.range(num_steps - 1):
        per_replica_losses = strategy.run(
            replicated_step, args=(next(iterator),))

      # For reporting, we returns the mean of losses.
      losses = tf.nest.map_structure(
          lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None),
          per_replica_losses)
      return losses

    return train_step

  def _create_test_step(self, strategy, model, metric):
    """Creates a distributed test step."""
    metrics = metrics_as_dict(metric)

    @tf.function
    def test_step(iterator):
      """Calculates evaluation metrics on distributed devices."""
      if not metric:
        logging.info('Skip test_step because metric is None (%s)', metric)
        return None, None

      def _test_step_fn(inputs):
        """Replicated accuracy calculation."""
        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        for m in metrics.values():
          m.update_state(labels, model_outputs)
        return labels, model_outputs

      return strategy.run(_test_step_fn, args=(next(iterator),))

    return test_step

  def train(
      self,
      train_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset],
      eval_input_fn: Optional[Callable[[params_dict.ParamsDict],
                                       tf.data.Dataset]] = None,
      model_dir: Optional[Text] = None,
      total_steps: int = 1,
      iterations_per_loop: int = 1,
      train_metric_fn: Optional[Callable[[], Any]] = None,
      eval_metric_fn: Optional[Callable[[], Any]] = None,
      summary_writer_fn: Callable[[Text, Text], SummaryWriter] = SummaryWriter,
      init_checkpoint: Optional[Callable[[tf.keras.Model], Any]] = None,
      custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
      continuous_eval: bool = False,
      save_config: bool = True):
    """Runs distributed training.

    Args:
      train_input_fn: (params: dict) -> tf.data.Dataset training data input
        function.
      eval_input_fn: (Optional) same type as train_input_fn. If not None, will
        trigger evaluating metric on eval data. If None, will not run the eval
        step.
      model_dir: the folder path for model checkpoints.
      total_steps: total training steps.
      iterations_per_loop: train steps per loop. After each loop, this job will
        update metrics like loss and save checkpoint.
      train_metric_fn: metric_fn for evaluation in train_step.
      eval_metric_fn: metric_fn for evaluation in test_step.
      summary_writer_fn: function to create summary writer.
      init_checkpoint: function to load checkpoint.
      custom_callbacks: A list of Keras Callbacks objects to run during
        training. More specifically, `on_batch_begin()`, `on_batch_end()`,
        methods are invoked during training.
      continuous_eval: If `True`, will continously run evaluation on every
        available checkpoints. If `False`, will do the evaluation once after the
        final step.
      save_config: bool. Whether to save params to model_dir.

    Returns:
      The training loss and eval metrics.
    """
    assert train_input_fn is not None
    if train_metric_fn and not callable(train_metric_fn):
      raise ValueError('if `train_metric_fn` is specified, '
                       'train_metric_fn must be a callable.')
    if eval_metric_fn and not callable(eval_metric_fn):
      raise ValueError('if `eval_metric_fn` is specified, '
                       'eval_metric_fn must be a callable.')
    train_metric_fn = train_metric_fn or _no_metric
    eval_metric_fn = eval_metric_fn or _no_metric

    if custom_callbacks and iterations_per_loop != 1:
      logging.warning(
          'It is sematically wrong to run callbacks when '
          'iterations_per_loop is not one (%s)', iterations_per_loop)

    custom_callbacks = custom_callbacks or []

    def _run_callbacks_on_batch_begin(batch):
      """Runs custom callbacks at the start of every step."""
      if not custom_callbacks:
        return
      for callback in custom_callbacks:
        if callback:
          callback.on_batch_begin(batch)

    def _run_callbacks_on_batch_end(batch):
      """Runs custom callbacks at the end of every step."""
      if not custom_callbacks:
        return
      for callback in custom_callbacks:
        if callback:
          callback.on_batch_end(batch)

    if save_config:
      self._save_config(model_dir)

    if FLAGS.save_checkpoint_freq:
      save_freq = FLAGS.save_checkpoint_freq
    else:
      save_freq = iterations_per_loop

    params = self._params
    strategy = self._strategy
    # To reduce unnecessary send/receive input pipeline operation, we place
    # input pipeline ops in worker task.
    train_iterator = self._get_input_iterator(train_input_fn, strategy)
    train_loss = None
    train_metric_result = None
    eval_metric_result = None
    tf.keras.backend.set_learning_phase(1)
    with strategy.scope():
      # To correctly place the model weights on accelerators,
      # model and optimizer should be created in scope.
      model = self.model_fn(params.as_dict())
      if not hasattr(model, 'optimizer'):
        raise ValueError('User should set optimizer attribute to model '
                         'inside `model_fn`.')
      optimizer = model.optimizer

      # Training loop starts here.
      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
      latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
      initial_step = 0
      if latest_checkpoint_file:
        logging.info(
            'Checkpoint file %s found and restoring from '
            'checkpoint', latest_checkpoint_file)
        checkpoint.restore(latest_checkpoint_file)
        initial_step = optimizer.iterations.numpy()
        logging.info('Loading from checkpoint file completed. Init step %d',
                     initial_step)
      elif init_checkpoint:
        logging.info('Restoring from init checkpoint function')
        init_checkpoint(model)
        logging.info('Loading from init checkpoint file completed')

      current_step = optimizer.iterations.numpy()
      checkpoint_name = self.checkpoint_name

      eval_metric = eval_metric_fn()
      train_metric = train_metric_fn()
      train_summary_writer = summary_writer_fn(model_dir, 'eval_train')
      self.train_summary_writer = train_summary_writer.writer

      test_summary_writer = summary_writer_fn(model_dir, 'eval_test')
      self.eval_summary_writer = test_summary_writer.writer

    # Use training summary writer in TimeHistory if it's in use
    for cb in custom_callbacks:
      if isinstance(cb, keras_utils.TimeHistory):
        cb.summary_writer = self.train_summary_writer

    # Continue training loop.
    train_step = self._create_train_step(
        strategy=strategy,
        model=model,
        loss_fn=self.loss_fn(),
        optimizer=optimizer,
        metric=train_metric)
    test_step = None
    if eval_input_fn and eval_metric:
      self.global_train_step = model.optimizer.iterations
      test_step = self._create_test_step(strategy, model, metric=eval_metric)

    # Step-0 operations
    if current_step == 0 and not latest_checkpoint_file:
      _save_checkpoint(checkpoint, model_dir,
                       checkpoint_name.format(step=current_step))
    if test_step:
      eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
      eval_metric_result = self._run_evaluation(test_step, current_step,
                                                eval_metric, eval_iterator)
      logging.info('Step: %s evalation metric = %s.', current_step,
                   eval_metric_result)
      test_summary_writer(metrics=eval_metric_result, step=optimizer.iterations)
      reset_states(eval_metric)

    logging.info('Training started')
    last_save_checkpoint_step = current_step
    while current_step < total_steps:

      num_steps = _steps_to_run(current_step, total_steps, iterations_per_loop)
      _run_callbacks_on_batch_begin(current_step)
      train_loss = train_step(train_iterator,
                              tf.convert_to_tensor(num_steps, dtype=tf.int32))
      current_step += num_steps

      train_loss = tf.nest.map_structure(lambda x: x.numpy().astype(float),
                                         train_loss)

      _run_callbacks_on_batch_end(current_step - 1)
      if not isinstance(train_loss, dict):
        train_loss = {'total_loss': train_loss}
      if np.isnan(train_loss['total_loss']):
        raise ValueError('total loss is NaN.')

      if train_metric:
        train_metric_result = metric_results(train_metric)
        train_metric_result.update(train_loss)
      else:
        train_metric_result = train_loss
      if callable(optimizer.lr):
        train_metric_result.update(
            {'learning_rate': optimizer.lr(current_step).numpy()})
      else:
        train_metric_result.update({'learning_rate': optimizer.lr.numpy()})
      logging.info('Train Step: %d/%d  / loss = %s / training metric = %s',
                   current_step, total_steps, train_loss, train_metric_result)

      train_summary_writer(
          metrics=train_metric_result, step=optimizer.iterations)

      # Saves model checkpoints and run validation steps at every
      # iterations_per_loop steps.
      # To avoid repeated model saving, we do not save after the last
      # step of training.
      if save_freq > 0 and current_step < total_steps and (
          current_step - last_save_checkpoint_step) >= save_freq:
        _save_checkpoint(checkpoint, model_dir,
                         checkpoint_name.format(step=current_step))
        last_save_checkpoint_step = current_step

      if continuous_eval and current_step < total_steps and test_step:
        eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
        eval_metric_result = self._run_evaluation(test_step, current_step,
                                                  eval_metric, eval_iterator)
        logging.info('Step: %s evalation metric = %s.', current_step,
                     eval_metric_result)
        test_summary_writer(
            metrics=eval_metric_result, step=optimizer.iterations)

      # Re-initialize evaluation metric, except the last step.
      if eval_metric and current_step < total_steps:
        reset_states(eval_metric)
      if train_metric and current_step < total_steps:
        reset_states(train_metric)

    # Reaches the end of training and saves the last checkpoint.
    if last_save_checkpoint_step < total_steps:
      _save_checkpoint(checkpoint, model_dir,
                       checkpoint_name.format(step=current_step))

    if test_step:
      logging.info('Running final evaluation after training is complete.')
      eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
      eval_metric_result = self._run_evaluation(test_step, current_step,
                                                eval_metric, eval_iterator)
      logging.info('Final evaluation metric = %s.', eval_metric_result)
      test_summary_writer(metrics=eval_metric_result, step=optimizer.iterations)

    self.train_summary_writer.close()
    self.eval_summary_writer.close()

    return train_metric_result, eval_metric_result

  def _run_evaluation(self, test_step, current_training_step, metric,
                      test_iterator):
    """Runs validation steps and aggregate metrics."""
    if not test_iterator or not metric:
      logging.warning(
          'Both test_iterator (%s) and metrics (%s) must not be None.',
          test_iterator, metric)
      return None
    logging.info('Running evaluation after step: %s.', current_training_step)
    eval_step = 0
    while True:
      try:
        with tf.experimental.async_scope():
          test_step(test_iterator)
          eval_step += 1
      except (StopIteration, tf.errors.OutOfRangeError):
        tf.experimental.async_clear_error()
        break

    metric_result = metric_results(metric)
    logging.info('Total eval steps: [%d]', eval_step)
    logging.info('At training step: [%r] Validation metric = %r',
                 current_training_step, metric_result)
    return metric_result

  def evaluate_from_model_dir(
      self,
      model_dir: Text,
      eval_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset],
      eval_metric_fn: Callable[[], Any],
      total_steps: int = -1,
      eval_timeout: Optional[int] = None,
      min_eval_interval: int = 180,
      summary_writer_fn: Callable[[Text, Text], SummaryWriter] = SummaryWriter):
    """Runs distributed evaluation on model folder.

    Args:
      model_dir: the folder for storing model checkpoints.
      eval_input_fn: (Optional) same type as train_input_fn. If not None, will
        trigger evaluting metric on eval data. If None, will not run eval step.
      eval_metric_fn: metric_fn for evaluation in test_step.
      total_steps: total training steps. If the current step reaches the
        total_steps, the evaluation loop will stop.
      eval_timeout: The maximum number of seconds to wait between checkpoints.
        If left as None, then the process will wait indefinitely. Used by
        tf.train.checkpoints_iterator.
      min_eval_interval: The minimum number of seconds between yielding
        checkpoints. Used by tf.train.checkpoints_iterator.
      summary_writer_fn: function to create summary writer.

    Returns:
      Eval metrics dictionary of the last checkpoint.
    """

    if not model_dir:
      raise ValueError('model_dir must be set.')

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      eval_timeout)
      return True

    summary_writer = summary_writer_fn(model_dir, 'eval')
    self.eval_summary_writer = summary_writer.writer

    # Read checkpoints from the given model directory
    # until `eval_timeout` seconds elapses.
    for checkpoint_path in tf.train.checkpoints_iterator(
        model_dir,
        min_interval_secs=min_eval_interval,
        timeout=eval_timeout,
        timeout_fn=terminate_eval):
      eval_metric_result, current_step = self.evaluate_checkpoint(
          checkpoint_path=checkpoint_path,
          eval_input_fn=eval_input_fn,
          eval_metric_fn=eval_metric_fn,
          summary_writer=summary_writer)
      if total_steps > 0 and current_step >= total_steps:
        logging.info('Evaluation finished after training step %d', current_step)
        break
    return eval_metric_result

  def evaluate_checkpoint(self,
                          checkpoint_path: Text,
                          eval_input_fn: Callable[[params_dict.ParamsDict],
                                                  tf.data.Dataset],
                          eval_metric_fn: Callable[[], Any],
                          summary_writer: Optional[SummaryWriter] = None):
    """Runs distributed evaluation on the one checkpoint.

    Args:
      checkpoint_path: the checkpoint to evaluate.
      eval_input_fn: (Optional) same type as train_input_fn. If not None, will
        trigger evaluting metric on eval data. If None, will not run eval step.
      eval_metric_fn: metric_fn for evaluation in test_step.
      summary_writer: function to create summary writer.

    Returns:
      Eval metrics dictionary of the last checkpoint.
    """
    if not callable(eval_metric_fn):
      raise ValueError('if `eval_metric_fn` is specified, '
                       'eval_metric_fn must be a callable.')

    old_phase = tf.keras.backend.learning_phase()
    tf.keras.backend.set_learning_phase(0)
    params = self._params
    strategy = self._strategy
    # To reduce unnecessary send/receive input pipeline operation, we place
    # input pipeline ops in worker task.
    with strategy.scope():

      # To correctly place the model weights on accelerators,
      # model and optimizer should be created in scope.
      model = self.model_fn(params.as_dict())
      checkpoint = tf.train.Checkpoint(model=model)

      eval_metric = eval_metric_fn()
      assert eval_metric, 'eval_metric does not exist'
      test_step = self._create_test_step(strategy, model, metric=eval_metric)

      logging.info('Starting to evaluate.')
      if not checkpoint_path:
        raise ValueError('checkpoint path is empty')
      reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
      current_step = reader.get_tensor(
          'optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE')
      logging.info('Checkpoint file %s found and restoring from '
                   'checkpoint', checkpoint_path)
      status = checkpoint.restore(checkpoint_path)
      status.expect_partial().assert_existing_objects_matched()

      self.global_train_step = model.optimizer.iterations
      eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
      eval_metric_result = self._run_evaluation(test_step, current_step,
                                                eval_metric, eval_iterator)
      logging.info('Step: %s evalation metric = %s.', current_step,
                   eval_metric_result)
      summary_writer(metrics=eval_metric_result, step=current_step)
      reset_states(eval_metric)

    tf.keras.backend.set_learning_phase(old_phase)
    return eval_metric_result, current_step

  def predict(self):
    return NotImplementedError('Unimplmented function.')


class ExecutorBuilder(object):
  """Builder of DistributedExecutor.

  Example 1: Builds an executor with supported Strategy.
    builder = ExecutorBuilder(
        strategy_type='tpu',
        strategy_config={'tpu': '/bns/xxx'})
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 2: Builds an executor with customized Strategy.
    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 3: Builds a customized executor with customized Strategy.
    class MyDistributedExecutor(DistributedExecutor):
      # implementation ...

    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        class_ctor=MyDistributedExecutor,
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)
  """

  def __init__(self, strategy_type=None, strategy_config=None):
    _ = distribute_utils.configure_cluster(strategy_config.worker_hosts,
                                           strategy_config.task_index)
    """Constructor.

    Args:
      strategy_type: string. One of 'tpu', 'mirrored', 'multi_worker_mirrored'.
        If None, the user is responsible to set the strategy before calling
        build_executor(...).
      strategy_config: necessary config for constructing the proper Strategy.
        Check strategy_flags_dict() for examples of the structure.
    """
    self._strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=strategy_type,
        num_gpus=strategy_config.num_gpus,
        all_reduce_alg=strategy_config.all_reduce_alg,
        num_packs=strategy_config.num_packs,
        tpu_address=strategy_config.tpu)

  @property
  def strategy(self):
    """Returns default checkpoint name."""
    return self._strategy

  @strategy.setter
  def strategy(self, new_strategy):
    """Sets default summary writer for the current thread."""
    self._strategy = new_strategy

  def build_executor(self,
                     class_ctor=DistributedExecutor,
                     params=None,
                     model_fn=None,
                     loss_fn=None,
                     **kwargs):
    """Creates an executor according to strategy type.

    See doc string of the DistributedExecutor.__init__ for more information of
    the
    input arguments.

    Args:
      class_ctor: A constructor of executor (default: DistributedExecutor).
      params: ParamsDict, all the model parameters and runtime parameters.
      model_fn: Keras model function.
      loss_fn: loss function.
      **kwargs: other arguments to the executor constructor.

    Returns:
      An instance of DistributedExecutor or its subclass.
    """
    if self._strategy is None:
      raise ValueError('`strategy` should not be None. You need to specify '
                       '`strategy_type` in the builder contructor or directly '
                       'set the `strategy` property of the builder.')
    return class_ctor(
        strategy=self._strategy,
        params=params,
        model_fn=model_fn,
        loss_fn=loss_fn,
        **kwargs)
