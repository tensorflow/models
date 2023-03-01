# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""TFM common training driver library."""
# pytype: disable=attribute-error
import os
import tempfile
from typing import Any, List, Mapping, Optional, Tuple

# Import libraries

from absl import logging
import orbit
import tensorflow as tf

from official.core import actions
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.core import train_utils

maybe_create_best_ckpt_exporter = train_utils.maybe_create_best_ckpt_exporter


class OrbitExperimentRunner:
  """Runs experiment with Orbit training loop.

  The default experiment runner for model garden experiments. User can
  customize the experiment pipeline by subclassing this class and replacing
  components or functions.

  For example, an experiment runner with customized checkpoint manager:

  ```python
  class MyExpRunnerWithExporter(OrbitExperimentRunner):
    def _maybe_build_checkpoint_manager(sefl):
      # Replaces the default CheckpointManger with a customized one.
      return MyCheckpointManager(*args)

  # In user code, instead of the orginal
  # `OrbitExperimentRunner(..).run(mode)`, now user can do:
  MyExpRunnerWithExporter(**needed_kwargs).run(mode)
  ```

  Similar override can be done to other components.
  """

  def __init__(
      self,
      distribution_strategy: tf.distribute.Strategy,
      task: base_task.Task,
      mode: str,
      params: config_definitions.ExperimentConfig,
      model_dir: str,
      run_post_eval: bool = False,
      save_summary: bool = True,
      train_actions: Optional[List[orbit.Action]] = None,
      eval_actions: Optional[List[orbit.Action]] = None,
      trainer: Optional[base_trainer.Trainer] = None,
      controller_cls=orbit.Controller,
      summary_manager: Optional[orbit.utils.SummaryManager] = None,
      eval_summary_manager: Optional[orbit.utils.SummaryManager] = None,
  ):
    """Constructor.

    Args:
      distribution_strategy: A distribution strategy.
      task: A Task instance.
      mode: A 'str', specifying the mode. Can be 'train', 'eval',
        'train_and_eval' or 'continuous_eval'.
      params: ExperimentConfig instance.
      model_dir: A 'str', a path to store model checkpoints and summaries.
      run_post_eval: Whether to run post eval once after training, metrics logs
        are returned.
      save_summary: Whether to save train and validation summary.
      train_actions: Optional list of Orbit train actions.
      eval_actions: Optional list of Orbit eval actions.
      trainer: the base_trainer.Trainer instance. It should be created within
        the strategy.scope().
      controller_cls: The controller class to manage the train and eval process.
        Must be a orbit.Controller subclass.
      summary_manager: Instance of the summary manager to override default
        summary manager.
      eval_summary_manager: Instance of the eval summary manager to override
        default eval summary manager.
    """
    self.strategy = distribution_strategy or tf.distribute.get_strategy()
    self._params = params
    self._model_dir = model_dir
    self._mode = mode
    self._run_post_eval = run_post_eval

    self._trainer = trainer or self._build_trainer(
        task,
        train='train' in mode,
        evaluate=('eval' in mode) or run_post_eval)
    assert self.trainer is not None
    self._checkpoint_manager = self._maybe_build_checkpoint_manager()
    self._summary_manager = summary_manager
    self._eval_summary_manager = eval_summary_manager
    self._controller = self._build_controller(
        trainer=self.trainer if 'train' in mode else None,
        evaluator=self.trainer,
        save_summary=save_summary,
        train_actions=train_actions,
        eval_actions=eval_actions,
        controller_cls=controller_cls)

  @property
  def params(self) -> config_definitions.ExperimentConfig:
    """The whole experiment parameters object."""
    return self._params

  @property
  def model_dir(self) -> str:
    """Path to the model folder, which stores checkpoints, params, log, etc."""
    return self._model_dir

  @property
  def trainer(self) -> base_trainer.Trainer:
    """The underlying Orbit Trainer object."""
    return self._trainer

  @property
  def checkpoint_manager(self) -> tf.train.CheckpointManager:
    """The CheckpointManager that stores the checkpoints in a train job."""
    return self._checkpoint_manager

  @property
  def controller(self) -> orbit.Controller:
    """The Orbit controller object."""
    return self._controller

  def _build_trainer(self, task: base_task.Task, train: bool,
                     evaluate: bool) -> base_trainer.Trainer:
    """Create trainer."""
    with self.strategy.scope():
      trainer = train_utils.create_trainer(
          self.params,
          task,
          train=train,
          evaluate=evaluate,
          checkpoint_exporter=self._build_best_checkpoint_exporter())
    return trainer

  def _build_best_checkpoint_exporter(self):
    return maybe_create_best_ckpt_exporter(self.params, self.model_dir)

  def _maybe_build_checkpoint_manager(
      self) -> Optional[tf.train.CheckpointManager]:
    """Maybe create a CheckpointManager."""
    assert self.trainer is not None
    if self.trainer.checkpoint:
      if self.model_dir is None:
        raise ValueError('model_dir must be specified, but got None')

      if (not self.strategy) or self.strategy.extended.should_checkpoint:
        ckpt_path = self.model_dir
        max_to_keep = self.params.trainer.max_to_keep
      else:
        # In multi worker training we need every worker to save checkpoint,
        # because variables can trigger synchronization on read and
        # synchronization needs all workers to participate. To avoid workers
        # overriding each other we save to a temporary directory on non-chief
        # workers.
        ckpt_path = tempfile.mkdtemp()
        max_to_keep = 1

      checkpoint_manager = tf.train.CheckpointManager(
          self.trainer.checkpoint,
          directory=ckpt_path,
          max_to_keep=max_to_keep,
          step_counter=self.trainer.global_step,
          checkpoint_interval=self.params.trainer.checkpoint_interval,
          init_fn=self.trainer.initialize)
    else:
      checkpoint_manager = None
    return checkpoint_manager

  def _build_controller(self,
                        trainer,
                        evaluator,
                        save_summary: bool = True,
                        train_actions: Optional[List[orbit.Action]] = None,
                        eval_actions: Optional[List[orbit.Action]] = None,
                        controller_cls=orbit.Controller) -> orbit.Controller:
    """Builds a Orbit controler."""
    train_actions = [] if not train_actions else train_actions
    if trainer:
      train_actions += actions.get_train_actions(
          self.params,
          trainer,
          self.model_dir,
          checkpoint_manager=self.checkpoint_manager)

    eval_actions = [] if not eval_actions else eval_actions
    if evaluator:
      eval_actions += actions.get_eval_actions(self.params, evaluator,
                                               self.model_dir)

    if save_summary:
      eval_summary_dir = os.path.join(
          self.model_dir, self.params.trainer.validation_summary_subdir
      )
    else:
      eval_summary_dir = None

    controller = controller_cls(
        strategy=self.strategy,
        trainer=trainer,
        evaluator=evaluator,
        global_step=self.trainer.global_step,
        steps_per_loop=self.params.trainer.steps_per_loop,
        checkpoint_manager=self.checkpoint_manager,
        summary_dir=os.path.join(self.model_dir, 'train')
        if (save_summary)
        else None,
        eval_summary_dir=eval_summary_dir,
        summary_interval=self.params.trainer.summary_interval
        if (save_summary)
        else None,
        train_actions=train_actions,
        eval_actions=eval_actions,
        summary_manager=self._summary_manager
        if hasattr(self, '_summary_manager')
        else None,
        eval_summary_manager=self._eval_summary_manager
        if hasattr(self, '_eval_summary_manager')
        else None,
    )
    return controller

  def run(self) -> Tuple[tf.keras.Model, Mapping[str, Any]]:
    """Run experiments by mode.

    Returns:
      A 2-tuple of (model, eval_logs).
        model: `tf.keras.Model` instance.
        eval_logs: returns eval metrics logs when run_post_eval is set to True,
          otherwise, returns {}.
    """
    mode = self._mode
    params = self.params
    logging.info('Starts to execute mode: %s', mode)
    with self.strategy.scope():
      if mode == 'train' or mode == 'train_and_post_eval':
        self.controller.train(steps=params.trainer.train_steps)
      elif mode == 'train_and_eval':
        self.controller.train_and_evaluate(
            train_steps=params.trainer.train_steps,
            eval_steps=params.trainer.validation_steps,
            eval_interval=params.trainer.validation_interval)
      elif mode == 'eval':
        self.controller.evaluate(steps=params.trainer.validation_steps)
      elif mode == 'continuous_eval':

        def timeout_fn():
          if self.trainer.global_step.numpy() >= params.trainer.train_steps:
            return True
          return False

        self.controller.evaluate_continuously(
            steps=params.trainer.validation_steps,
            timeout=params.trainer.continuous_eval_timeout,
            timeout_fn=timeout_fn)
      else:
        raise NotImplementedError('The mode is not implemented: %s' % mode)

    num_params = train_utils.try_count_params(self.trainer.model)
    if num_params is not None:
      logging.info('Number of trainable params in model: %f Millions.',
                   num_params / 10.**6)

    flops = train_utils.try_count_flops(self.trainer.model)
    if flops is not None:
      logging.info('FLOPs (multi-adds) in model: %f Billions.',
                   flops / 10.**9 / 2)

    if self._run_post_eval or mode == 'train_and_post_eval':
      with self.strategy.scope():
        return self.trainer.model, self.controller.evaluate(
            steps=params.trainer.validation_steps)
    else:
      return self.trainer.model, {}


def run_experiment(
    distribution_strategy: tf.distribute.Strategy,
    task: base_task.Task,
    mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    save_summary: bool = True,
    train_actions: Optional[List[orbit.Action]] = None,
    eval_actions: Optional[List[orbit.Action]] = None,
    trainer: Optional[base_trainer.Trainer] = None,
    controller_cls=orbit.Controller,
    summary_manager: Optional[orbit.utils.SummaryManager] = None,
    eval_summary_manager: Optional[orbit.utils.SummaryManager] = None,
) -> Tuple[tf.keras.Model, Mapping[str, Any]]:
  """Runs train/eval configured by the experiment params.

  Args:
    distribution_strategy: A distribution distribution_strategy.
    task: A Task instance.
    mode: A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    save_summary: Whether to save train and validation summary.
    train_actions: Optional list of Orbit train actions.
    eval_actions: Optional list of Orbit eval actions.
    trainer: the base_trainer.Trainer instance. It should be created within the
      strategy.scope().
    controller_cls: The controller class to manage the train and eval process.
      Must be a orbit.Controller subclass.
    summary_manager: Instance of the summary manager to override default summary
      manager.
    eval_summary_manager: Instance of the eval summary manager to override
      default eval summary manager.

  Returns:
    A 2-tuple of (model, eval_logs).
      model: `tf.keras.Model` instance.
      eval_logs: returns eval metrics logs when run_post_eval is set to True,
        otherwise, returns {}.
  """
  runner = OrbitExperimentRunner(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=mode,
      params=params,
      model_dir=model_dir,
      run_post_eval=run_post_eval,
      save_summary=save_summary,
      train_actions=train_actions,
      eval_actions=eval_actions,
      trainer=trainer,
      controller_cls=controller_cls,
      summary_manager=summary_manager,
      eval_summary_manager=eval_summary_manager,
  )
  return runner.run()
