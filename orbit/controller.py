# Copyright 2020 The Orbit Authors. All Rights Reserved.
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
"""A light weight utilities to train TF2 models."""

import time
from typing import Callable, Dict, Optional, Text, Union
from absl import logging
import numpy as np
from orbit import runner
from orbit import utils

import tensorflow as tf


def _log_info(message: Text):
  """Logs `message` to the `info` log, and also prints to stdout."""
  logging.info(message)
  print(message)


class Controller:
  """Class that facilitates training and evaluation of models."""

  def __init__(
      self,
      strategy: Optional[tf.distribute.Strategy] = None,
      trainer: Optional[runner.AbstractTrainer] = None,
      evaluator: Optional[runner.AbstractEvaluator] = None,
      global_step: Optional[tf.Variable] = None,
      # Train related
      steps_per_loop: Optional[int] = None,
      checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
      # Summary related
      summary_interval: Optional[int] = None,
      summary_dir: Optional[Text] = None,
      # Evaluation related
      eval_summary_dir: Optional[Text] = None):
    """Constructs a `Controller` instance.

    Args:
      strategy: An instance of `tf.distribute.Strategy`.
      trainer: An instance of `orbit.AbstractTrainer`, which represents model
        training details.
      evaluator: An instance of `orbit.AbstractEvaluator`, which represents
        model evaluation details.
      global_step: An integer `tf.Variable` indicating the global training step
        number. Usually this can be obtained from `iterations` property of the
        model's optimizer (e.g. `self.optimizer.iterations`), or users can
        create their own global step variable as well. If the users create their
        own global step variable, it is recommended to create the `tf.Variable`
        inside strategy scope, and with
        `aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA`.
      steps_per_loop: The number of steps to run in each "inner loop" of
        training (passed to the `num_steps` parameter of `trainer.train`).
      checkpoint_manager: An instance of `tf.train.CheckpointManager`.
      summary_interval: Step interval for training summaries. Note that this
        argument only applies to the summaries inside `trainer.train` function.
        Summaries outside like "steps_per_second" and outputs from
        `trainer.train` function will always be enabled. If set, the value
        should be divisible by steps_per_loop.
      summary_dir: The directory to restore and write checkpoints and summaries.
        For example, You can set it to `checkpoint_manager.directory`.
        If None, it will not write training summarizes.
      eval_summary_dir: The directory to write eval summaries. If None, it will
        be set to `summary_dir`. If both `summary_dir` and `eval_summary_dir`
        are None, it will not write evaluation summarizes.

    Raises:
      ValueError: If both `trainer` and `evaluator` are None.
      ValueError: If `steps_per_loop` is not a positive integer.
      ValueError: If `summary_interval` is not a positive integer or it cannot
        be divisible by `steps_per_loop`.
    """
    if trainer is None and evaluator is None:
      raise ValueError("`trainer` and `evaluator` should not both be None")

    if trainer is not None:
      if steps_per_loop is None:
        raise ValueError("`steps_per_loop` is required when `trainer` is "
                         "provided.")

      if not isinstance(steps_per_loop, int) or steps_per_loop < 1:
        raise ValueError("`steps_per_loop` should be a positive integer")

      if summary_interval is not None:
        if summary_interval <= 0:
          raise ValueError("`summary_interval` should be larger than 0")
        if summary_interval % steps_per_loop != 0:
          raise ValueError("The summary interval ({}) must be a multiple "
                           "of the steps_per_loop ({})".format(
                               summary_interval, steps_per_loop))

    self.trainer = trainer
    self.evaluator = evaluator

    self.strategy = strategy or tf.distribute.get_strategy()

    self.global_step = global_step
    self.checkpoint_manager = checkpoint_manager

    if self.trainer is not None:
      self.step_timer = None
      self.steps_per_loop = steps_per_loop
      self.summary_interval = summary_interval
      self.summary_manager = utils.SummaryManager(
          summary_dir, tf.summary.scalar, global_step=self.global_step)

    if self.evaluator is not None:
      eval_summary_dir = eval_summary_dir or summary_dir
      if eval_summary_dir == summary_dir and self.trainer is not None:
        # Reuse the summary writer if train and evaluation summary directory
        # are the same.
        self.eval_summary_manager = self.summary_manager
      else:
        self.eval_summary_manager = utils.SummaryManager(
            eval_summary_dir, tf.summary.scalar, global_step=self.global_step)

    if self.global_step is not None:
      tf.summary.experimental.set_step(self.global_step)

    # Restores the model if needed.
    # TODO(momernick): We probably only want to do this on certain occasions?
    if self.checkpoint_manager is not None:
      checkpoint_interval = self.checkpoint_manager.checkpoint_interval
      restored_path = self.restore_checkpoint()
      if restored_path:
        logging.info("Restored from checkpoint: %s", restored_path)

  def train(self, steps: int, checkpoint_at_completion: bool = True):
    """Runs training.

    This method calls the `train` method on the Trainable object until the
    global step count is equal to `steps`. It will optionally save checkpoints,
    if a CheckpointManager was passed to the Controller instance's `__init__`.

    Args:
      steps: The global step count to train up to.
      checkpoint_at_completion: Whether to save a checkpoint when this method
        returns. Defaults to True (write the checkpoint). This is always
        triggered, regardless of the checkpointing interval.
    """
    if self.trainer is None:
      raise ValueError("`self.trainer` is required when calling `train` "
                       "method.")
    if self.global_step is None:
      raise ValueError("`self.global_step` is required when calling `train` "
                       "method.")

    # TODO(momernick): Support steps=None or -1 (training to exhaustion).
    current_step = self.global_step.numpy()  # This is an expensive access.
    while current_step < steps:
      logging.info("Train at step %s of %s", current_step, steps)
      # Calculates steps to run for the next train loop.
      num_steps = min(steps - current_step, self.steps_per_loop)
      self._train_n_steps(num_steps)
      self._maybe_save_checkpoint()
      current_step = self.global_step.numpy()  # This is an expensive access.

    if checkpoint_at_completion:
      self.save_checkpoint()

  def evaluate(self, steps: int = None) -> Optional[Dict[Text, np.number]]:
    """Runs evaluation.

    This method calls the `evaluate` method on the Evaluator object for `steps`
    steps, then writes the returned summaries (if any).

    Args:
      steps: The number of steps to evaluate for.

    Returns:
      The evaluation results as a dictionary of numpy values.

    Raises:
      ValueError: If no checkpoint found in `self.checkpoint_manager.directory`.
      ValueError: If `evaluator` is not provided.
    """
    if self.evaluator is None:
      raise ValueError("`evaluator` must be provided to call `evaluate()` "
                       "method.")

    steps = steps or -1
    current_step = self.global_step.numpy()
    if steps > 0:
      logging.info("Running %s steps of evaluation at train step: %s", steps,
                   current_step)
      steps = tf.convert_to_tensor(steps, dtype=tf.int32)
    else:
      logging.info("Evaluating at train step: %s", current_step)

    with self.eval_summary_manager.summary_writer().as_default():
      eval_outputs = self.evaluator.evaluate(steps)

    if eval_outputs:
      eval_outputs = tf.nest.map_structure(utils.get_value, eval_outputs)

    info = "step: {}        evaluation metric: {}".format(
        current_step, eval_outputs)
    _log_info(info)

    self.eval_summary_manager.write_summaries(eval_outputs)
    self.eval_summary_manager.flush()

    return eval_outputs

  def restore_checkpoint(self, checkpoint_path: Text = None):
    """Restore or initialize the model.

    Args:
      checkpoint_path: An optional string indicates the checkpoint path to
        restore. If None, will restore from `self.checkpoint_manager`.

    Returns:
      The path to the restored checkpoint if a restore happened, or None
        if no restore occurred.
    """
    with self.strategy.scope():
      # Checkpoint restoring should be inside scope. b/139450638
      if checkpoint_path is not None:
        self.checkpoint_manager.checkpoint.restore(checkpoint_path)
        return checkpoint_path
      return self.checkpoint_manager.restore_or_initialize()

  def save_checkpoint(self):
    """Checkpoint the model.

    This method will write a checkpoint containing the current state of the
    model.

    Raises:
      ValueError: if no CheckpointManager was provided to this Controller's
        init args.
    """
    self._maybe_save_checkpoint(force_trigger=True)

  def train_and_evaluate(self,
                         train_steps: int = None,
                         eval_steps: int = None,
                         eval_interval: int = None):
    """Train and evaluate in an interleaved manner.

    This method will train the model until the global step count equals
    `train_steps`, running an evaluation for `eval_steps` every `eval_interval`
    training steps. In addition, this method will run a final evaluation at the
    end of the training sequence.

    Args:
      train_steps: The global step count to train up to.
      eval_steps: The number of steps to run during an evaluation. If None,
        this method will evaluate over the entire evaluation dataset.
      eval_interval: The number of training steps to run between evaluations.
        If set, training will always stop every `eval_interval` steps, even if
        this results in a shorter inner loop than specified by `steps_per_loop`
        setting. If None, evaluation will only be performed after training is
        complete.

    Raises:
      ValueError: If eval_interval is not a multiple of self.steps_per_loop.
    """
    current_step = self.global_step.numpy()  # This is an expensive access.
    eval_interval = eval_interval or (train_steps - current_step)
    while current_step < train_steps:
      interval = min(train_steps - current_step, eval_interval)
      num_steps = current_step + interval
      self.train(steps=num_steps, checkpoint_at_completion=False)
      self.evaluate(steps=eval_steps)
      current_step = self.global_step.numpy()  # This is an expensive access.
    self.save_checkpoint()

  def evaluate_continuously(self,
                            steps: int = None,
                            timeout: Optional[Union[int, float]] = None,
                            timeout_fn: Optional[Callable[[], bool]] = None):
    """Monitor a directory and evaluate on checkpoints in it.

    This method continuously monitors a directory as specified by this
    Controller's CheckpointManager init arg and runs evaluation on the
    checkpoints found there.

    Args:
      steps: The number of steps to run when evaluating.
      timeout: The maximum number of seconds to wait between checkpoints. See
        tf.train.checkpoints_iterator documentation.
      timeout_fn: Optional callable to call after a timeout. If the function
        returns True, then it means that no new checkpoints will be generated
        and the iterator will exit.

    Raises:
      ValueError: If no checkpoint found in `self.checkpoint_manager.directory`.
      ValueError: If `evaluator` was not provided as a controller init arg.

    """
    for checkpoint_path in tf.train.checkpoints_iterator(
        self.checkpoint_manager.directory,
        timeout=timeout,
        timeout_fn=timeout_fn):
      self.restore_checkpoint(checkpoint_path)
      self.evaluate(steps)

  def _train_n_steps(self, num_steps: int):
    """Run training for `num_steps`.

    It will also write training outputs to summaries if there is any.

    Args:
      num_steps: An integer indicates how many steps to run for this training
        loop.

    Raises:
      RuntimeError: If `global_step` is not updated correctly in
        `trainer.train`.
    """
    if not self.step_timer:
      self.step_timer = StepTimer(self.global_step)

    # Calculates steps to run for the next train loop.
    current_step = self.global_step.numpy()
    logging.info("Entering training loop at step %s to run %s steps",
                 current_step, num_steps)
    current_step += num_steps
    num_steps = tf.convert_to_tensor(num_steps, dtype=tf.int32)

    with self.summary_manager.summary_writer().as_default():
      # Create a lambda that returns true when summaries should be written.
      should_record = False  # Allows static optimization in no-summary cases.
      if self.summary_interval:
        should_record = lambda: (self.global_step % self.summary_interval == 0)
      with tf.summary.record_if(should_record):
        train_outputs = self.trainer.train(num_steps)

    # Updates and verifies the current step after a training loop finishes.
    if current_step != self.global_step.numpy():
      raise RuntimeError("`trainer.train` function is not updating "
                         "`global_step` correctly, expected: %s, actual: %s" %
                         (current_step, self.global_step.numpy()))

    # Print information like metrics and steps_per_second after a training
    # loop.
    if train_outputs:
      train_outputs = tf.nest.map_structure(utils.get_value, train_outputs)

    train_outputs = train_outputs or {}
    steps_per_second = self.step_timer.steps_per_second()
    info = "step: {}        steps_per_second: {:.2f}        {}".format(
        current_step, steps_per_second, train_outputs)
    _log_info(info)

    train_outputs["steps_per_second"] = steps_per_second
    self.summary_manager.write_summaries(train_outputs)

  def _maybe_save_checkpoint(self, force_trigger: bool = False):
    """Save checkpoints if necessary.

    Args:
      force_trigger: A boolean indicates whether to force saving checkpoints
        regardless of the checkpoint interval.

    Returns:
      A boolean indicating whether a checkpoint was saved.
    """
    if self.checkpoint_manager and self.checkpoint_manager.checkpoint_interval:
      ckpt_path = self.checkpoint_manager.save(
          checkpoint_number=self.global_step.numpy(),
          check_interval=not force_trigger)
      if ckpt_path is not None:
        logging.info("Saved checkpoints in %s", ckpt_path)
        return True
    return False


class StepTimer:
  """Utility class for measuring steps/second."""

  def __init__(self, step):
    self.step = step
    self.start()

  def start(self):
    self.last_iteration = self.step.numpy()
    self.last_time = time.time()

  def steps_per_second(self, restart=True):
    value = ((self.step.numpy() - self.last_iteration) /
             (time.time() - self.last_time))
    if restart:
      self.start()
    return value
