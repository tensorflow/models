# Copyright 2018 The TensorFlow Authors.
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

"""Functions for training and evaluation using a TensorFlow Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def evaluate(estimator, eval_args):
  """Runs evaluation on the latest model checkpoint.

  Args:
    estimator: Instance of tf.Estimator.
    eval_args: Dictionary of {eval_name: (input_fn, eval_steps)} where eval_name
      is the name of the evaluation set (e.g. "train" or "val"), input_fn is an
      input function returning a tuple (features, labels), and eval_steps is the
      number of steps for which to evaluate the model (if None, evaluates until
      input_fn raises an end-of-input exception).

  Returns:
    global_step: The global step of the checkpoint evaluated.
    values: A dict of metric values from the evaluation. May be empty, e.g. if
        the training job has not yet saved a checkpoint or the checkpoint is
        deleted by the time the TPU worker initializes.
  """
  # Default return values if evaluation fails.
  global_step = None
  values = {}

  latest_checkpoint = estimator.latest_checkpoint()
  if not latest_checkpoint:
    # This is expected if the training job has not yet saved a checkpoint.
    return global_step, values

  tf.logging.info("Starting evaluation on checkpoint %s", latest_checkpoint)
  try:
    for eval_name, (input_fn, eval_steps) in eval_args.items():
      values[eval_name] = estimator.evaluate(
          input_fn, steps=eval_steps, name=eval_name)
      if global_step is None:
        global_step = values[eval_name].get("global_step")
  except (tf.errors.NotFoundError, ValueError):
    # Expected under some conditions, e.g. checkpoint is already deleted by the
    # trainer process. Increasing RunConfig.keep_checkpoint_max may prevent this
    # in some cases.
    tf.logging.info("Checkpoint %s no longer exists, skipping evaluation.",
                    latest_checkpoint)

  return global_step, values


def continuous_eval(estimator,
                    eval_args,
                    train_steps=None,
                    timeout_secs=None,
                    timeout_fn=None):
  """Runs evaluation whenever there's a new checkpoint.

  Args:
    estimator: Instance of tf.Estimator.
    eval_args: Dictionary of {eval_name: (input_fn, eval_steps)} where eval_name
      is the name of the evaluation set (e.g. "train" or "val"), input_fn is an
      input function returning a tuple (features, labels), and eval_steps is the
      number of steps for which to evaluate the model (if None, evaluates until
      input_fn raises an end-of-input exception).
    train_steps: The number of steps the model will train for. This function
      will terminate once the model has finished training.
    timeout_secs: Number of seconds to wait for new checkpoints. If None, wait
      indefinitely.
    timeout_fn: Optional function to call after timeout. The iterator will exit
      if and only if the function returns True.

  Yields:
    A dict of metric values from each evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  for _ in tf.contrib.training.checkpoints_iterator(
      estimator.model_dir, timeout=timeout_secs, timeout_fn=timeout_fn):
    global_step, values = evaluate(estimator, eval_args)
    yield global_step, values

    global_step = global_step or 0  # Ensure global_step is not None.
    if train_steps and global_step >= train_steps:
      break


def continuous_train_and_eval(estimator,
                              train_input_fn,
                              eval_args,
                              local_eval_frequency=None,
                              train_hooks=None,
                              train_steps=None):
  """Alternates training and evaluation.

  Args:
    estimator: Instance of tf.Estimator.
    train_input_fn: Input function returning a tuple (features, labels).
    eval_args: Dictionary of {eval_name: (input_fn, eval_steps)} where eval_name
      is the name of the evaluation set (e.g. "train" or "val"), input_fn is an
      input function returning a tuple (features, labels), and eval_steps is the
      number of steps for which to evaluate the model (if None, evaluates until
      input_fn raises an end-of-input exception).
    local_eval_frequency: The number of training steps between evaluations. If
      None, trains until train_input_fn raises an end-of-input exception.
    train_hooks: List of SessionRunHook subclass instances. Used for callbacks
      inside the training call.
    train_steps: The total number of steps to train the model for.

  Yields:
    A dict of metric values from each evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  while True:
    # We run evaluation before training in this loop to prevent evaluation from
    # being skipped if the process is interrupted.
    global_step, values = evaluate(estimator, eval_args)
    yield global_step, values

    global_step = global_step or 0  # Ensure global_step is not None.
    if train_steps and global_step >= train_steps:
      break

    # Decide how many steps before the next evaluation.
    steps = local_eval_frequency
    if train_steps:
      remaining_steps = train_steps - global_step
      steps = min(steps, remaining_steps) if steps else remaining_steps

    tf.logging.info("Starting training at global step %d", global_step)
    estimator.train(train_input_fn, hooks=train_hooks, steps=steps)
