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

"""Functions for training models with the TensorFlow Estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from astronet.ops import dataset_ops
from astronet.ops import metrics
from astronet.ops import training


def create_input_fn(file_pattern,
                    input_config,
                    mode,
                    shuffle_values_buffer=0,
                    repeat=1):
  """Creates an input_fn that reads a dataset from sharded TFRecord files.

  Args:
    file_pattern: File pattern matching input TFRecord files, e.g.
        "/tmp/train-?????-of-00100". May also be a comma-separated list of file
        patterns.
    input_config: ConfigDict containing feature and label specifications.
    mode: A tf.estimator.ModeKeys.
    shuffle_values_buffer: If > 0, shuffle examples using a buffer of this size.
    repeat: The number of times to repeat the dataset. If None or -1 the
        elements will be repeated indefinitely.

  Returns:
    A callable that builds an input pipeline and returns (features, labels).
  """
  include_labels = (
      mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL])
  reverse_time_series_prob = 0.5 if mode == tf.estimator.ModeKeys.TRAIN else 0
  shuffle_filenames = (mode == tf.estimator.ModeKeys.TRAIN)

  def input_fn(config, params):
    """Builds an input pipeline that reads a dataset from TFRecord files."""
    # Infer whether this input_fn was called by Estimator or TPUEstimator using
    # the config type.
    use_tpu = isinstance(config, tf.contrib.tpu.RunConfig)

    dataset = dataset_ops.build_dataset(
        file_pattern=file_pattern,
        input_config=input_config,
        batch_size=params["batch_size"],
        include_labels=include_labels,
        reverse_time_series_prob=reverse_time_series_prob,
        shuffle_filenames=shuffle_filenames,
        shuffle_values_buffer=shuffle_values_buffer,
        repeat=repeat,
        use_tpu=use_tpu)

    return dataset

  return input_fn


def create_model_fn(model_class, hparams, use_tpu=False):
  """Wraps model_class as an Estimator or TPUEstimator model_fn.

  Args:
    model_class: AstroModel or a subclass.
    hparams: ConfigDict of configuration parameters for building the model.
    use_tpu: If True, a TPUEstimator model_fn is returned. Otherwise an
        Estimator model_fn is returned.

  Returns:
    model_fn: A callable that constructs the model and returns a
        TPUEstimatorSpec if use_tpu is True, otherwise an EstimatorSpec.
  """
  hparams = copy.deepcopy(hparams)

  def model_fn(features, labels, mode, params):
    """Builds the model and returns an EstimatorSpec or TPUEstimatorSpec."""
    # For TPUEstimator, params contains the batch size per TPU core.
    if "batch_size" in params:
      hparams.batch_size = params["batch_size"]

    # Allow labels to be passed in the features dictionary.
    if "labels" in features:
      if labels is not None and labels is not features["labels"]:
        raise ValueError(
            "Conflicting labels: features['labels'] = %s, labels = %s" %
            (features["labels"], labels))
      labels = features.pop("labels")

    model = model_class(features, labels, hparams, mode)
    model.build()

    # Possibly create train_op.
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      learning_rate = training.create_learning_rate(hparams, model.global_step)
      optimizer = training.create_optimizer(hparams, learning_rate, use_tpu)
      train_op = training.create_train_op(model, optimizer)

    # Possibly create evaluation metrics.
    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = (
          metrics.create_metric_fn(model)
          if use_tpu else metrics.create_metrics(model))

    if use_tpu:
      estimator = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=model.predictions,
          loss=model.total_loss,
          train_op=train_op,
          eval_metrics=eval_metrics)
    else:
      estimator = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=model.predictions,
          loss=model.total_loss,
          train_op=train_op,
          eval_metric_ops=eval_metrics)

    return estimator

  return model_fn


def create_estimator(model_class,
                     hparams,
                     run_config=None,
                     model_dir=None,
                     eval_batch_size=None):
  """Wraps model_class as an Estimator or TPUEstimator.

  If run_config is None or a tf.estimator.RunConfig, an Estimator is returned.
  If run_config is a tf.contrib.tpu.RunConfig, a TPUEstimator is returned.

  Args:
    model_class: AstroModel or a subclass.
    hparams: ConfigDict of configuration parameters for building the model.
    run_config: Optional tf.estimator.RunConfig or tf.contrib.tpu.RunConfig.
    model_dir: Optional directory for saving the model. If not passed
        explicitly, it must be specified in run_config.
    eval_batch_size: Optional batch size for evaluation on TPU. Only applicable
        if run_config is a tf.contrib.tpu.RunConfig. Defaults to
        hparams.batch_size.

  Returns:
    An Estimator object if run_config is None or a tf.estimator.RunConfig, or a
    TPUEstimator object if run_config is a tf.contrib.tpu.RunConfig.

  Raises:
    ValueError:
      If model_dir is not passed explicitly or in run_config.model_dir, or if
      eval_batch_size is specified and run_config is not a
      tf.contrib.tpu.RunConfig.
  """
  if run_config is None:
    run_config = tf.estimator.RunConfig()
  else:
    run_config = copy.deepcopy(run_config)

  if not model_dir and not run_config.model_dir:
    raise ValueError(
        "model_dir must be passed explicitly or specified in run_config")

  use_tpu = isinstance(run_config, tf.contrib.tpu.RunConfig)
  model_fn = create_model_fn(model_class, hparams, use_tpu)

  if use_tpu:
    eval_batch_size = eval_batch_size or hparams.batch_size
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=eval_batch_size)

  else:
    if eval_batch_size is not None:
      raise ValueError("eval_batch_size can only be specified for TPU.")

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={"batch_size": hparams.batch_size})

  return estimator


def evaluate(estimator, input_fn, eval_steps=None, eval_name="val"):
  """Runs evaluation on the latest model checkpoint.

  Args:
    estimator: Instance of tf.Estimator.
    input_fn: Input function returning a tuple (features, labels).
    eval_steps: The number of steps for which to evaluate the model. If None,
        evaluates until input_fn raises an end-of-input exception.
    eval_name: Name of the evaluation set, e.g. "train" or "val".

  Returns:
    A dict of metric values from the evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  values = {}  # Default return value if evaluation fails.

  latest_checkpoint = tf.train.latest_checkpoint(estimator.model_dir)
  if not latest_checkpoint:
    # This is expected if the training job has not yet saved a checkpoint.
    return values

  tf.logging.info("Starting evaluation on checkpoint %s", latest_checkpoint)
  try:
    values = estimator.evaluate(input_fn, steps=eval_steps, name=eval_name)
  except tf.errors.NotFoundError:
    # Expected under some conditions, e.g. TPU worker does not finish
    # initializing until long after the CPU job tells it to start evaluating
    # and the checkpoint file is deleted already.
    tf.logging.info("Checkpoint %s no longer exists, skipping evaluation",
                    latest_checkpoint)

  return values


def continuous_eval(estimator,
                    input_fn,
                    train_steps=None,
                    eval_steps=None,
                    eval_name="val"):
  """Runs evaluation whenever there's a new checkpoint.

  Args:
    estimator: Instance of tf.Estimator.
    input_fn: Input function returning a tuple (features, labels).
    train_steps: The number of steps the model will train for. This function
        will terminate once the model has finished training. If None, this
        function will run forever.
    eval_steps: The number of steps for which to evaluate the model. If None,
        evaluates until input_fn raises an end-of-input exception.
    eval_name: Name of the evaluation set, e.g. "train" or "val".

  Yields:
    A dict of metric values from each evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  for _ in tf.contrib.training.checkpoints_iterator(estimator.model_dir):
    values = evaluate(estimator, input_fn, eval_steps, eval_name)
    yield values

    global_step = values.get("global_step", 0)
    if train_steps and global_step >= train_steps:
      break


def continuous_train_and_eval(estimator,
                              train_input_fn,
                              eval_input_fn,
                              local_eval_frequency=None,
                              train_hooks=None,
                              train_steps=None,
                              eval_steps=None,
                              eval_name="val"):
  """Alternates training and evaluation.

  Args:
    estimator: Instance of tf.Estimator.
    train_input_fn: Input function returning a tuple (features, labels).
    eval_input_fn: Input function returning a tuple (features, labels).
    local_eval_frequency: The number of training steps between evaluations. If
        None, trains until train_input_fn raises an end-of-input exception.
    train_hooks: List of SessionRunHook subclass instances. Used for callbacks
        inside the training call.
    train_steps: The total number of steps to train the model for.
    eval_steps: The number of steps for which to evaluate the model. If None,
        evaluates until eval_input_fn raises an end-of-input exception.
    eval_name: Name of the evaluation set, e.g. "train" or "val".

  Yields:
    A dict of metric values from each evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  while True:
    # We run evaluation before training in this loop to prevent evaluation from
    # being skipped if the process is interrupted.
    values = evaluate(estimator, eval_input_fn, eval_steps, eval_name)
    yield values

    global_step = values.get("global_step", 0)
    if train_steps and global_step >= train_steps:
      break

    # Decide how many steps before the next evaluation.
    steps = local_eval_frequency
    if train_steps:
      remaining_steps = train_steps - global_step
      steps = min(steps, remaining_steps) if steps else remaining_steps

    tf.logging.info("Starting training at global step %d", global_step)
    estimator.train(train_input_fn, hooks=train_hooks, steps=steps)
