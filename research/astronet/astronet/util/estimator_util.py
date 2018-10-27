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

"""Helper functions for creating a TensorFlow Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from astronet.ops import dataset_ops
from astronet.ops import metrics
from astronet.ops import training


class _InputFn(object):
  """Class that acts as a callable input function for Estimator train / eval."""

  def __init__(self,
               file_pattern,
               input_config,
               mode,
               shuffle_values_buffer=0,
               repeat=1):
    """Initializes the input function.

    Args:
      file_pattern: File pattern matching input TFRecord files, e.g.
        "/tmp/train-?????-of-00100". May also be a comma-separated list of file
        patterns.
      input_config: ConfigDict containing feature and label specifications.
      mode: A tf.estimator.ModeKeys.
      shuffle_values_buffer: If > 0, shuffle examples using a buffer of this
        size.
      repeat: The number of times to repeat the dataset. If None or -1 the
        elements will be repeated indefinitely.
    """
    self._file_pattern = file_pattern
    self._input_config = input_config
    self._mode = mode
    self._shuffle_values_buffer = shuffle_values_buffer
    self._repeat = repeat

  def __call__(self, config, params):
    """Builds the input pipeline."""
    # Infer whether this input_fn was called by Estimator or TPUEstimator using
    # the config type.
    use_tpu = isinstance(config, tf.contrib.tpu.RunConfig)

    mode = self._mode
    include_labels = (
        mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL])
    reverse_time_series_prob = 0.5 if mode == tf.estimator.ModeKeys.TRAIN else 0
    shuffle_filenames = (mode == tf.estimator.ModeKeys.TRAIN)
    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=params["batch_size"],
        include_labels=include_labels,
        reverse_time_series_prob=reverse_time_series_prob,
        shuffle_filenames=shuffle_filenames,
        shuffle_values_buffer=self._shuffle_values_buffer,
        repeat=self._repeat,
        use_tpu=use_tpu)

    return dataset


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
    A callable that builds the input pipeline and returns a tf.data.Dataset
    object.
  """
  return _InputFn(file_pattern, input_config, mode, shuffle_values_buffer,
                  repeat)


class _ModelFn(object):
  """Class that acts as a callable model function for Estimator train / eval."""

  def __init__(self, model_class, hparams, use_tpu=False):
    """Initializes the model function.

    Args:
      model_class: Model class.
      hparams: ConfigDict containing hyperparameters for building and training
        the model.
      use_tpu: If True, a TPUEstimator will be returned. Otherwise an Estimator
        will be returned.
    """
    self._model_class = model_class
    self._base_hparams = hparams
    self._use_tpu = use_tpu

  def __call__(self, features, labels, mode, params):
    """Builds the model and returns an EstimatorSpec or TPUEstimatorSpec."""
    hparams = copy.deepcopy(self._base_hparams)
    if "batch_size" in params:
      hparams.batch_size = params["batch_size"]

    # Allow labels to be passed in the features dictionary.
    if "labels" in features:
      if labels is not None and labels is not features["labels"]:
        raise ValueError(
            "Conflicting labels: features['labels'] = {}, labels = {}".format(
                features["labels"], labels))
      labels = features.pop("labels")

    model = self._model_class(features, labels, hparams, mode)
    model.build()

    # Possibly create train_op.
    use_tpu = self._use_tpu
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
  return _ModelFn(model_class, hparams, use_tpu)


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
