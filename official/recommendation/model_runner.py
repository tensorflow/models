# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Contains NcfModelRunner, which can train and evaluate an NCF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import time

import tensorflow as tf

from tensorflow.contrib.compiler import xla
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import neumf_model


class NcfModelRunner(object):
  """Creates a graph to train/evaluate an NCF model, and runs it.

  This class builds both a training model and evaluation model in the graph.
  The two models share variables, so that during evaluation, the trained
  variables are used.
  """

  # _TrainModelProperties and _EvalModelProperties store useful properties of
  # the training and evaluation models, respectively.
  # _SHARED_MODEL_PROPERTY_FIELDS is their shared fields.
  _SHARED_MODEL_PROPERTY_FIELDS = (
      # A scalar tf.string placeholder tensor, that will be fed the path to the
      # directory storing the TFRecord files for the input data.
      "record_files_placeholder",
      # The tf.data.Iterator to iterate over the input data.
      "iterator",
      # A scalar float tensor representing the model loss.
      "loss",
      # The batch size, as a Python int.
      "batch_size",
      # The op to run the model. For the training model, this trains the model
      # for one step. For the evaluation model, this computes the metrics and
      # updates the metric variables.
      "run_model_op")
  _TrainModelProperties = namedtuple("_TrainModelProperties",  # pylint: disable=invalid-name
                                     _SHARED_MODEL_PROPERTY_FIELDS)
  _EvalModelProperties = namedtuple(  # pylint: disable=invalid-name
      "_EvalModelProperties", _SHARED_MODEL_PROPERTY_FIELDS + (
          # A dict from metric name to metric tensor.
          "metrics",
          # Initializes the metric variables.
          "metric_initializer",))

  def __init__(self, ncf_dataset, params, num_train_steps, num_eval_steps,
               use_while_loop):
    self._num_train_steps = num_train_steps
    self._num_eval_steps = num_eval_steps
    self._use_while_loop = use_while_loop
    with tf.Graph().as_default() as self._graph:
      if params["use_xla_for_gpu"]:
        # The XLA functions we use require resource variables.
        tf.enable_resource_variables()
      self._ncf_dataset = ncf_dataset
      self._global_step = tf.train.create_global_step()
      self._train_model_properties = self._build_model(params, num_train_steps,
                                                       is_training=True)
      self._eval_model_properties = self._build_model(params, num_eval_steps,
                                                      is_training=False)

      initializer = tf.global_variables_initializer()
    self._graph.finalize()
    self._session = tf.Session(graph=self._graph)
    self._session.run(initializer)

  def _compute_metric_mean(self, metric_name):
    """Computes the mean from a call tf tf.metrics.mean().

    tf.metrics.mean() already returns the mean, so normally this call is
    unnecessary. But, if tf.metrics.mean() is called inside a tf.while_loop, the
    mean cannot be accessed outside the while loop. Calling this function
    recomputes the mean from the variables created by tf.metrics.mean(),
    allowing the mean to be accessed outside the while loop.

    Args:
      metric_name: The string passed to the 'name' argument of tf.metrics.mean()

    Returns:
      The mean of the metric.
    """
    metric_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
    total_suffix = metric_name + "/total:0"
    total_vars = [v for v in metric_vars if v.name.endswith(total_suffix)]
    assert len(total_vars) == 1., (
        "Found {} metric variables ending with '{}' but expected to find "
        "exactly 1. All metric variables: {}".format(
            len(total_vars), total_suffix, metric_vars))
    total_var = total_vars[0]

    count_suffix = metric_name + "/count:0"
    count_vars = [v for v in metric_vars if v.name.endswith(count_suffix)]
    assert len(count_vars) == 1., (
        "Found {} metric variables ending with '{}' but expected to find "
        "exactly 1. All metric variables: {}".format(
            len(count_vars), count_suffix, metric_vars))
    count_var = count_vars[0]
    return total_var / count_var


  def _build_model(self, params, num_steps, is_training):
    """Builds the NCF model.

    Args:
      params: A dict of hyperparameters.
      is_training: If True, build the training model. If False, build the
        evaluation model.
    Returns:
      A _TrainModelProperties if is_training is True, or an _EvalModelProperties
      otherwise.
    """
    record_files_placeholder = tf.placeholder(tf.string, ())
    input_fn, _, _ = \
      data_preprocessing.make_input_fn(
          ncf_dataset=self._ncf_dataset, is_training=is_training,
          record_files=record_files_placeholder)
    dataset = input_fn(params)
    iterator = dataset.make_initializable_iterator()

    model_fn = neumf_model.neumf_model_fn
    if params["use_xla_for_gpu"]:
      model_fn = xla.estimator_model_fn(model_fn)

    if is_training:
      return self._build_train_specific_graph(
          iterator, model_fn, params, record_files_placeholder, num_steps)
    else:
      return self._build_eval_specific_graph(
          iterator, model_fn, params, record_files_placeholder, num_steps)

  def _build_train_specific_graph(self, iterator, model_fn, params,
                                  record_files_placeholder, num_train_steps):
    """Builds the part of the model that is specific to training."""

    def build():
      features, labels = iterator.get_next()
      estimator_spec = model_fn(
          features, labels, tf.estimator.ModeKeys.TRAIN, params)
      with tf.control_dependencies([estimator_spec.train_op]):
        run_model_op = self._global_step.assign_add(1)
      return run_model_op, estimator_spec.loss

    if self._use_while_loop:
      def body(i):
        run_model_op_single_step, _ = build()
        with tf.control_dependencies([run_model_op_single_step]):
          return i + 1

      run_model_op = tf.while_loop(lambda i: i < num_train_steps, body, [0],
                                   parallel_iterations=1)
      loss = None
    else:
      run_model_op, loss = build()

    return self._TrainModelProperties(
        record_files_placeholder, iterator, loss, params["batch_size"],
        run_model_op)

  def _build_eval_specific_graph(self, iterator, model_fn, params,
                                 record_files_placeholder, num_eval_steps):
    """Builds the part of the model that is specific to evaluation."""

    def build():
      features = iterator.get_next()
      estimator_spec = model_fn(
          features, None, tf.estimator.ModeKeys.EVAL, params)
      run_model_op = tf.group(*(update_op for _, update_op in
                                estimator_spec.eval_metric_ops.values()))
      eval_metric_tensors = {k: tensor for (k, (tensor, _))
                             in estimator_spec.eval_metric_ops.items()}
      return run_model_op, estimator_spec.loss, eval_metric_tensors

    if self._use_while_loop:
      def body(i):
        run_model_op_single_step, _, _ = build()
        with tf.control_dependencies([run_model_op_single_step]):
          return i + 1

      run_model_op = tf.while_loop(lambda i: i < num_eval_steps, body, [0],
                                   parallel_iterations=1)
      loss = None
      eval_metric_tensors = {
          "HR": self._compute_metric_mean(rconst.HR_METRIC_NAME),
          "NDCG": self._compute_metric_mean(rconst.NDCG_METRIC_NAME),
      }
    else:
      run_model_op, loss, eval_metric_tensors = build()

    metric_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
    return self._EvalModelProperties(
        record_files_placeholder, iterator, loss, params["eval_batch_size"],
        run_model_op, eval_metric_tensors, metric_initializer)

  def _train_or_eval(self, model_properties, num_steps, is_training):
    """Either trains or evaluates, depending on whether `is_training` is True.

    Args:
      model_properties: _TrainModelProperties or an _EvalModelProperties
        containing the properties of the training or evaluation graph.
      num_steps: The number of steps to train or evaluate for.
      is_training: If True, run the training model. If False, run the evaluation
        model.

    Returns:
      record_dir: The directory of TFRecords where the training/evaluation input
      data was read from.
    """
    if self._ncf_dataset is not None:
      epoch_metadata, record_dir, template = data_preprocessing.get_epoch_info(
          is_training=is_training, ncf_dataset=self._ncf_dataset)
      batch_count = epoch_metadata["batch_count"]
      if batch_count != num_steps:
        raise ValueError(
            "Step counts do not match. ({} vs. {}) The async process is "
            "producing incorrect shards.".format(batch_count, num_steps))
      record_files = os.path.join(record_dir, template.format("*"))
      initializer_feed_dict = {
          model_properties.record_files_placeholder: record_files}
      del batch_count
    else:
      initializer_feed_dict = None
      record_dir = None

    self._session.run(model_properties.iterator.initializer,
                      initializer_feed_dict)
    fetches = (model_properties.run_model_op,)
    if model_properties.loss is not None:
      fetches += (model_properties.loss,)
    mode = "Train" if is_training else "Eval"
    start = None
    times_to_run = 1 if self._use_while_loop else num_steps
    for i in range(times_to_run):
      fetches_ = self._session.run(fetches)
      if i % 100 == 0:
        if start is None:
          # Only start the timer after 100 steps so there is a warmup.
          start = time.time()
          start_step = i
        if model_properties.loss is not None:
          _, loss = fetches_
          tf.logging.info("{} Loss = {}".format(mode, loss))
    end = time.time()
    if start is not None:
      print("{} peformance: {} examples/sec".format(
          mode, (i - start_step) * model_properties.batch_size / (end - start)))
    return record_dir


  def train(self):
    """Trains the graph for a single cycle."""
    record_dir = self._train_or_eval(self._train_model_properties,
                                     self._num_train_steps, is_training=True)
    if record_dir:
      # We delete the record_dir because each cycle, new TFRecords is generated
      # by the async process.
      tf.gfile.DeleteRecursively(record_dir)

  def eval(self):
    """Evaluates the graph on the eval data.

    Returns:
      A dict of evaluation results.
    """
    self._session.run(self._eval_model_properties.metric_initializer)
    self._train_or_eval(self._eval_model_properties, self._num_eval_steps,
                        is_training=False)
    eval_results = {
        'global_step': self._session.run(self._global_step)}
    for key, val in self._eval_model_properties.metrics.items():
      val_ = self._session.run(val)
      tf.logging.info("{} = {}".format(key, self._session.run(val)))
      eval_results[key] = val_
    return eval_results
