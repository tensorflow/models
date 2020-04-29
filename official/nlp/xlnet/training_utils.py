# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""XLNet training utils."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
import re

from absl import logging

# pytype: disable=attribute-error
# pylint: disable=g-bare-generic,unused-import
import tensorflow as tf
from typing import Any, Callable, Dict, Text, Optional

from official.nlp.bert import model_training_utils
from official.nlp.xlnet import data_utils
from official.nlp.xlnet import xlnet_modeling as modeling

_MIN_SUMMARY_STEPS = 10


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
  """Saves model to with provided checkpoint prefix."""

  checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
  saved_path = checkpoint.save(checkpoint_path)
  logging.info("Saving model as TF checkpoint: %s", saved_path)
  return


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


def train(
    strategy: tf.distribute.Strategy,
    model_fn: Callable,
    input_meta_data: Dict,
    train_input_fn: Callable,
    total_training_steps: int,
    steps_per_loop: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate_fn: tf.keras.optimizers.schedules.LearningRateSchedule,
    eval_fn: Optional[Callable[[tf.keras.Model, int, tf.summary.SummaryWriter],
                               Any]] = None,
    metric_fn: Optional[Callable[[], tf.keras.metrics.Metric]] = None,
    init_checkpoint: Optional[Text] = None,
    init_from_transformerxl: Optional[bool] = False,
    model_dir: Optional[Text] = None,
    save_steps: Optional[int] = None,
    run_eagerly: Optional[bool] = False):
  """Runs customized training.

  Args:
      strategy: Distribution strategy on which to run low level training loop.
      model_fn: The function returns a keras.Model.
      input_meta_data: A dictionary of params: `mem_len`, `lr_layer_decay_rate`,
        `n_layer`, `batch_size_per_core` and `d_model`.
      train_input_fn: Function returns a tf.data.Dataset used for training.
      total_training_steps: Number of steps to train in total.
      steps_per_loop: Number of steps per graph-mode loop. In order to reduce
        communication in eager context, training logs are printed every
        steps_per_loop.
      optimizer: The optimizer for model.
      learning_rate_fn: the learning rate schedule.
      eval_fn: A callback of evaluation function, that takes a keras.Model,
        current step and evaluation summary writer.
      metric_fn: A metrics function returns a Keras Metric object to record
        evaluation result using evaluation dataset or with training dataset
        after every epoch.
      init_checkpoint: Optional checkpoint to load to `sub_model` returned by
        `model_fn`.
      init_from_transformerxl: Whether to load to `transformerxl_model` of
        `model_fn`.
      model_dir: The directory of model (checkpoints, summaries).
      save_steps: The frequency to save checkpoints. Every save_steps, we save a
        model checkpoint. Model checkpoint will be saved and evaluation will be
        conducted if evaluation dataset is provided.
      run_eagerly: Whether to run training eagerly.

  Returns:
      Last training step logits if training happens, otherwise returns None.
  Raises:
    TypeError: if model directory is not specified.
  """
  required_arguments = [
      train_input_fn, total_training_steps, steps_per_loop, optimizer,
      learning_rate_fn, save_steps
  ]
  if [arg for arg in required_arguments if arg is None]:
    raise ValueError("`train_input_fn`, `total_training_steps`, "
                     "`steps_per_loop`, `optimizer`, `save_steps` and "
                     "`learning_rate_fn` are required parameters.")
  if not model_dir:
    raise TypeError("Model directory must be specified.")
  train_iterator = data_utils.get_input_iterator(train_input_fn, strategy)
  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.mkdir(model_dir)
  # Create summary writers
  summary_dir = os.path.join(model_dir, "summaries")
  if not tf.io.gfile.exists(summary_dir):
    tf.io.gfile.mkdir(summary_dir)
  train_summary_writer = None
  eval_summary_writer = None
  if eval_fn:
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, "eval"))
  if steps_per_loop >= _MIN_SUMMARY_STEPS:
    # Only writes summary when the stats are collected sufficiently over
    # enough steps.
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, "train"))

  with strategy.scope():
    model = model_fn()

    if init_checkpoint:
      logging.info("restore from %s", init_checkpoint)
      if init_from_transformerxl:
        checkpoint = tf.train.Checkpoint(
            transformer_xl=model.transformerxl_model)
      else:
        checkpoint = tf.train.Checkpoint(model=model)
      checkpoint.restore(init_checkpoint)

    model.optimizer = optimizer

    if not hasattr(model, "optimizer"):
      raise ValueError("User should set optimizer attribute to model.")

    train_loss_metric = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
    train_metric = None
    if metric_fn:
      train_metric = metric_fn()

    def _replicated_step(inputs, mem=None):
      """Replicated training step."""

      inputs["mems"] = mem
      with tf.GradientTape() as tape:
        mem, logits = model(inputs, training=True)
        loss = model.losses
        train_loss_metric.update_state(loss)
        if train_metric:
          train_metric.update_state(inputs["label_ids"], logits)
        scaled_loss = loss[0] * 1.0 / float(strategy.num_replicas_in_sync)

      # Collects training variables.
      tvars = model.trainable_variables
      grads = tape.gradient(scaled_loss, tvars)
      clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

      if input_meta_data["lr_layer_decay_rate"] != 1.0:
        n_layer = 0
        for i in range(len(clipped)):
          m = re.search(r"model/transformer/layer_(\d+?)/", tvars[i].name)
          if not m:
            continue
          n_layer = max(n_layer, int(m.group(1)) + 1)

        for i in range(len(clipped)):
          for l in range(n_layer):
            if "model/transformer/layer_{}/".format(l) in tvars[i].name:
              abs_rate = input_meta_data["lr_layer_decay_rate"]**(
                  n_layer - 1 - l)
              clipped[i] *= abs_rate
              logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
                  abs_rate, l, tvars[i].name))
              break

      optimizer.apply_gradients(zip(clipped, tvars))
      if input_meta_data["mem_len"] > 0:
        return mem

    def train_steps(iterator, steps):
      """Performs distributed training steps in a loop.

      Args:
        iterator: the distributed iterator of training datasets.
        steps: an tf.int32 integer tensor to specify number of steps to run
          inside host training loop.

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.

      Returns:
        logits: logits computed.
      """
      if not isinstance(steps, tf.Tensor):
        raise ValueError("steps should be an Tensor. Python object may cause "
                         "retracing.")

      def cache_fn():
        """Initializes memory tensor used in XLNet pretraining."""
        mems = []
        if input_meta_data["mem_len"] > 0:
          for _ in range(input_meta_data["n_layer"]):
            zeros = tf.zeros([
                input_meta_data["mem_len"],
                input_meta_data["batch_size_per_core"],
                input_meta_data["d_model"]
            ],
                             dtype=tf.float32)
            mems.append(zeros)
        return mems

      if input_meta_data["mem_len"] > 0:
        mem = strategy.run(cache_fn)
        for _ in tf.range(steps):
          mem = strategy.run(
              _replicated_step, args=(
                  next(iterator),
                  mem,
              ))
      else:
        for _ in tf.range(steps):
          strategy.run(_replicated_step, args=(next(iterator),))

    if not run_eagerly:
      train_steps = tf.function(train_steps)

    logging.info("Start training...")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint_file:
      logging.info("Checkpoint file %s found and restoring from checkpoint",
                   latest_checkpoint_file)
      checkpoint.restore(latest_checkpoint_file)
      logging.info("Loading from checkpoint file completed")

    current_step = optimizer.iterations.numpy()
    checkpoint_name = "xlnet_step_{step}.ckpt"

    while current_step < total_training_steps:
      train_loss_metric.reset_states()
      if train_metric:
        train_metric.reset_states()

      steps = model_training_utils.steps_to_run(current_step, save_steps,
                                                steps_per_loop)
      train_steps(train_iterator, tf.convert_to_tensor(steps, dtype=tf.int32))
      current_step += steps
      train_loss = _float_metric_value(train_loss_metric)
      log_stream = "Train step: %d/%d  /  lr = %.9f  /  loss = %.7f" % (
          current_step, total_training_steps, learning_rate_fn(current_step),
          train_loss)
      if train_metric:
        log_stream += "  /  %s = %f" % (train_metric.name,
                                        _float_metric_value(train_metric))
      logging.info(log_stream)
      if train_summary_writer:
        with train_summary_writer.as_default():
          tf.summary.scalar(
              "learning_rate",
              learning_rate_fn(current_step),
              step=current_step)
          tf.summary.scalar(
              train_loss_metric.name, train_loss, step=current_step)
          if train_metric:
            tf.summary.scalar(
                train_metric.name,
                _float_metric_value(train_metric),
                step=current_step)
          train_summary_writer.flush()
      if model_dir and current_step % save_steps == 0:
        _save_checkpoint(checkpoint, model_dir,
                         checkpoint_name.format(step=current_step))

      if eval_fn and current_step % save_steps == 0:

        logging.info("Running evaluation after step: %s.", current_step)

        eval_fn(model, current_step, eval_summary_writer)
    if model_dir:
      _save_checkpoint(checkpoint, model_dir,
                       checkpoint_name.format(step=current_step))
    if eval_fn:
      logging.info("Running final evaluation after training is complete.")
      eval_metric = eval_fn(model, current_step, eval_summary_writer)

    training_summary = {
        "total_training_steps": total_training_steps,
        "train_loss": _float_metric_value(train_loss_metric),
    }
    if train_metric:
      training_summary["last_train_metrics"] = _float_metric_value(train_metric)
    if eval_fn:
      # eval_metric is supposed to be a float.
      training_summary["eval_metrics"] = eval_metric

    model_training_utils.write_txt_summary(training_summary, summary_dir)

    return model
