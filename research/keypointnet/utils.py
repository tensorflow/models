# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Utility functions for KeypointNet.

These are helper / tensorflow related functions. The actual implementation and
algorithm is in main.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import traceback


class TrainingHook(tf.train.SessionRunHook):
  """A utility for displaying training information such as the loss, percent
  completed, estimated finish date and time."""

  def __init__(self, steps):
    self.steps = steps

    self.last_time = time.time()
    self.last_est = self.last_time

    self.eta_interval = int(math.ceil(0.1 * self.steps))
    self.current_interval = 0

  def before_run(self, run_context):
    graph = tf.get_default_graph()
    return tf.train.SessionRunArgs(
        {"loss": graph.get_collection("total_loss")[0]})

  def after_run(self, run_context, run_values):
    step = run_context.session.run(tf.train.get_global_step())
    now = time.time()

    if self.current_interval < self.eta_interval:
      self.duration = now - self.last_est
      self.current_interval += 1
    if step % self.eta_interval == 0:
      self.duration = now - self.last_est
      self.last_est = now

    eta_time = float(self.steps - step) / self.current_interval * \
        self.duration
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta = "%d:%02d:%02d" % (h, m, s)

    print("%.2f%% (%d/%d): %.3e t %.3f  @ %s (%s)" % (
        step * 100.0 / self.steps,
        step,
        self.steps,
        run_values.results["loss"],
        now - self.last_time,
        time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
        eta))

    self.last_time = now


def standard_model_fn(
    func, steps, run_config=None, sync_replicas=0, optimizer_fn=None):
  """Creates model_fn for tf.Estimator.

  Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
    sync_replicas: The number of replicas used to compute gradient for
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

  Returns:
    model_fn for tf.estimator.Estimator.
  """

  def fn(features, labels, mode, params):
    """Returns model_fn for tf.estimator.Estimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    ret = func(features, labels, mode, params)

    tf.add_to_collection("total_loss", ret["loss"])
    train_op = None

    training_hooks = []
    if is_training:
      training_hooks.append(TrainingHook(steps))

      if optimizer_fn is None:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
      else:
        optimizer = optimizer_fn

      if run_config is not None and run_config.num_worker_replicas > 1:
        sr = sync_replicas
        if sr <= 0:
          sr = run_config.num_worker_replicas

        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=sr,
            total_num_replicas=run_config.num_worker_replicas)

        training_hooks.append(
            optimizer.make_session_run_hook(
                run_config.is_chief, num_tokens=run_config.num_worker_replicas))

      optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
      train_op = slim.learning.create_train_op(ret["loss"], optimizer)

    if "eval_metric_ops" not in ret:
      ret["eval_metric_ops"] = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=ret["predictions"],
        loss=ret["loss"],
        train_op=train_op,
        eval_metric_ops=ret["eval_metric_ops"],
        training_hooks=training_hooks)
  return fn


def train_and_eval(
    model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    keep_checkpoint_every_n_hours=0.5,
    save_checkpoints_secs=180,
    save_summary_steps=50,
    eval_steps=20,
    eval_start_delay_secs=10,
    eval_throttle_secs=300,
    sync_replicas=0):
  """Trains and evaluates our model. Supports local and distributed training.

  Args:
    model_dir: The output directory for trained parameters, checkpoints, etc.
    steps: Training steps.
    batch_size: Batch size.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    hparams: tf.HParams containing a set of hyperparameters.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    save_summary_steps: Save summaries every this many steps.
    eval_steps: Number of steps to evaluate model.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago
    sync_replicas: Number of synchronous replicas for distributed training.

  Returns:
    None
  """

  run_config = tf.estimator.RunConfig(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_checkpoints_secs=save_checkpoints_secs,
      save_summary_steps=save_summary_steps)

  estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          sync_replicas=sync_replicas),
      params=hparams, config=run_config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(split="train", batch_size=batch_size),
      max_steps=steps)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(split="validation", batch_size=batch_size),
      steps=eval_steps,
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def draw_circle(rgb, u, v, col, r):
  """Draws a simple anti-aliasing circle in-place.

  Args:
    rgb: Input image to be modified.
    u: Horizontal coordinate.
    v: Vertical coordinate.
    col: Color.
    r: Radius.
  """

  ir = int(math.ceil(r))
  for i in range(-ir-1, ir+2):
    for j in range(-ir-1, ir+2):
      nu = int(round(u + i))
      nv = int(round(v + j))
      if nu < 0 or nu >= rgb.shape[1] or nv < 0 or nv >= rgb.shape[0]:
        continue

      du = abs(nu - u)
      dv = abs(nv - v)

      # need sqrt to keep scale
      t = math.sqrt(du * du + dv * dv) - math.sqrt(r * r)
      if t < 0:
        rgb[nv, nu, :] = col
      else:
        t = 1 - t
        if t > 0:
          # t = t ** 0.3
          rgb[nv, nu, :] = col * t + rgb[nv, nu, :] * (1-t)


def draw_ndc_points(rgb, xy, cols):
  """Draws keypoints onto an input image.

  Args:
    rgb: Input image to be modified.
    xy: [n x 2] matrix of 2D locations.
    cols: A list of colors for the keypoints.
  """

  vh, vw = rgb.shape[0], rgb.shape[1]

  for j in range(len(cols)):
    x, y = xy[j, :2]
    x = (min(max(x, -1), 1) * vw / 2 + vw / 2) - 0.5
    y = vh - 0.5 - (min(max(y, -1), 1) * vh / 2 + vh / 2)

    x = int(round(x))
    y = int(round(y))
    if x < 0 or y < 0 or x >= vw or y >= vh:
      continue

    rad = 1.5
    rad *= rgb.shape[0] / 128.0
    draw_circle(rgb, x, y, np.array([0.0, 0.0, 0.0, 1.0]), rad * 1.5)
    draw_circle(rgb, x, y, cols[j], rad)


def colored_hook(home_dir):
  """Colorizes python's error message.

  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook
