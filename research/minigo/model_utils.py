# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Utilities for DualNet model."""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
import petname
import re

import tensorflow as tf

from tensorflow.python.training.summary_io import SummaryWriterCache

# Regular expression of model number and name.
MODEL_NUM_REGEX = r'^\d{6}'
MODEL_NAME_REGEX = r'^\d{6}(-\w+)+'


def generate(model_num):
  """Generate a full model name for the given model number.

  Args:
    model_num: The number/generation of the model.

  Returns:
    The model's full name: model_num-model_name.
  """
  if model_num == 0:  # Model number for bootstrap model
    new_name = 'bootstrap'
  else:
    new_name = petname.generate()
  full_name = '%06d-%s' % (model_num, new_name)
  return full_name


def detect_model_num(full_name):
  """Take the full name of a model and extract its model number.

  Args:
    full_name: The full name of a model.

  Returns:
    The model number. For example: '000000-bootstrap.index' => 0.
  """
  match = re.match(MODEL_NUM_REGEX, full_name)
  if match:
    return int(match.group())
  else:
    return None


def detect_model_name(full_name):
  """Take the full name of a model and extract its model name.

  Args:
    full_name: The full name of a model.

  Returns:
    The model name. For example: '000000-bootstrap.index' => '000000-bootstrap'.
  """
  match = re.match(MODEL_NAME_REGEX, full_name)
  if match:
    return match.group()
  else:
    return None


def get_models(models_dir):
  """Get all models.

  Args:
    models_dir: The directory of all models.

  Returns:
    A list of model number and names sorted increasingly. For example:
    [(13, 000013-modelname), (17, 000017-modelname), ...etc]
  """
  all_models = tf.gfile.Glob(os.path.join(models_dir, '*.meta'))
  model_filenames = [os.path.basename(m) for m in all_models]
  model_numbers_names = sorted([
      (detect_model_num(m), detect_model_name(m))
      for m in model_filenames])
  return model_numbers_names


def get_latest_model(models_dir):
  """Find the latest model.

  Args:
    models_dir: The directory of all models.

  Returns:
    The model number and name of the latest model. For example:
    (17, 000017-modelname)
  """
  models = get_models(models_dir)
  if models is None:
    models = [(0, '000000-bootstrap')]
  return models[-1]


def compute_update_ratio(weight_tensors, before_weights, after_weights):
  """Compute the ratio of gradient norm to weight norm."""
  deltas = [after - before for after,
            before in zip(after_weights, before_weights)]
  delta_norms = [numpy.linalg.norm(d.ravel()) for d in deltas]
  weight_norms = [numpy.linalg.norm(w.ravel()) for w in before_weights]
  ratios = [d / w for d, w in zip(delta_norms, weight_norms)]
  all_summaries = [
      tf.Summary.Value(tag='update_ratios/' + tensor.name, simple_value=ratio)
      for tensor, ratio in zip(weight_tensors, ratios)]
  return tf.Summary(value=all_summaries)


class UpdateRatioSessionHook(tf.train.SessionRunHook):
  """Update the ratio of gradient norm to weight norm every N steps."""

  def __init__(self, working_dir, every_n_steps=100):
    """Initializes a `UpdateRatioSessionHook`.

    Args:
      working_dir: `str`, the directory to save the update ratios.
      every_n_steps: `int`, update the ratio every N steps.
    """
    self.working_dir = working_dir
    self.every_n_steps = every_n_steps
    self.before_weights = None

  def begin(self):
    """Called once before using the session."""
    self.summary_writer = SummaryWriterCache.get(self.working_dir)
    self.weight_tensors = tf.trainable_variables()
    self.global_step = tf.train.get_or_create_global_step()

  def before_run(self, run_context):
    """Called before each call to run()."""
    global_step = run_context.session.run(self.global_step)
    if global_step % self.every_n_steps == 0:
      self.before_weights = run_context.session.run(self.weight_tensors)

  def after_run(self, run_context, run_values):
    """Called after each call to run()."""
    if self.before_weights is not None:
      after_weights = run_context.session.run(self.weight_tensors)
      weight_update_summaries = compute_update_ratio(
          self.weight_tensors, self.before_weights, after_weights)
      global_step = run_context.session.run(self.global_step)
      self.summary_writer.add_summary(weight_update_summaries, global_step)
      self.before_weights = None


class MiniGoModelParams(object):
  """Parameters for MiniGo."""
  # RL pipeline
  max_games_per_generation = 10  # Number of games per selfplay generation
  max_iters_per_pipeline = 2  # Number of RL iterations in one pipeline

  # dual_net
  # How many positions to look at per generation.
  # Per AGZ, 2048 minibatch * 1k = 2M positions/generation
  examples_per_generation = 2000000

  # for learning rate
  l2_strength = 1e-4  # Regularization strength
  momentum = 0.9  # Momentum used in SGD

  kernel_size = [3, 3]  # kernel size of conv and res blocks is from AGZ paper

  # selfplay
  readouts = 100  # How many simulations to run per move
  verbose = 1  # >=2 will print debug info, >=3 will print boards
  resign_threshold = 0.95  # an Absolute value of threshold to resign at
  holdout_pct = 0.05  # How many games to hold out for validation
  holdout_generation = 50  # How many recent generations/models for holdout data

  # gather
  gather_generation = 50  # How many recent generations/models for gathered data

  # How many positions we should aggregate per 'chunk'.
  examples_per_chunk = 10000
  # How many positions to draw from for our training window.
  # AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
  train_window_size = 125000000

  # evaluation
  eval_games = 10  # The number of games to play in evaluation
  eval_readouts = 100  # How many readouts to make per move in evaluation
  eval_verbose = 1  # How verbose the players should be in evaluation

