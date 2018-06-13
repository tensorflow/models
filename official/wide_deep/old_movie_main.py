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
"""Train DNN on Kaggle movie dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.wide_deep import old_movie_dataset
from official.wide_deep import movielens_dataset
from official.wide_deep import wide_deep_run_loop


def define_movie_flags():
  """Define flags for movie dataset training."""
  wide_deep_run_loop.define_wide_deep_flags()
  flags.DEFINE_boolean(
      name="small_dataset", short_name="small", default=False,
      help=flags_core.help_wrap("Use a smaller dataset (~45K examples) instead "
                                "of the full (~12M examples) dataset."))
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir="/tmp/kaggle-movies/",
                          model_dir='/tmp/movie_model',
                          model_type="deep",
                          train_epochs=50,
                          epochs_between_evals=5,
                          batch_size=256)

  @flags.validator("stop_threshold",
                   message="stop_threshold not supported for movie model")
  def _no_stop(stop_threshold):
    return stop_threshold is None


def build_estimator(model_dir, model_type, model_column_fn):
  """Build an estimator appropriate for the given model type."""
  if model_type != "deep":
    raise NotImplementedError("movie dataset only supports `deep` model_type")
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [128, 128, 128]

  return tf.estimator.DNNRegressor(
      model_dir=model_dir,
      feature_columns=deep_columns,
      hidden_units=hidden_units,
      optimizer=tf.train.AdamOptimizer(),
      dropout=0.5,
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


def run_movie(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """

  # if flags_obj.download_if_missing:
  #   old_movie_dataset.download_and_extract(flags_obj.data_dir)
  #
  # train_input_fn, eval_input_fn, model_column_fn = old_movie_dataset.get_input_fns(
  #     flags_obj.data_dir, repeat=flags_obj.epochs_between_evals,
  #     batch_size=flags_obj.batch_size, small=flags_obj.small_dataset
  # )
  train_input_fn, eval_input_fn, model_column_fn = movielens_dataset.construct_input_fns(dataset="ml-1m", data_dir="/tmp/movielens-data/")

  tensors_to_log = {
      'loss': '{loss_prefix}head/weighted_loss/value'
  }

  wide_deep_run_loop.run_loop(
      name="Kaggle Movies", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=model_column_fn,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=False)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_movie(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_movie_flags()
  absl_app.run(main)
