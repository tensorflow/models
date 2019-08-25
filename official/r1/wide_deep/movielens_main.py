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

from official.datasets import movielens
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.r1.wide_deep import movielens_dataset
from official.r1.wide_deep import wide_deep_run_loop


def define_movie_flags():
  """Define flags for movie dataset training."""
  wide_deep_run_loop.define_wide_deep_flags()
  flags.DEFINE_enum(
      name="dataset", default=movielens.ML_1M,
      enum_values=movielens.DATASETS, case_sensitive=False,
      help=flags_core.help_wrap("Dataset to be trained and evaluated."))
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir="/tmp/movielens-data/",
                          model_dir='/tmp/movie_model',
                          model_type="deep",
                          train_epochs=50,
                          epochs_between_evals=5,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=256)

  @flags.validator("stop_threshold",
                   message="stop_threshold not supported for movielens model")
  def _no_stop(stop_threshold):
    return stop_threshold is None


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  if model_type != "deep":
    raise NotImplementedError("movie dataset only supports `deep` model_type")
  _, deep_columns = model_column_fn()
  hidden_units = [256, 256, 256, 128]

  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0},
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))
  return tf.estimator.DNNRegressor(
      model_dir=model_dir,
      feature_columns=deep_columns,
      hidden_units=hidden_units,
      optimizer=tf.train.AdamOptimizer(),
      activation_fn=tf.nn.sigmoid,
      dropout=0.3,
      loss_reduction=tf.losses.Reduction.MEAN)


def run_movie(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """

  if flags_obj.download_if_missing:
    movielens.download(dataset=flags_obj.dataset, data_dir=flags_obj.data_dir)

  train_input_fn, eval_input_fn, model_column_fn = \
    movielens_dataset.construct_input_fns(
        dataset=flags_obj.dataset, data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size, repeat=flags_obj.epochs_between_evals)

  tensors_to_log = {
      'loss': '{loss_prefix}head/weighted_loss/value'
  }

  wide_deep_run_loop.run_loop(
      name="MovieLens", train_input_fn=train_input_fn,
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
