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


import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.wide_deep import movie_dataset
from official.wide_deep import wide_deep_run_loop


def define_movie_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir="/tmp/kaggle-movies/",
                          model_dir='/tmp/movie_model',
                          train_epochs=50,
                          epochs_between_evals=5,
                          batch_size=256)


def run_movie(flags_obj):
  movie_dataset.download_and_extract(flags_obj.data_dir)
  train_input_fn, eval_input_fn, model_column_fn = movie_dataset.get_input_fns(
      flags_obj.data_dir, repeat=flags_obj.epochs_between_evals,
      batch_size=flags_obj.batch_size, small=False
  )

  wide_deep_run_loop.run_loop(
      name="Kaggle Movies", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=model_column_fn,
      flags_obj=flags_obj)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_movie(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_movie_flags()
  absl_app.run(main)
