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

"""Script for training an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from astronet import models
from astronet.util import config_util
from astronet.util import configdict
from astronet.util import estimator_runner
from astronet.util import estimator_util

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    help="Name of the model and training configuration. Exactly one of "
    "--config_name or --config_json is required.")

parser.add_argument(
    "--config_json",
    type=str,
    help="JSON string or JSON file containing the model and training "
    "configuration. Exactly one of --config_name or --config_json is required.")

parser.add_argument(
    "--train_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the training dataset.")

parser.add_argument(
    "--eval_files",
    type=str,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the validation dataset.")

parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory for model checkpoints and summaries.")

parser.add_argument(
    "--train_steps",
    type=int,
    default=10000,
    help="Total number of steps to train the model for.")

parser.add_argument(
    "--shuffle_buffer_size",
    type=int,
    default=15000,
    help="Size of the shuffle buffer for the training dataset.")


def main(_):
  model_class = models.get_model_class(FLAGS.model)

  # Look up the model configuration.
  assert (FLAGS.config_name is None) != (FLAGS.config_json is None), (
      "Exactly one of --config_name or --config_json is required.")
  config = (
      models.get_model_config(FLAGS.model, FLAGS.config_name)
      if FLAGS.config_name else config_util.parse_json(FLAGS.config_json))

  config = configdict.ConfigDict(config)
  config_util.log_and_save_config(config, FLAGS.model_dir)

  # Create the estimator.
  run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)
  estimator = estimator_util.create_estimator(model_class, config.hparams,
                                              run_config, FLAGS.model_dir)

  # Create an input function that reads the training dataset. We iterate through
  # the dataset once at a time if we are alternating with evaluation, otherwise
  # we iterate infinitely.
  train_input_fn = estimator_util.create_input_fn(
      file_pattern=FLAGS.train_files,
      input_config=config.inputs,
      mode=tf.estimator.ModeKeys.TRAIN,
      shuffle_values_buffer=FLAGS.shuffle_buffer_size,
      repeat=1 if FLAGS.eval_files else None)

  if not FLAGS.eval_files:
    estimator.train(train_input_fn, max_steps=FLAGS.train_steps)
  else:
    eval_input_fn = estimator_util.create_input_fn(
        file_pattern=FLAGS.eval_files,
        input_config=config.inputs,
        mode=tf.estimator.ModeKeys.EVAL)
    eval_args = {
        "val": (eval_input_fn, None)  # eval_name: (input_fn, eval_steps)
    }

    for _ in estimator_runner.continuous_train_and_eval(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_args=eval_args,
        train_steps=FLAGS.train_steps):
      # continuous_train_and_eval() yields evaluation metrics after each
      # training epoch. We don't do anything here.
      pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
