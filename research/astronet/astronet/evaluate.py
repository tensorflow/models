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

"""Script for evaluating an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from astronet import models
from astronet.util import config_util
from astronet.util import configdict
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
    "--eval_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the evaluation dataset.")

parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory containing a model checkpoint.")

parser.add_argument(
    "--eval_name", type=str, default="test", help="Name of the evaluation set.")


def main(_):
  model_class = models.get_model_class(FLAGS.model)

  # Look up the model configuration.
  assert (FLAGS.config_name is None) != (FLAGS.config_json is None), (
      "Exactly one of --config_name or --config_json is required.")
  config = (
      models.get_model_config(FLAGS.model, FLAGS.config_name)
      if FLAGS.config_name else config_util.parse_json(FLAGS.config_json))

  config = configdict.ConfigDict(config)

  # Create the estimator.
  estimator = estimator_util.create_estimator(
      model_class, config.hparams, model_dir=FLAGS.model_dir)

  # Create an input function that reads the evaluation dataset.
  input_fn = estimator_util.create_input_fn(
      file_pattern=FLAGS.eval_files,
      input_config=config.inputs,
      mode=tf.estimator.ModeKeys.EVAL)

  # Run evaluation. This will log the result to stderr and also write a summary
  # file in the model_dir.
  estimator_util.evaluate(estimator, input_fn, eval_name=FLAGS.eval_name)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
