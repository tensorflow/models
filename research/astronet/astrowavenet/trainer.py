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

"""Script for training and evaluating AstroWaveNet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path

from absl import flags
import tensorflow as tf

from astronet.util import config_util
from astronet.util import configdict
from astronet.util import estimator_runner
from astrowavenet import astrowavenet_model
from astrowavenet import configurations
from astrowavenet.data import kepler_light_curves
from astrowavenet.data import synthetic_transits
from astrowavenet.util import estimator_util


FLAGS = flags.FLAGS

flags.DEFINE_enum("dataset", None,
                  ["synthetic_transits", "kepler_light_curves"],
                  "Dataset for training and/or evaluation.")

flags.DEFINE_string("model_dir", None, "Base output directory.")

flags.DEFINE_string(
    "train_files", None,
    "Comma-separated list of file patterns matching the TFRecord files in the "
    "training dataset.")

flags.DEFINE_string(
    "eval_files", None,
    "Comma-separated list of file patterns matching the TFRecord files in the "
    "evaluation dataset.")

flags.DEFINE_string("config_name", "base",
                    "Name of the AstroWaveNet configuration.")

flags.DEFINE_string(
    "config_overrides", "{}",
    "JSON string or JSON file containing overrides to the base configuration.")

flags.DEFINE_enum("schedule", None,
                  ["train", "train_and_eval", "continuous_eval"],
                  "Schedule for running the model.")

flags.DEFINE_string("eval_name", "val", "Name of the evaluation task.")

flags.DEFINE_integer("train_steps", None, "Total number of steps for training.")

flags.DEFINE_integer("eval_steps", None, "Number of steps for each evaluation.")

flags.DEFINE_integer(
    "local_eval_frequency", 1000,
    "The number of training steps in between evaluation runs. Only applies "
    "when schedule == 'train_and_eval'.")

flags.DEFINE_integer("save_summary_steps", None,
                     "The frequency at which to save model summaries.")

flags.DEFINE_integer("save_checkpoints_steps", None,
                     "The frequency at which to save model checkpoints.")

flags.DEFINE_integer("save_checkpoints_secs", None,
                     "The frequency at which to save model checkpoints.")

flags.DEFINE_integer("keep_checkpoint_max", 1,
                     "The maximum number of model checkpoints to keep.")

# ------------------------------------------------------------------------------
# TPU-only flags
# ------------------------------------------------------------------------------

flags.DEFINE_boolean("use_tpu", False, "Whether to execute on TPU.")

flags.DEFINE_string("master", None, "Address of the TensorFlow TPU master.")

flags.DEFINE_integer("tpu_num_shards", 8, "Number of TPU shards.")

flags.DEFINE_integer("tpu_iterations_per_loop", 1000,
                     "Number of iterations per TPU training loop.")

flags.DEFINE_integer(
    "eval_batch_size", None,
    "Batch size for TPU evaluation. Defaults to the training batch size.")


def _create_run_config():
  """Creates a TPU RunConfig if FLAGS.use_tpu is True, else a RunConfig."""
  session_config = tf.ConfigProto(allow_soft_placement=True)
  run_config_kwargs = {
      "save_summary_steps": FLAGS.save_summary_steps,
      "save_checkpoints_steps": FLAGS.save_checkpoints_steps,
      "save_checkpoints_secs": FLAGS.save_checkpoints_secs,
      "session_config": session_config,
      "keep_checkpoint_max": FLAGS.keep_checkpoint_max
  }

  if FLAGS.use_tpu:
    if not FLAGS.master:
      raise ValueError("FLAGS.master must be set for TPUEstimator.")

    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.tpu_iterations_per_loop,
        num_shards=FLAGS.tpu_num_shards,
        per_host_input_for_training=(FLAGS.tpu_num_shards <= 8))
    run_config = tf.contrib.tpu.RunConfig(
        tpu_config=tpu_config, master=FLAGS.master, **run_config_kwargs)
  else:
    if FLAGS.master:
      raise ValueError("FLAGS.master should only be set for TPUEstimator.")

    run_config = tf.estimator.RunConfig(**run_config_kwargs)

  return run_config


def _get_file_pattern(mode):
  """Gets the value of the file pattern flag for the specified mode."""
  flag_name = ("train_files"
               if mode == tf.estimator.ModeKeys.TRAIN else "eval_files")
  file_pattern = FLAGS[flag_name].value
  if file_pattern is None:
    raise ValueError("--{} is required for mode '{}'".format(flag_name, mode))
  return file_pattern


def _create_dataset_builder(mode, config_overrides=None):
  """Creates a dataset builder for the input pipeline."""
  if FLAGS.dataset == "synthetic_transits":
    return synthetic_transits.SyntheticTransits(config_overrides)

  file_pattern = _get_file_pattern(mode)
  if FLAGS.dataset == "kepler_light_curves":
    builder_class = kepler_light_curves.KeplerLightCurves
  else:
    raise ValueError("Unsupported dataset: {}".format(FLAGS.dataset))

  return builder_class(
      file_pattern,
      mode,
      config_overrides=config_overrides,
      use_tpu=FLAGS.use_tpu)


def _create_input_fn(mode, config_overrides=None):
  """Creates an Estimator input_fn."""
  builder = _create_dataset_builder(mode, config_overrides)
  tf.logging.info("Dataset config for mode '%s': %s", mode,
                  config_util.to_json(builder.config))
  return estimator_util.create_input_fn(builder)


def _create_eval_args(config_overrides=None):
  """Builds eval_args for estimator_runner.evaluate()."""
  if FLAGS.dataset == "synthetic_transits" and not FLAGS.eval_steps:
    raise ValueError("Dataset '{}' requires --eval_steps for evaluation".format(
        FLAGS.dataset))
  input_fn = _create_input_fn(tf.estimator.ModeKeys.EVAL, config_overrides)
  return {FLAGS.eval_name: (input_fn, FLAGS.eval_steps)}


def main(argv):
  del argv  # Unused.

  config = configdict.ConfigDict(configurations.get_config(FLAGS.config_name))
  config_overrides = json.loads(FLAGS.config_overrides)
  for key in config_overrides:
    if key not in ["dataset", "hparams"]:
      raise ValueError("Unrecognized config override: {}".format(key))
  config.hparams.update(config_overrides.get("hparams", {}))

  # Log configs.
  configs_json = [
      ("config_overrides", config_util.to_json(config_overrides)),
      ("config", config_util.to_json(config)),
  ]
  for config_name, config_json in configs_json:
    tf.logging.info("%s: %s", config_name, config_json)

  # Create the estimator.
  run_config = _create_run_config()
  estimator = estimator_util.create_estimator(
      astrowavenet_model.AstroWaveNet, config.hparams, run_config,
      FLAGS.model_dir, FLAGS.eval_batch_size)

  if FLAGS.schedule in ["train", "train_and_eval"]:
    # Save configs.
    tf.gfile.MakeDirs(FLAGS.model_dir)
    for config_name, config_json in configs_json:
      filename = os.path.join(FLAGS.model_dir, "{}.json".format(config_name))
      with tf.gfile.Open(filename, "w") as f:
        f.write(config_json)

    train_input_fn = _create_input_fn(tf.estimator.ModeKeys.TRAIN,
                                      config_overrides.get("dataset"))

    train_hooks = []
    if FLAGS.schedule == "train":
      estimator.train(
          train_input_fn, hooks=train_hooks, max_steps=FLAGS.train_steps)
    else:
      assert FLAGS.schedule == "train_and_eval"

      eval_args = _create_eval_args(config_overrides.get("dataset"))
      for _ in estimator_runner.continuous_train_and_eval(
          estimator=estimator,
          train_input_fn=train_input_fn,
          eval_args=eval_args,
          local_eval_frequency=FLAGS.local_eval_frequency,
          train_hooks=train_hooks,
          train_steps=FLAGS.train_steps):
        # continuous_train_and_eval() yields evaluation metrics after each
        # FLAGS.local_eval_frequency. It also saves and logs them, so we don't
        # do anything here.
        pass

  else:
    assert FLAGS.schedule == "continuous_eval"

    eval_args = _create_eval_args(config_overrides.get("dataset"))
    for _ in estimator_runner.continuous_eval(
        estimator=estimator, eval_args=eval_args,
        train_steps=FLAGS.train_steps):
      # continuous_train_and_eval() yields evaluation metrics after each
      # checkpoint. It also saves and logs them, so we don't do anything here.
      pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)

  flags.mark_flags_as_required(["dataset", "model_dir", "schedule"])

  def _validate_schedule(flag_values):
    """Validates the --schedule flag and the flags it interacts with."""
    schedule = flag_values["schedule"]
    save_checkpoints_steps = flag_values["save_checkpoints_steps"]
    save_checkpoints_secs = flag_values["save_checkpoints_secs"]

    if schedule in ["train", "train_and_eval"]:
      if not (save_checkpoints_steps or save_checkpoints_secs):
        raise flags.ValidationError(
            "--schedule='%s' requires --save_checkpoints_steps or "
            "--save_checkpoints_secs." % schedule)

    return True

  flags.register_multi_flags_validator(
      ["schedule", "save_checkpoints_steps", "save_checkpoints_secs"],
      _validate_schedule)

  tf.app.run()
