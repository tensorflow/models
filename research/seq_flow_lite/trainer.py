# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# Lint as: python3
"""A utility for PRADO model to do train, eval, inference and model export."""

import importlib
import json

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import input_fn_reader # import root module
import metric_functions # import root module

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("config_path", None, "Path to a RunnerConfig.")
flags.DEFINE_enum("runner_mode", None, ["train", "train_and_eval", "eval"],
                  "Runner mode.")
flags.DEFINE_string("master", None, "TensorFlow master URL.")
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def load_runner_config():
  with tf.gfile.GFile(FLAGS.config_path, "r") as f:
    return json.loads(f.read())


def create_model(model, model_config, features, mode):
  """Creates a sequence labeling model."""
  keras_model = model.Encoder(model_config, mode)
  logits = keras_model(features["projection"], features["seq_length"])
  if not model_config["multilabel"]:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=features["label"], logits=logits)
  else:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=features["label"], logits=logits)
  loss = tf.reduce_mean(loss)
  loss += tf.add_n(keras_model.losses)

  return (loss, logits)


def create_optimizer(loss, runner_config):
  """Returns a train_op using Adam optimizer."""
  learning_rate = tf.train.exponential_decay(
      learning_rate=runner_config["learning_rate"],
      global_step=tf.train.get_global_step(),
      decay_steps=runner_config["learning_rate_decay_steps"],
      decay_rate=runner_config["learning_rate_decay_rate"],
      staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  if FLAGS.use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  return optimizer.minimize(loss, global_step=tf.train.get_global_step())


def model_fn_builder(runner_config):
  """Returns `model_fn` closure for TPUEstimator."""

  rel_module_path = "" # empty base dir
  model = importlib.import_module(rel_module_path + runner_config["name"])

  def model_fn(features, mode, params):
    """The `model_fn` for TPUEstimator."""
    del params
    label_ids = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      label_ids = features["label"]

    model_config = runner_config["model_config"]
    loss, logits = create_model(model, model_config, features, mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = create_optimizer(loss, runner_config)
      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      if not runner_config["model_config"]["multilabel"]:
        metric_fn = metric_functions.classification_metric
      else:
        metric_fn = metric_functions.labeling_metric

      eval_metrics = (metric_fn, [loss, label_ids, logits])
      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, eval_metrics=eval_metrics)
    else:
      assert False, "Expected to be called in TRAIN or EVAL mode."

  return model_fn


def main(_):
  runner_config = load_runner_config()

  if FLAGS.output_dir:
    tf.gfile.MakeDirs(FLAGS.output_dir)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=runner_config["save_checkpoints_steps"],
      keep_checkpoint_max=20,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=runner_config["iterations_per_loop"],
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(runner_config)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  batch_size = runner_config["batch_size"]
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size)

  if FLAGS.runner_mode == "train":
    train_input_fn = input_fn_reader.create_input_fn(
        runner_config=runner_config,
        mode=tf.estimator.ModeKeys.TRAIN,
        drop_remainder=True)
    estimator.train(
        input_fn=train_input_fn, max_steps=runner_config["train_steps"])
  elif FLAGS.runner_mode == "eval":
    # TPU needs fixed shapes, so if the last batch is smaller, we drop it.
    eval_input_fn = input_fn_reader.create_input_fn(
        runner_config=runner_config,
        mode=tf.estimator.ModeKeys.EVAL,
        drop_remainder=True)

    for _ in tf.train.checkpoints_iterator(FLAGS.output_dir, timeout=600):
      result = estimator.evaluate(input_fn=eval_input_fn)
      for key in sorted(result):
        logging.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
  app.run(main)
