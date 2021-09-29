# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Binary to train PRADO model with TF 2.0."""

import importlib
import json

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import input_fn_reader # import root module

FLAGS = flags.FLAGS

flags.DEFINE_string("config_path", None, "Path to a RunnerConfig.")
flags.DEFINE_enum("runner_mode", "train", ["train", "train_and_eval", "eval"],
                  "Runner mode.")
flags.DEFINE_string("master", None, "TensorFlow master URL.")
flags.DEFINE_string(
    "output_dir", "/tmp/testV2",
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def load_runner_config():
  with tf.io.gfile.GFile(FLAGS.config_path, "r") as f:
    return json.loads(f.read())


def compute_loss(logits, labels, model_config, mode):
  """Creates a sequence labeling model."""
  if mode != tf.estimator.ModeKeys.PREDICT:
    if not model_config["multilabel"]:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    else:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
  else:
    loss = None

  return loss


def model_fn_builder(runner_config, mode):
  """Returns `model_fn` closure for TPUEstimator."""

  rel_module_path = "" # empty base dir
  model = importlib.import_module(rel_module_path + runner_config["name"])
  model_config = runner_config["model_config"]
  return model.Encoder(model_config, mode)


def main(_):
  runner_config = load_runner_config()

  if FLAGS.output_dir:
    tf.io.gfile.makedirs(FLAGS.output_dir)

  train_model = model_fn_builder(runner_config, tf.estimator.ModeKeys.TRAIN)
  optimizer = tf.keras.optimizers.Adam()
  train_input_fn = input_fn_reader.create_input_fn(
      runner_config=runner_config,
      mode=tf.estimator.ModeKeys.TRAIN,
      drop_remainder=True)
  params = {"batch_size": runner_config["batch_size"]}
  train_ds = train_input_fn(params)
  train_loss = tf.keras.metrics.Mean(name="train_loss")

  @tf.function
  def train_step(features):
    with tf.GradientTape() as tape:
      logits = train_model(features["projection"], features["seq_length"])
      loss = compute_loss(logits, features["label"],
                          runner_config["model_config"],
                          tf.estimator.ModeKeys.TRAIN)
    gradients = tape.gradient(loss, train_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))
    train_loss(loss)

  for epoch in range(1):
    train_loss.reset_states()
    for features in train_ds:
      train_step(features)
      step = optimizer.iterations.numpy()
      if step % 100 == 0:
        logging.info("Running step %s in epoch %s", step, epoch)
        logging.info("Training loss: %s, epoch: %s, step: %s",
                     round(train_loss.result().numpy(), 4), epoch, step)


if __name__ == "__main__":
  app.run(main)
