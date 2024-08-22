# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Run NHNet model training and eval."""

import os

# Import libraries

from absl import app
from absl import flags
from absl import logging
from six.moves import zip
import tensorflow as tf, tf_keras

from official.common import distribute_utils
from official.legacy.transformer import metrics as transformer_metrics
from official.modeling.hyperparams import params_dict
from official.projects.nhnet import evaluation
from official.projects.nhnet import input_pipeline
from official.projects.nhnet import models
from official.projects.nhnet import optimizer
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS


def define_flags():
  """Defines command line flags used by NHNet trainer."""
  ## Required parameters
  flags.DEFINE_enum("mode", "train", ["train", "eval", "train_and_eval"],
                    "Execution mode.")
  flags.DEFINE_string("train_file_pattern", "", "Train file pattern.")
  flags.DEFINE_string("eval_file_pattern", "", "Eval file pattern.")
  flags.DEFINE_string(
      "model_dir", None,
      "The output directory where the model checkpoints will be written.")

  # Model training specific flags.
  flags.DEFINE_enum(
      "distribution_strategy", "mirrored", ["tpu", "mirrored"],
      "Distribution Strategy type to use for training. `tpu` uses TPUStrategy "
      "for running on TPUs, `mirrored` uses GPUs with single host.")
  flags.DEFINE_string("tpu", "", "TPU address to connect to.")
  flags.DEFINE_string(
      "init_checkpoint", None,
      "Initial checkpoint (usually from a pre-trained BERT model).")
  flags.DEFINE_integer("train_steps", 100000, "Max train steps")
  flags.DEFINE_integer("eval_steps", 32, "Number of eval steps per run.")
  flags.DEFINE_integer("eval_timeout", 3000, "Timeout waiting for checkpoints.")
  flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
  flags.DEFINE_integer("eval_batch_size", 4, "Total batch size for evaluation.")
  flags.DEFINE_integer(
      "steps_per_loop", 1000,
      "Number of steps per graph-mode loop. Only training step "
      "happens inside the loop.")
  flags.DEFINE_integer("checkpoint_interval", 2000, "Checkpointing interval.")
  flags.DEFINE_integer("len_title", 15, "Title length.")
  flags.DEFINE_integer("len_passage", 200, "Passage length.")
  flags.DEFINE_integer("num_encoder_layers", 12,
                       "Number of hidden layers of encoder.")
  flags.DEFINE_integer("num_decoder_layers", 12,
                       "Number of hidden layers of decoder.")
  flags.DEFINE_string("model_type", "nhnet",
                      "Model type to choose a model configuration.")
  flags.DEFINE_integer(
      "num_nhnet_articles", 5,
      "Maximum number of articles in NHNet, only used when model_type=nhnet")
  flags.DEFINE_string(
      "params_override",
      default=None,
      help=("a YAML/JSON string or a YAML file which specifies additional "
            "overrides over the default parameters"))
  # Enables MLIR-based TF/XLA bridge. This is part of a soft rollout and will
  # eventually be the Google-wide default.
  flags.DEFINE_bool("enable_mlir_bridge", True,
                    "Use MLIR TF/XLA bridge (experimental).")


# pylint: disable=protected-access


class Trainer(tf_keras.Model):
  """A training only model."""

  def __init__(self, model, params):
    super(Trainer, self).__init__()
    self.model = model
    self.params = params
    self._num_replicas_in_sync = tf.distribute.get_strategy(
    ).num_replicas_in_sync

  def call(self, inputs, mode="train"):
    return self.model(inputs, mode)

  def train_step(self, inputs):
    """The logic for one training step."""
    with tf.GradientTape() as tape:
      logits, _, _ = self(inputs, mode="train", training=True)
      targets = models.remove_sos_from_seq(inputs["target_ids"],
                                           self.params.pad_token_id)
      loss = transformer_metrics.transformer_loss(logits, targets,
                                                  self.params.label_smoothing,
                                                  self.params.vocab_size)
      # Scales the loss, which results in using the average loss across all
      # of the replicas for backprop.
      scaled_loss = loss / self._num_replicas_in_sync

    tvars = self.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    self.optimizer.apply_gradients(list(zip(grads, tvars)))
    if isinstance(self.optimizer, tf_keras.optimizers.experimental.Optimizer):
      learning_rate = self.optimizer.learning_rate
    else:
      learning_rate = self.optimizer._decayed_lr(var_dtype=tf.float32)
    return {
        "training_loss": loss,
        "learning_rate": learning_rate,
    }


def train(params, strategy, dataset=None):
  """Runs training."""

  if not dataset:
    dataset = input_pipeline.get_input_dataset(
        FLAGS.train_file_pattern,
        FLAGS.train_batch_size,
        params,
        is_training=True,
        strategy=strategy)

  with strategy.scope():
    model = models.create_model(
        FLAGS.model_type, params, init_checkpoint=FLAGS.init_checkpoint)
    opt = optimizer.create_optimizer(params)
    trainer = Trainer(model, params)

    trainer.compile(
        optimizer=opt,
        steps_per_execution=FLAGS.steps_per_loop)
    summary_dir = os.path.join(FLAGS.model_dir, "summaries")
    summary_callback = tf_keras.callbacks.TensorBoard(
        summary_dir, update_freq=max(100, FLAGS.steps_per_loop))
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=opt, global_step=opt.iterations)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=10,
        step_counter=opt.iterations,
        checkpoint_interval=FLAGS.checkpoint_interval)
    if checkpoint_manager.restore_or_initialize():
      logging.info("Training restored from the checkpoints in: %s",
                   FLAGS.model_dir)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

  # Trains the model.
  steps_per_epoch = min(FLAGS.train_steps, FLAGS.checkpoint_interval)
  epochs = FLAGS.train_steps // steps_per_epoch
  history = trainer.fit(
      x=dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      callbacks=[summary_callback, checkpoint_callback],
      verbose=2)
  train_hist = history.history
  # Gets final loss from training.
  stats = dict(training_loss=float(train_hist["training_loss"][-1]))
  return stats


def run():
  """Runs NHNet using Keras APIs."""
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy, tpu_address=FLAGS.tpu)
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)

  params = models.get_model_params(FLAGS.model_type)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override(
      {
          "len_title":
              FLAGS.len_title,
          "len_passage":
              FLAGS.len_passage,
          "num_hidden_layers":
              FLAGS.num_encoder_layers,
          "num_decoder_layers":
              FLAGS.num_decoder_layers,
          "passage_list":
              [chr(ord("b") + i) for i in range(FLAGS.num_nhnet_articles)],
      },
      is_strict=False)
  stats = {}
  if "train" in FLAGS.mode:
    stats = train(params, strategy)
  if "eval" in FLAGS.mode:
    timeout = 0 if FLAGS.mode == "train_and_eval" else FLAGS.eval_timeout
    # Uses padded decoding for TPU. Always uses cache.
    padded_decode = isinstance(strategy, tf.distribute.TPUStrategy)
    params.override({
        "padded_decode": padded_decode,
    }, is_strict=False)
    stats = evaluation.continuous_eval(
        strategy,
        params,
        model_type=FLAGS.model_type,
        eval_file_pattern=FLAGS.eval_file_pattern,
        batch_size=FLAGS.eval_batch_size,
        eval_steps=FLAGS.eval_steps,
        model_dir=FLAGS.model_dir,
        timeout=timeout)
  return stats


def main(_):
  stats = run()
  if stats:
    logging.info("Stats:\n%s", stats)

if __name__ == "__main__":
  define_flags()
  app.run(main)
