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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import app as absl_app  # pylint: disable=unused-import
from absl import flags
from absl import logging
import tensorflow as tf

# pylint: disable=g-bad-import-order
from official.transformer import compute_bleu
from official.transformer.utils import tokenizer
from official.transformer.v2 import data_pipeline
from official.transformer.v2 import metrics
from official.transformer.v2 import misc
from official.transformer.v2 import optimizer
from official.transformer.v2 import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import keras_utils
from official.utils.misc import distribution_utils


INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1


def translate_and_compute_bleu(model, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model, bleu_source, bleu_ref, vocab_file):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, subtokenizer, bleu_source, bleu_ref)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["use_tpu"] = flags_obj.use_tpu
    params["is_tpu_pod"] = flags_obj.is_tpu_pod
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    if params["use_tpu"] and not params["static_batch"]:
      raise ValueError("TPU requires static batch for input data.")
    params["max_length"] = flags_obj.max_length
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training

    if params["dtype"] == tf.float16:
      # TODO(reedwm): It's pretty ugly to set the global policy in a constructor
      # like this. What if multiple instances of TransformerTask are created?
      # We should have a better way in the tf.keras.mixed_precision API of doing
      # this.
      policy = tf.keras.mixed_precision.experimental.Policy(
          "infer_float32_vars")
      tf.keras.mixed_precision.experimental.set_policy(policy)

    # TODO(b/139414143): Move the following logic to distribution utils when
    # remote_eager related code can pass the copybara and being added
    # completely.
    if params["use_tpu"]:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          flags_obj.tpu or "local")
      tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
      self.distribution_strategy = tf.distribute.experimental.TPUStrategy(
          cluster_resolver)
    else:
      self.distribution_strategy = distribution_utils.get_distribution_strategy(
          distribution_strategy=flags_obj.distribution_strategy,
          num_gpus=num_gpus)
      print("Running transformer with num_gpus =", num_gpus)

    if self.distribution_strategy:
      print("For training, using distribution strategy: ",
            self.distribution_strategy)
    else:
      print("Not using any distribution strategy.")

  def train(self):
    """Trains the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, True
    # Sets config options.
    keras_utils.set_session_config(
        enable_xla=flags_obj.enable_xla)

    _ensure_dir(flags_obj.model_dir)
    if self.distribution_strategy:
      with self.distribution_strategy.scope():
        model = transformer.create_model(params, is_train)
        opt = self._create_optimizer()
        if not params["use_ctl"]:
          model.compile(opt)
    else:
      model = transformer.create_model(params, is_train)
      opt = self._create_optimizer()
      model.compile(opt)

    model.summary()

    train_ds = data_pipeline.train_input_fn(params)
    if params["use_tpu"]:
      if params["is_tpu_pod"]:
        train_ds = (
            self.distribution_strategy
            .experimental_distribute_datasets_from_function(
                lambda: data_pipeline.train_input_fn(params)))
      else:
        train_ds = (
            self.distribution_strategy.experimental_distribute_dataset(train_ds)
        )
    else:
      map_data_fn = data_pipeline.map_data_for_transformer_fn
      train_ds = train_ds.map(
          map_data_fn, num_parallel_calls=params["num_parallel_calls"])

    callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

    # TODO(b/139418525): Refactor the custom training loop logic.
    @tf.function
    def train_steps(iterator, steps):
      """Training steps function for TPU runs.

      Args:
        iterator: The input iterator of the training dataset.
        steps: An integer, the number of training steps.

      Returns:
        A float, the loss value.
      """

      def _step_fn(inputs):
        """Per-replica step function."""
        inputs, targets = inputs
        with tf.GradientTape() as tape:
          logits = model([inputs, targets], training=True)
          loss = metrics.transformer_loss(logits, targets,
                                          params["label_smoothing"],
                                          params["vocab_size"])
          # Scales the loss, which results in using the average loss across all
          # of the replicas for backprop.
          scaled_loss = loss / self.distribution_strategy.num_replicas_in_sync

        trainable_vars = model.trainable_variables
        grads = tape.gradient(scaled_loss, trainable_vars)
        opt.apply_gradients(zip(grads, trainable_vars))
        return scaled_loss

      loss = tf.constant(0.0)
      for _ in tf.range(steps):
        per_replica_losses = self.distribution_strategy.experimental_run_v2(
            _step_fn, args=(next(iterator),))
        # Recover the mean of the loss across replicas. E.g.,
        # loss_0 / 2 + loss_1 / 2 = (loss_0 + loss_1) / 2 for two replicas.
        loss = self.distribution_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

      return loss

    if params["use_tpu"]:
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
      if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        logging.info("Loaded checkpoint %s", latest_checkpoint)

    if flags_obj.train_steps < flags_obj.steps_between_evals:
      flags_obj.steps_between_evals = flags_obj.train_steps
    iterations = flags_obj.train_steps // flags_obj.steps_between_evals

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []
    for i in range(1, iterations + 1):
      print("Start train iteration:{}/{}".format(i, iterations))
      history = None
      if params["use_ctl"]:
        if not params["use_tpu"]:
          raise NotImplementedError(
              "Custom training loop on GPUs is not implemented.")
        train_steps_per_eval = tf.convert_to_tensor(
            flags_obj.steps_between_evals, dtype=tf.int32)
        train_loss = train_steps(iter(train_ds),
                                 train_steps_per_eval).numpy().astype(float)
        logging.info("Train Step: %d/%d / loss = %s",
                                  i * flags_obj.steps_between_evals,
                                  flags_obj.train_steps, train_loss)
        checkpoint_name = checkpoint.save(
            os.path.join(
                flags_obj.model_dir,
                "ctl_step_{}.ckpt".format(i * flags_obj.steps_between_evals)))
        logging.info("Saved checkpoint to %s", checkpoint_name)
      else:
        if params["use_tpu"]:
          raise NotImplementedError(
              "Keras model.fit on TPUs is not implemented.")
        history = model.fit(
            train_ds,
            initial_epoch=i - 1,
            epochs=i,
            steps_per_epoch=flags_obj.steps_between_evals,
            callbacks=callbacks,
            # If TimeHistory is enabled, progress bar would be messy. Increase
            # the verbose level to get rid of it.
            verbose=(2 if flags_obj.enable_time_history else 1))
        logging.info("Train history: {}".format(history.history))

      print("End train iteration:{}/{} global step:{}".format(
          i,
          iterations,
          i*flags_obj.steps_between_evals))

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()
        cased_score_history.append([i, cased_score])
        uncased_score_history.append([i, uncased_score])

    stats = ({
        "loss": train_loss
    } if history is None else misc.build_stats(history, callbacks))
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history
    return stats

  def eval(self):
    """Evaluates the model."""
    if not self.predict_model:
      self.predict_model = transformer.create_model(self.params, False)
    self._load_weights_if_possible(
        self.predict_model,
        tf.train.latest_checkpoint(self.flags_obj.model_dir))
    self.predict_model.summary()
    return evaluate_and_log_bleu(self.predict_model,
                                 self.flags_obj.bleu_source,
                                 self.flags_obj.bleu_ref,
                                 self.flags_obj.vocab_file)

  def predict(self):
    """Predicts result from the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, False

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train)
      self._load_weights_if_possible(
          model, tf.train.latest_checkpoint(self.flags_obj.model_dir))
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

    ds = data_pipeline.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    callbacks = misc.get_callbacks()
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      # TODO(b/139414977): Having the same variable restoring method for both
      # TPU and GPU.
      if self.flags_obj.use_tpu:
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=self._create_optimizer())
        checkpoint.restore(init_weight_path)
      else:
        model.load_weights(init_weight_path)
    else:
      print("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    # TODO(b/139414679): Explore the difference between using
    # LearningRateSchedule and callback for GPU runs, and try to merge them.
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule if params["use_tpu"] else params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])
    if params["dtype"] == tf.float16:
      opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          opt, loss_scale=flags_core.get_loss_scale(self.flags_obj,
                                                    default_for_fp16="dynamic"))
    return opt


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)

    def _run_task(task):
      if flags_obj.mode == "train":
        task.train()
      elif flags_obj.mode == "predict":
        task.predict()
      elif flags_obj.mode == "eval":
        task.eval()
      else:
        raise ValueError("Invalid mode {}".format(flags_obj.mode))

    if not flags_obj.use_tpu:
      _run_task(task)
    else:
      primary_cpu_task = ("/job:worker"
                          if flags_obj.use_tpu_2vm_config is not None else "")
      with tf.device(primary_cpu_task):
        _run_task(task)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  misc.define_transformer_flags()
  absl_app.run(main)
