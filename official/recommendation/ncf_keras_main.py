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
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# pylint: disable=g-bad-import-order
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants as rconst
from official.recommendation import movielens
from official.recommendation import ncf_common
from official.recommendation import ncf_input_pipeline
from official.recommendation import neumf_model
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS


def metric_fn(logits, dup_mask, params):
  dup_mask = tf.cast(dup_mask, tf.float32)
  logits = tf.slice(logits, [0, 1], [-1, -1])
  in_top_k, _, metric_weights, _ = neumf_model.compute_top_k_and_ndcg(
      logits,
      dup_mask,
      params["match_mlperf"])
  metric_weights = tf.cast(metric_weights, tf.float32)
  return in_top_k, metric_weights


class MetricLayer(tf.keras.layers.Layer):
  """Custom layer of metrics for NCF model."""

  def __init__(self, params):
    super(MetricLayer, self).__init__()
    self.params = params

  def call(self, inputs, training=False):
    logits, dup_mask = inputs

    if training:
      hr_sum = 0.0
      hr_count = 0.0
    else:
      metric, metric_weights = metric_fn(logits, dup_mask, self.params)
      hr_sum = tf.reduce_sum(metric * metric_weights)
      hr_count = tf.reduce_sum(metric_weights)

    self.add_metric(hr_sum, name="hr_sum", aggregation="mean")
    self.add_metric(hr_count, name="hr_count", aggregation="mean")
    return logits


class LossLayer(tf.keras.layers.Layer):
  """Pass-through loss layer for NCF model."""

  def __init__(self, loss_normalization_factor):
    # The loss may overflow in float16, so we use float32 instead.
    super(LossLayer, self).__init__(dtype="float32")
    self.loss_normalization_factor = loss_normalization_factor
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="sum")

  def call(self, inputs):
    logits, labels, valid_pt_mask_input = inputs
    loss = self.loss(
        y_true=labels, y_pred=logits, sample_weight=valid_pt_mask_input)
    loss = loss * (1.0 / self.loss_normalization_factor)
    self.add_loss(loss)
    return logits


class IncrementEpochCallback(tf.keras.callbacks.Callback):
  """A callback to increase the requested epoch for the data producer.

  The reason why we need this is because we can only buffer a limited amount of
  data. So we keep a moving window to represent the buffer. This is to move the
  one of the window's boundaries for each epoch.
  """

  def __init__(self, producer):
    self._producer = producer

  def on_epoch_begin(self, epoch, logs=None):
    self._producer.increment_request_epoch()


class CustomEarlyStopping(tf.keras.callbacks.Callback):
  """Stop training has reached a desired hit rate."""

  def __init__(self, monitor, desired_value):
    super(CustomEarlyStopping, self).__init__()

    self.monitor = monitor
    self.desired = desired_value
    self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current and current >= self.desired:
      self.stopped_epoch = epoch
      self.model.stop_training = True

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning("Early stopping conditioned on metric `%s` "
                      "which is not available. Available metrics are: %s",
                      self.monitor, ",".join(list(logs.keys())))
    return monitor_value


def _get_keras_model(params):
  """Constructs and returns the model."""
  batch_size = params["batch_size"]

  user_input = tf.keras.layers.Input(
      shape=(1,), name=movielens.USER_COLUMN, dtype=tf.int32)

  item_input = tf.keras.layers.Input(
      shape=(1,), name=movielens.ITEM_COLUMN, dtype=tf.int32)

  valid_pt_mask_input = tf.keras.layers.Input(
      shape=(1,), name=rconst.VALID_POINT_MASK, dtype=tf.bool)

  dup_mask_input = tf.keras.layers.Input(
      shape=(1,), name=rconst.DUPLICATE_MASK, dtype=tf.int32)

  label_input = tf.keras.layers.Input(
      shape=(1,), name=rconst.TRAIN_LABEL_KEY, dtype=tf.bool)

  base_model = neumf_model.construct_model(user_input, item_input, params)

  logits = base_model.output

  zeros = tf.keras.layers.Lambda(
      lambda x: x * 0)(logits)

  softmax_logits = tf.keras.layers.concatenate(
      [zeros, logits],
      axis=-1)

  # Custom training loop calculates loss and metric as a part of
  # training/evaluation step function.
  if not params["keras_use_ctl"]:
    softmax_logits = MetricLayer(params)([softmax_logits, dup_mask_input])
    # TODO(b/134744680): Use model.add_loss() instead once the API is well
    # supported.
    softmax_logits = LossLayer(batch_size)(
        [softmax_logits, label_input, valid_pt_mask_input])

  keras_model = tf.keras.Model(
      inputs={
          movielens.USER_COLUMN: user_input,
          movielens.ITEM_COLUMN: item_input,
          rconst.VALID_POINT_MASK: valid_pt_mask_input,
          rconst.DUPLICATE_MASK: dup_mask_input,
          rconst.TRAIN_LABEL_KEY: label_input},
      outputs=softmax_logits)

  keras_model.summary()
  return keras_model


def run_ncf(_):
  """Run NCF training and eval with Keras."""

  keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

  if FLAGS.seed is not None:
    print("Setting tf seed")
    tf.random.set_seed(FLAGS.seed)

  model_helpers.apply_clean(FLAGS)

  if FLAGS.dtype == "fp16" and FLAGS.fp16_implementation == "keras":
    policy = tf.keras.mixed_precision.experimental.Policy(
        "mixed_float16",
        loss_scale=flags_core.get_loss_scale(FLAGS, default_for_fp16="dynamic"))
    tf.keras.mixed_precision.experimental.set_policy(policy)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)

  params = ncf_common.parse_flags(FLAGS)
  params["distribute_strategy"] = strategy

  if not keras_utils.is_v2_0() and strategy is not None:
    logging.error("NCF Keras only works with distribution strategy in TF 2.0")
    return
  if (params["keras_use_ctl"] and (
      not keras_utils.is_v2_0() or strategy is None)):
    logging.error(
        "Custom training loop only works with tensorflow 2.0 and dist strat.")
    return
  if params["use_tpu"] and not params["keras_use_ctl"]:
    logging.error("Custom training loop must be used when using TPUStrategy.")
    return

  batch_size = params["batch_size"]
  time_callback = keras_utils.TimeHistory(batch_size, FLAGS.log_steps)
  callbacks = [time_callback]

  producer, input_meta_data = None, None
  generate_input_online = params["train_dataset_path"] is None

  if generate_input_online:
    # Start data producing thread.
    num_users, num_items, _, _, producer = ncf_common.get_inputs(params)
    producer.start()
    per_epoch_callback = IncrementEpochCallback(producer)
    callbacks.append(per_epoch_callback)
  else:
    assert params["eval_dataset_path"] and params["input_meta_data_path"]
    with tf.io.gfile.GFile(params["input_meta_data_path"], "rb") as reader:
      input_meta_data = json.loads(reader.read().decode("utf-8"))
      num_users = input_meta_data["num_users"]
      num_items = input_meta_data["num_items"]

  params["num_users"], params["num_items"] = num_users, num_items

  if FLAGS.early_stopping:
    early_stopping_callback = CustomEarlyStopping(
        "val_HR_METRIC", desired_value=FLAGS.hr_threshold)
    callbacks.append(early_stopping_callback)

  (train_input_dataset, eval_input_dataset,
   num_train_steps, num_eval_steps) = \
    (ncf_input_pipeline.create_ncf_input_data(
        params, producer, input_meta_data, strategy))
  steps_per_epoch = None if generate_input_online else num_train_steps

  with distribution_utils.get_strategy_scope(strategy):
    keras_model = _get_keras_model(params)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"],
        beta_1=params["beta1"],
        beta_2=params["beta2"],
        epsilon=params["epsilon"])
    if FLAGS.fp16_implementation == "graph_rewrite":
      optimizer = \
        tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
            optimizer,
            loss_scale=flags_core.get_loss_scale(FLAGS,
                                                 default_for_fp16="dynamic"))
    elif FLAGS.dtype == "fp16" and params["keras_use_ctl"]:
      # When keras_use_ctl is False, instead Model.fit() automatically applies
      # loss scaling so we don't need to create a LossScaleOptimizer.
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          optimizer,
          tf.keras.mixed_precision.experimental.global_policy().loss_scale)

    if params["keras_use_ctl"]:
      train_loss, eval_results = run_ncf_custom_training(
          params,
          strategy,
          keras_model,
          optimizer,
          callbacks,
          train_input_dataset,
          eval_input_dataset,
          num_train_steps,
          num_eval_steps,
          generate_input_online=generate_input_online)
    else:
      # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
      # a valid arg for this model. Also remove as a valid flag.
      if FLAGS.force_v2_in_keras_compile is not None:
        keras_model.compile(
            optimizer=optimizer,
            run_eagerly=FLAGS.run_eagerly,
            experimental_run_tf_function=FLAGS.force_v2_in_keras_compile)
      else:
        keras_model.compile(optimizer=optimizer, run_eagerly=FLAGS.run_eagerly)

      if not FLAGS.ml_perf:
        # Create Tensorboard summary and checkpoint callbacks.
        summary_dir = os.path.join(FLAGS.model_dir, "summaries")
        summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
        checkpoint_path = os.path.join(FLAGS.model_dir, "checkpoint")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True)

        callbacks += [summary_callback, checkpoint_callback]

      history = keras_model.fit(
          train_input_dataset,
          epochs=FLAGS.train_epochs,
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks,
          validation_data=eval_input_dataset,
          validation_steps=num_eval_steps,
          verbose=2)

      logging.info("Training done. Start evaluating")

      eval_loss_and_metrics = keras_model.evaluate(
          eval_input_dataset, steps=num_eval_steps, verbose=2)

      logging.info("Keras evaluation is done.")

      # Keras evaluate() API returns scalar loss and metric values from
      # evaluation as a list. Here, the returned list would contain
      # [evaluation loss, hr sum, hr count].
      eval_hit_rate = eval_loss_and_metrics[1] / eval_loss_and_metrics[2]

      # Format evaluation result into [eval loss, eval hit accuracy].
      eval_results = [eval_loss_and_metrics[0], eval_hit_rate]

      if history and history.history:
        train_history = history.history
        train_loss = train_history["loss"][-1]

  stats = build_stats(train_loss, eval_results, time_callback)
  return stats


def run_ncf_custom_training(params,
                            strategy,
                            keras_model,
                            optimizer,
                            callbacks,
                            train_input_dataset,
                            eval_input_dataset,
                            num_train_steps,
                            num_eval_steps,
                            generate_input_online=True):
  """Runs custom training loop.

  Args:
    params: Dictionary containing training parameters.
    strategy: Distribution strategy to be used for distributed training.
    keras_model: Model used for training.
    optimizer: Optimizer used for training.
    callbacks: Callbacks to be invoked between batches/epochs.
    train_input_dataset: tf.data.Dataset used for training.
    eval_input_dataset: tf.data.Dataset used for evaluation.
    num_train_steps: Total number of steps to run for training.
    num_eval_steps: Total number of steps to run for evaluation.
    generate_input_online: Whether input data was generated by data producer.
      When data is generated by data producer, then train dataset must be
      re-initialized after every epoch.

  Returns:
    A tuple of train loss and a list of training and evaluation results.
  """
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction="sum", from_logits=True)
  train_input_iterator = iter(
      strategy.experimental_distribute_dataset(train_input_dataset))

  def train_step(train_iterator):
    """Called once per step to train the model."""

    def step_fn(features):
      """Computes loss and applied gradient per replica."""
      with tf.GradientTape() as tape:
        softmax_logits = keras_model(features)
        # The loss can overflow in float16, so we cast to float32.
        softmax_logits = tf.cast(softmax_logits, "float32")
        labels = features[rconst.TRAIN_LABEL_KEY]
        loss = loss_object(
            labels,
            softmax_logits,
            sample_weight=features[rconst.VALID_POINT_MASK])
        loss *= (1.0 / params["batch_size"])
        if FLAGS.dtype == "fp16":
          loss = optimizer.get_scaled_loss(loss)

      grads = tape.gradient(loss, keras_model.trainable_variables)
      if FLAGS.dtype == "fp16":
        grads = optimizer.get_unscaled_gradients(grads)
      # Converting gradients to dense form helps in perf on GPU for NCF
      grads = neumf_model.sparse_to_dense_grads(
          list(zip(grads, keras_model.trainable_variables)))
      optimizer.apply_gradients(grads)
      return loss

    per_replica_losses = strategy.experimental_run_v2(
        step_fn, args=(next(train_iterator),))
    mean_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return mean_loss

  def eval_step(eval_iterator):
    """Called once per eval step to compute eval metrics."""

    def step_fn(features):
      """Computes eval metrics per replica."""
      softmax_logits = keras_model(features)
      in_top_k, metric_weights = metric_fn(softmax_logits,
                                           features[rconst.DUPLICATE_MASK],
                                           params)
      hr_sum = tf.reduce_sum(in_top_k * metric_weights)
      hr_count = tf.reduce_sum(metric_weights)
      return hr_sum, hr_count

    per_replica_hr_sum, per_replica_hr_count = (
        strategy.experimental_run_v2(
            step_fn, args=(next(eval_iterator),)))
    hr_sum = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_hr_sum, axis=None)
    hr_count = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_hr_count, axis=None)
    return hr_sum, hr_count

  if not FLAGS.run_eagerly:
    train_step = tf.function(train_step)
    eval_step = tf.function(eval_step)

  for callback in callbacks:
    callback.on_train_begin()

  # Not writing tensorboard summaries if running in MLPerf.
  if FLAGS.ml_perf:
    eval_summary_writer, train_summary_writer = None, None
  else:
    summary_dir = os.path.join(FLAGS.model_dir, "summaries")
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, "eval"))
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, "train"))

  train_loss = 0
  for epoch in range(FLAGS.train_epochs):
    for cb in callbacks:
      cb.on_epoch_begin(epoch)

    # As NCF dataset is sampled with randomness, not repeating
    # data elements in each epoch has significant impact on
    # convergence. As so, offline-generated TF record files
    # contains all epoch worth of data. Thus we do not need
    # to initialize dataset when reading from tf record files.
    if generate_input_online:
      train_input_iterator = iter(
          strategy.experimental_distribute_dataset(train_input_dataset))

    train_loss = 0
    for step in range(num_train_steps):
      current_step = step + epoch * num_train_steps
      for c in callbacks:
        c.on_batch_begin(current_step)

      train_loss += train_step(train_input_iterator)

      # Write train loss once in every 1000 steps.
      if train_summary_writer and step % 1000 == 0:
        with train_summary_writer.as_default():
          tf.summary.scalar("training_loss", train_loss/(step + 1),
                            step=current_step)

      for c in callbacks:
        c.on_batch_end(current_step)

    train_loss /= num_train_steps
    logging.info("Done training epoch %s, epoch loss=%s.", epoch + 1,
                 train_loss)

    eval_input_iterator = iter(
        strategy.experimental_distribute_dataset(eval_input_dataset))
    hr_sum = 0
    hr_count = 0
    for _ in range(num_eval_steps):
      step_hr_sum, step_hr_count = eval_step(eval_input_iterator)
      hr_sum += step_hr_sum
      hr_count += step_hr_count

    logging.info("Done eval epoch %s, hit_rate=%s.", epoch + 1,
                 hr_sum / hr_count)
    if eval_summary_writer:
      with eval_summary_writer.as_default():
        tf.summary.scalar("hit_rate", hr_sum / hr_count, step=current_step)

    if (FLAGS.early_stopping and
        float(hr_sum / hr_count) > params["hr_threshold"]):
      break

  for c in callbacks:
    c.on_train_end()

  # Saving the model at the end of training.
  if not FLAGS.ml_perf:
    checkpoint = tf.train.Checkpoint(model=keras_model, optimizer=optimizer)
    checkpoint_path = os.path.join(FLAGS.model_dir, "ctl_checkpoint")
    checkpoint.save(checkpoint_path)
    logging.info("Saving model as TF checkpoint: %s", checkpoint_path)

  return train_loss, [None, hr_sum / hr_count]


def build_stats(loss, eval_result, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    loss: The final loss at training time.
    eval_result: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback likely used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if loss:
    stats["loss"] = loss

  if eval_result:
    stats["eval_loss"] = eval_result[0]
    stats["eval_hit_rate"] = eval_result[1]

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats["step_timestamp_log"] = timestamp_log
    stats["train_finish_time"] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats["avg_exp_per_second"] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log)-1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats


def main(_):
  with logger.benchmark_context(FLAGS), \
      mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    run_ncf(FLAGS)


if __name__ == "__main__":
  ncf_common.define_ncf_flags()
  app.run(main)
