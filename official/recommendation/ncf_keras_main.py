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

import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import ncf_common
from official.recommendation import neumf_model
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers


FLAGS = flags.FLAGS


def _keras_loss(y_true, y_pred):
  # Here we are using the exact same loss used by the estimator
  loss = tf.keras.losses.sparse_categorical_crossentropy(
      y_pred=y_pred,
      y_true=tf.cast(y_true, tf.int32),
      from_logits=True)
  return loss


def _get_metric_fn(params):
  """Get the metrix fn used by model compile."""
  batch_size = params["batch_size"]

  def metric_fn(y_true, y_pred):
    """Returns the in_top_k metric."""
    softmax_logits = y_pred[0, :]
    logits = tf.slice(softmax_logits, [0, 1], [batch_size, 1])

    # The dup mask should be obtained from input data, but we did not yet find
    # a good way of getting it with keras, so we set it to zeros to neglect the
    # repetition correction
    dup_mask = tf.zeros([batch_size, 1])

    _, _, in_top_k, _, _ = (
        neumf_model.compute_eval_loss_and_metrics_helper(
            logits,
            softmax_logits,
            dup_mask,
            params["num_neg"],
            params["match_mlperf"],
            params["use_xla_for_gpu"]))

    is_training = tf.keras.backend.learning_phase()
    if isinstance(is_training, int):
      is_training = tf.constant(bool(is_training), dtype=tf.bool)

    in_top_k = tf.cond(
        is_training,
        lambda: tf.zeros(shape=in_top_k.shape, dtype=in_top_k.dtype),
        lambda: in_top_k)

    return in_top_k

  return metric_fn


def _get_train_and_eval_data(producer, params):
  """Returns the datasets for training and evalutating."""

  def preprocess_train_input(features, labels):
    """Pre-process the training data.

    This is needed because:
    - Distributed training with keras fit does not support extra inputs. The
      current implementation for fit does not use the VALID_POINT_MASK in the
      input, which makes it extra, so it needs to be removed when using keras
      fit.
    - The label needs to be extended to be used in the loss fn
    """
    if not params["keras_use_ctl"]:
      features.pop(rconst.VALID_POINT_MASK)
    labels = tf.expand_dims(labels, -1)
    return features, labels

  train_input_fn = producer.make_input_fn(is_training=True)
  train_input_dataset = train_input_fn(params).map(
      preprocess_train_input)
  if not params["keras_use_ctl"]:
    train_input_dataset = train_input_dataset.repeat(FLAGS.train_epochs)

  def preprocess_eval_input(features):
    """Pre-process the eval data.

    This is needed because:
    - Distributed training with keras fit does not support extra inputs. The
      current implementation for fit does not use the DUPLICATE_MASK in the
      input, which makes it extra, so it needs to be removed when using keras
      fit.
    - The label needs to be extended to be used in the loss fn
    """
    if not params["keras_use_ctl"]:
      features.pop(rconst.DUPLICATE_MASK)
    labels = tf.zeros_like(features[movielens.USER_COLUMN])
    labels = tf.expand_dims(labels, -1)
    return features, labels

  eval_input_fn = producer.make_input_fn(is_training=False)
  eval_input_dataset = eval_input_fn(params).map(
      lambda features: preprocess_eval_input(features))

  return train_input_dataset, eval_input_dataset


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

  # The input layers are of shape (1, batch_size), to match the size of the
  # input data. The first dimension is needed because the input data are
  # required to be batched to use distribution strategies, and in this case, it
  # is designed to be of batch_size 1 for each replica.
  user_input = tf.keras.layers.Input(
      shape=(batch_size,),
      batch_size=params["batches_per_step"],
      name=movielens.USER_COLUMN,
      dtype=tf.int32)

  item_input = tf.keras.layers.Input(
      shape=(batch_size,),
      batch_size=params["batches_per_step"],
      name=movielens.ITEM_COLUMN,
      dtype=tf.int32)

  base_model = neumf_model.construct_model(
      user_input, item_input, params, need_strip=True)

  base_model_output = base_model.output

  logits = tf.keras.layers.Lambda(
      lambda x: tf.expand_dims(x, 0),
      name="logits")(base_model_output)

  zeros = tf.keras.layers.Lambda(
      lambda x: x * 0)(logits)

  softmax_logits = tf.keras.layers.concatenate(
      [zeros, logits],
      axis=-1)

  keras_model = tf.keras.Model(
      inputs=[user_input, item_input],
      outputs=softmax_logits)

  keras_model.summary()
  return keras_model


def run_ncf(_):
  """Run NCF training and eval with Keras."""
  # TODO(seemuch): Support different train and eval batch sizes
  if FLAGS.eval_batch_size != FLAGS.batch_size:
    logging.warning(
        "The Keras implementation of NCF currently does not support batch_size "
        "!= eval_batch_size ({} vs. {}). Overriding eval_batch_size to match "
        "batch_size".format(FLAGS.eval_batch_size, FLAGS.batch_size)
        )
    FLAGS.eval_batch_size = FLAGS.batch_size

  params = ncf_common.parse_flags(FLAGS)

  if params["keras_use_ctl"] and int(tf.__version__.split(".")[0]) == 1:
    logging.error(
        "Custom training loop only works with tensorflow 2.0 and above.")
    return

  # ncf_common rounds eval_batch_size (this is needed due to a reshape during
  # eval). This carries over that rounding to batch_size as well. This is the
  # per device batch size
  params["batch_size"] = params["eval_batch_size"]
  batch_size = params["batch_size"]

  num_users, num_items, num_train_steps, num_eval_steps, producer = (
      ncf_common.get_inputs(params))

  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  batches_per_step = params["batches_per_step"]
  train_input_dataset, eval_input_dataset = _get_train_and_eval_data(producer,
                                                                     params)
  # It is required that for distributed training, the dataset must call
  # batch(). The parameter of batch() here is the number of replicas involed,
  # such that each replica evenly gets a slice of data.
  train_input_dataset = train_input_dataset.batch(batches_per_step)
  eval_input_dataset = eval_input_dataset.batch(batches_per_step)

  time_callback = keras_utils.TimeHistory(batch_size, FLAGS.log_steps)
  per_epoch_callback = IncrementEpochCallback(producer)
  callbacks = [per_epoch_callback, time_callback]

  if FLAGS.early_stopping:
    early_stopping_callback = CustomEarlyStopping(
        "val_metric_fn", desired_value=FLAGS.hr_threshold)
    callbacks.append(early_stopping_callback)

  strategy = ncf_common.get_distribution_strategy(params)
  with distribution_utils.get_strategy_scope(strategy):
    keras_model = _get_keras_model(params)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"],
        beta_1=params["beta1"],
        beta_2=params["beta2"],
        epsilon=params["epsilon"])

  if params["keras_use_ctl"]:
    loss_object = tf.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM,
        from_logits=True)
    train_input_iterator = strategy.make_dataset_iterator(train_input_dataset)
    eval_input_iterator = strategy.make_dataset_iterator(eval_input_dataset)

    @tf.function
    def train_step():
      """Called once per step to train the model."""
      def step_fn(inputs):
        """Computes loss and applied gradient per replica."""
        features, labels = inputs
        with tf.GradientTape() as tape:
          softmax_logits = keras_model([features[movielens.USER_COLUMN],
                                        features[movielens.ITEM_COLUMN]])
          loss = loss_object(labels, softmax_logits,
                             sample_weight=features[rconst.VALID_POINT_MASK])
          loss *= (1.0 / (batch_size*strategy.num_replicas_in_sync))

        grads = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads,
                                           keras_model.trainable_variables)))
        return loss

      per_replica_losses = strategy.experimental_run(step_fn,
                                                     train_input_iterator)
      mean_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      return mean_loss

    @tf.function
    def eval_step():
      """Called once per eval step to compute eval metrics."""
      def step_fn(inputs):
        """Computes eval metrics per replica."""
        features, _ = inputs
        softmax_logits = keras_model([features[movielens.USER_COLUMN],
                                      features[movielens.ITEM_COLUMN]])
        logits = tf.slice(softmax_logits, [0, 0, 1], [-1, -1, -1])
        dup_mask = features[rconst.DUPLICATE_MASK]
        in_top_k, _, metric_weights, _ = neumf_model.compute_top_k_and_ndcg(
            logits,
            dup_mask,
            params["match_mlperf"])
        metric_weights = tf.cast(metric_weights, tf.float32)
        hr_sum = tf.reduce_sum(in_top_k*metric_weights)
        hr_count = tf.reduce_sum(metric_weights)
        return hr_sum, hr_count

      per_replica_hr_sum, per_replica_hr_count = (
          strategy.experimental_run(step_fn, eval_input_iterator))
      hr_sum = strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_hr_sum, axis=None)
      hr_count = strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_hr_count, axis=None)
      return hr_sum, hr_count

    time_callback.on_train_begin()
    for epoch in range(FLAGS.train_epochs):
      per_epoch_callback.on_epoch_begin(epoch)
      train_input_iterator.initialize()
      train_loss = 0
      for step in range(num_train_steps):
        time_callback.on_batch_begin(step+epoch*num_train_steps)
        train_loss += train_step()
        time_callback.on_batch_end(step+epoch*num_train_steps)
      logging.info("Done training epoch %s, epoch loss=%s.",
                   epoch+1, train_loss/num_train_steps)
      eval_input_iterator.initialize()
      hr_sum = 0
      hr_count = 0
      for _ in range(num_eval_steps):
        step_hr_sum, step_hr_count = eval_step()
        hr_sum += step_hr_sum
        hr_count += step_hr_count
      logging.info("Done eval epoch %s, hr=%s.", epoch+1, hr_sum/hr_count)

      if (FLAGS.early_stopping and
          float(hr_sum/hr_count) > params["hr_threshold"]):
        break

    time_callback.on_train_end()
    eval_results = [None, hr_sum/hr_count]

  else:
    with distribution_utils.get_strategy_scope(strategy):

      keras_model.compile(
          loss=_keras_loss,
          metrics=[_get_metric_fn(params)],
          optimizer=optimizer,
          cloning=params["clone_model_in_keras_dist_strat"])

      history = keras_model.fit(train_input_dataset,
                                steps_per_epoch=num_train_steps,
                                epochs=FLAGS.train_epochs,
                                callbacks=callbacks,
                                validation_data=eval_input_dataset,
                                validation_steps=num_eval_steps,
                                verbose=2)

      logging.info("Training done. Start evaluating")

      eval_results = keras_model.evaluate(
          eval_input_dataset,
          steps=num_eval_steps,
          verbose=2)

      logging.info("Keras evaluation is done.")

    if history and history.history:
      train_history = history.history
      train_loss = train_history["loss"][-1]

  stats = build_stats(train_loss, eval_results, time_callback)
  return stats


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
    if FLAGS.tpu:
      raise ValueError("NCF in Keras does not support TPU for now")
    run_ncf(FLAGS)


if __name__ == "__main__":
  ncf_common.define_ncf_flags()
  absl_app.run(main)
