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

import contextlib
import heapq
import json
import logging
import math
import multiprocessing
import os
import signal
import typing

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import data_preprocessing
from official.recommendation import ncf_common
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


tf.enable_eager_execution()

FLAGS = flags.FLAGS


def main(_):
  with logger.benchmark_context(FLAGS), \
      mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    if FLAGS.tpu:
      raise ValueError("NCF in Keras does not support TPU for now")
    run_ncf(FLAGS)


def run_ncf(_):
  """Run NCF training and eval with Keras."""
  params = ncf_common.parse_flags(FLAGS)

  distribution = ncf_common.get_distribution_strategy(params)

  num_users, num_items, num_train_steps, num_eval_steps, producer = (
      ncf_common.get_inputs(params))

  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  with distribution_utils.MaybeDistributionScope(distribution):
    keras_model = _get_compiled_keras_model(params)
    optimizer = ncf_common.get_optimizer(params)
    keras_model.compile(
        loss=_keras_loss_fn,
        metrics=[_get_metrics_fn(params)],
        optimizer=optimizer)

    train_input_dataset, eval_input_dataset = _get_train_and_eval_data(
        producer, params)

    batches_per_step = params["batches_per_step"]
    if FLAGS.turn_off_distribution_strategy:
      batches_per_step = 1

    train_input_dataset = train_input_dataset.batch(batches_per_step)
    eval_input_dataset = eval_input_dataset.batch(batches_per_step)

    keras_model.fit(train_input_dataset,
        epochs=FLAGS.train_epochs,
        steps_per_epoch=num_train_steps,
        callbacks=[IncrementEpochCallback(producer)],
        verbose=2)

    tf.logging.info("Training done. Start evaluating")

    eval_results = keras_model.evaluate(
        eval_input_dataset,
        steps=num_eval_steps,
        verbose=2)

    tf.logging.info("Keras evaluation is done.")

  return eval_results


def _strip_first_and_last_dimension(x, batch_size):
  return tf.reshape(x[0, :], (batch_size,))

def _get_compiled_keras_model(params):
  batch_size = params['batch_size']

  user_input = tf.keras.layers.Input(
      shape=(), batch_size=batch_size, name=movielens.USER_COLUMN, dtype=tf.int32)
  item_input = tf.keras.layers.Input(
      shape=(), batch_size=batch_size, name=movielens.ITEM_COLUMN, dtype=tf.int32)

  base_model = neumf_model.construct_model(
      user_input, item_input, params)

  # The following two layers act as the input layer to the keras_model.
  # The reason for them is that we did a dataset.batch() for the purpose of using
  # distribution strategies in data_pipeline.py
  user_input_1 = tf.keras.layers.Input(
      shape=(batch_size, 1),
      batch_size=1,
      name=movielens.USER_COLUMN,
      dtype=tf.int32)
  item_input_1 = tf.keras.layers.Input(
      shape=(batch_size, 1),
      batch_size=1,
      name=movielens.ITEM_COLUMN,
      dtype=tf.int32)
  # valid_point_mask as input for the custom loss function
  valid_pt_mask_input = tf.keras.layers.Input(
      shape=(batch_size,),
      batch_size=1,
      name=rconst.VALID_POINT_MASK,
      dtype=tf.bool)

  user_input_reshape = tf.keras.layers.Lambda(
      lambda x: _strip_first_and_last_dimension(
          x, batch_size))(user_input_1)
  item_input_reshape = tf.keras.layers.Lambda(
      lambda x: _strip_first_and_last_dimension(
          x, batch_size))(item_input_1)
  valid_pt_mask_input_reshape = tf.keras.layers.Lambda(
      lambda x: _strip_first_and_last_dimension(
          x, batch_size))(valid_pt_mask_input)

  base_model_output = base_model([user_input_reshape, item_input_reshape])
  logits= tf.keras.layers.Lambda(
          lambda x: tf.expand_dims(x, 0),
          name="logits")(base_model_output)

  zeros = tf.keras.layers.Lambda(
      lambda x: x * 0)(logits)

  softmax_output = tf.keras.layers.concatenate(
          [zeros, logits],
          axis=-1)

  keras_model = tf.keras.Model(
      inputs=[user_input_1, item_input_1],
      outputs=softmax_output)

  keras_model.summary()
  return keras_model


def _get_metrics_fn(params):
  num_neg = params["num_neg"]
  match_mlperf = params["match_mlperf"]
  use_xla_for_gpu = params["use_xla_for_gpu"]
  batch_size = params["batch_size"]

  def metric_fn(y_true, y_pred):
    softmax_logits = y_pred[0, :]
    logits = tf.slice(softmax_logits, [0, 1], [batch_size, 1])

    # TODO(shiningsun): this mask should come from feature's
    # DUPLICATE_MASK
    dup_mask = tf.zeros([batch_size, 1])

    return _get_hit_rate_metric(
        logits,
        softmax_logits,
        dup_mask,
        num_neg,
        match_mlperf,
        use_xla_for_gpu)

  return metric_fn


def _keras_loss_fn(y_true, y_pred, sample_weights=None):
  batch_losses = tf.keras.losses.sparse_categorical_crossentropy(
      y_true,
      y_pred,
      from_logits=True)

  #TODO(shiningsun): this result should be masked by VALID_POINT_MASK
  # from features
  return batch_losses


def _get_hit_rate_metric(
    logits,
    softmax_logits,
    dup_mask,
    num_neg,
    match_mlperf,
    use_xla_for_gpu):

  cross_entropy, metric_fn, in_top_k, ndcg, metric_weights =(
      neumf_model.compute_eval_loss_and_metrics_helper(
        logits,
        softmax_logits,
        dup_mask,
        num_neg,
        match_mlperf,
        use_tpu_spec=use_xla_for_gpu))

  '''
  in_top_k = tf.cond(
      tf.keras.backend.learning_phase(),
      lambda: tf.zeros(shape=in_top_k.shape, dtype=in_top_k.dtype),
      lambda: in_top_k)
  '''

  return in_top_k


def _get_train_and_eval_data(producer, params):
  train_input_fn = producer.make_input_fn(is_training=True)
  train_input_dataset = train_input_fn(params)

  def preprocess_training_input(features, labels):
    # weights = features.pop(rconst.VALID_POINT_MASK)
    for k in features:
        tensor = features[k]
        features[k] = tf.expand_dims(tensor, -1)

    labels = tf.expand_dims(labels, -1)

    # TODO(shiningsun): the following stmt should not be needed.
    # but needed here due to b/124362769
    features.pop(rconst.VALID_POINT_MASK)

    return features, labels

  train_input_dataset = train_input_dataset.map(
      lambda features, labels : preprocess_training_input(features, labels))

  def preprocess_eval_input(features):
    for k in features:
        tensor = features[k]
        features[k] = tf.expand_dims(tensor, -1)

    # TODO(shiningsun): the following stmt should not be needed.
    # but needed here due to b/124362769
    features.pop(rconst.DUPLICATE_MASK)

    labels = tf.zeros_like(features['user_id'], dtype=tf.float32)
    return features, labels

  eval_input_fn = producer.make_input_fn(is_training=False)
  eval_input_dataset = eval_input_fn(params)
  eval_input_dataset = eval_input_dataset.map(
      lambda features : preprocess_eval_input(features))

  return train_input_dataset, eval_input_dataset

class IncrementEpochCallback(tf.keras.callbacks.Callback):
  """A callback to increase the requested epoch for the data producer.

  The reason why we need this is because we can only buffer a limited amount of data.
  So we keep a moving window to represent the buffer. This is to move the one of the
  window's boundaries for each epoch.
  """

  def __init__(self, producer):
    self._producer = producer

  def on_epoch_begin(self, epoch, logs=None):
    self._producer.increment_request_epoch()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  ncf_common.define_ncf_flags()
  absl_app.run(main)

