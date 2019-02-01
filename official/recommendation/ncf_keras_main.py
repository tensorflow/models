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

from tensorflow.contrib.compiler import xla
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
  batch_size = params['batch_size']

  distribution = ncf_common.get_distribution_strategy(params)

  num_users, \
      num_items, \
      num_train_steps, \
      num_eval_steps, \
      producer = ncf_common.get_inputs(params)

  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  # TODO(shiningsun): Both MirroredStrategy and OneDeviceStrategy error out.
  # Find out why and change the distribute to distribution
  with distribution_utils.MaybeDistributionScope(None):
    user_input = tf.keras.layers.Input(
        shape=(), batch_size=batch_size, name=movielens.USER_COLUMN, dtype=tf.int32)
    item_input = tf.keras.layers.Input(
        shape=(), batch_size=batch_size, name=movielens.ITEM_COLUMN, dtype=tf.int32)

    # Dummy duplicate mask
    dup_mask_input = tf.keras.layers.Input(
        shape=(), batch_size=batch_size, name=rconst.DUPLICATE_MASK, dtype=tf.int32)
    # Labels as input for the custom loss function
    labels_input = tf.keras.layers.Input(
        shape=(), batch_size=batch_size, name="labels", dtype=tf.int32)
    # valid_point_mask as input for the custom loss function
    valid_pt_mask_input = tf.keras.layers.Input(
        shape=(), batch_size=batch_size, name=rconst.VALID_POINT_MASK, dtype=tf.int32)

    base_model = neumf_model.construct_model(user_input, item_input, params)
    keras_model_input = base_model.input

    keras_model_input.append(dup_mask_input)
    keras_model_input.append(labels_input)
    keras_model_input.append(valid_pt_mask_input)

    keras_model = tf.keras.Model(
        inputs=keras_model_input,
        outputs=base_model.output)
    keras_model.summary()

    def pre_process_training_input(features, labels):
      # Add a dummy dup_mask to the input dataset
      features[rconst.DUPLICATE_MASK] = tf.zeros_like(
          features[movielens.USER_COLUMN], dtype=tf.float32)
      features["labels"] = labels
      return features, labels

    optimizer = ncf_common.get_optimizer(params)

    train_input_fn = producer.make_input_fn(is_training=True)
    train_input_dataset = train_input_fn(params).map(
        lambda features, labels: pre_process_training_input(features, labels))
    train_input_dataset = train_input_dataset.repeat(FLAGS.train_epochs)

    # Custom loss function to include the valid point mask
    softmax_logits = ncf_common.softmax_logitfy(keras_model.output)
    loss_tensor = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_input,
        logits=softmax_logits,
        weights=tf.cast(valid_pt_mask_input, tf.float32),
    )
    keras_model.add_loss(loss_tensor)

    # Custom loss function for the hit rate
    logits = keras_model.output
    softmax_logits = ncf_common.softmax_logitfy(logits)

    cross_entropy, \
    metric_fn, \
    in_top_k, \
    ndcg, \
    metric_weights = neumf_model.compute_eval_loss_and_metrics_helper(
        logits,
        softmax_logits,
        tf.cast(dup_mask_input, tf.float32),
        params["num_neg"],
        params["match_mlperf"],
        use_tpu_spec=params["use_xla_for_gpu"])

    hit_rate_metric = in_top_k

    hit_rate_metric = tf.cond(
        tf.keras.backend.learning_phase(),
        lambda: tf.zeros(shape=in_top_k.shape, dtype=in_top_k.dtype),
        lambda: hit_rate_metric)
    keras_model.add_metric(
        hit_rate_metric,
        name='hit_rate',
        aggregation='mean')

    keras_model.compile(optimizer=optimizer)

  keras_model.fit(train_input_dataset,
      epochs=FLAGS.train_epochs,
      steps_per_epoch=num_train_steps,
      callbacks=[IncrementEpochCallback(producer)],
      verbose=2)

  tf.logging.info("Training done. Start evaluating")

  def pre_process_eval_input(features):
    features["labels"] = tf.zeros_like(features['user_id'], dtype=tf.float32)
    features[rconst.VALID_POINT_MASK] = tf.zeros_like(features['user_id'], dtype=tf.float32)
    return features

  eval_input_fn = producer.make_input_fn(is_training=False)
  eval_input_dataset = eval_input_fn(params)
  eval_input_dataset = eval_input_dataset.map(
      lambda features : pre_process_eval_input(features))
  eval_results = keras_model.evaluate(
      eval_input_dataset,
      steps=num_eval_steps,
      verbose=2)

  tf.logging.info("Keras evaluation is done.")
  return eval_results


class IncrementEpochCallback(tf.keras.callbacks.Callback):

  def __init__(self, producer):
    self._producer = producer

  def on_epoch_begin(self, epoch, logs=None):
    self._producer.increment_request_epoch()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  ncf_common.define_ncf_flags()
  absl_app.run(main)

