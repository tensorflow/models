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
    run_ncf(FLAGS)


def _logitfy(inputs, base_model):
  logits = base_model(inputs)
  zero_tensor = tf.keras.layers.Lambda(lambda x: x * 0)(logits)
  to_concatenate = [zero_tensor, logits]
  concat_layer = tf.keras.layers.Concatenate(axis=1)(to_concatenate)

  reshape_layer = tf.keras.layers.Reshape(
      target_shape=(concat_layer.shape[1].value,))(concat_layer)

  model = tf.keras.Model(inputs=inputs, outputs=reshape_layer)
  return model


def run_ncf(_):
  """Run NCF training and eval with Keras."""
  params = ncf_common.parse_flags(FLAGS)

  num_users, \
      num_items, \
      num_train_steps, \
      num_eval_steps, \
      producer = ncf_common.get_inputs(params)

  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  user_input = tf.keras.layers.Input(
      shape=(), batch_size=FLAGS.batch_size, name="user_id", dtype=tf.int32)
  item_input = tf.keras.layers.Input(
      shape=(), batch_size=FLAGS.batch_size, name="item_id", dtype=tf.int32)

  base_model = neumf_model.construct_model(user_input, item_input, params)
  keras_model = _logitfy([user_input, item_input], base_model)
  keras_model.summary()

  optimizer = neumf_model.get_optimizer(params)
  distribution = ncf_common.get_distribution_strategy(params)
  train_input_fn = producer.make_input_fn(is_training=True)
  print(">>>>>>>>>>>>>>before get train data set")
  train_input_dataset = train_input_fn(params).repeat(FLAGS.train_epochs)

  keras_model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=["accuracy"],
      distribute=None)

  keras_model.fit(train_input_dataset,
      epochs=FLAGS.train_epochs,
      steps_per_epoch=num_train_steps,
      callbacks=[],
      verbose=0)

  tf.logging.info("Training done. Start evaluating")

  """
  eval_input_fn = data_preprocessing.make_input_fn(
      producer, is_training=False, use_tpu=False)
  eval_input_dataset = eval_input_fn(params)
  eval_results = keras_model.evaluate(eval_input_dataset, stesp=num_eval_steps)
  tf.logging.info("Keras fit is done. Start evaluating")
  """


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  ncf_common.define_ncf_flags()
  absl_app.run(main)

