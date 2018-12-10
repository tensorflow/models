
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
from official.recommendation import ncf_main
from official.recommendation import data_preprocessing
from official.recommendation import model_runner
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


FLAGS = flags.FLAGS


def _logitfy(inputs, base_model):
  logits = base_model(inputs)
  zero_tensor = tf.keras.layers.Lambda(lambda x: x * 0)(logits)
  to_concatenate = [zero_tensor, logits]
  concat_layer = tf.keras.layers.Concatenate(axis=1)(to_concatenate)

  reshape_layer = tf.keras.layers.Reshape(
      target_shape=(concat_layer.shape[1].value,))(concat_layer)

  model = tf.keras.Model(inputs=inputs, outputs=reshape_layer)
  return model


def main(_):
  with logger.benchmark_context(FLAGS), \
       mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    run_ncf(FLAGS)
    mlperf_helper.stitch_ncf()


def run_ncf(_):
  """Run NCF training and eval loop."""
  if FLAGS.download_if_missing and not FLAGS.use_synthetic_data:
    movielens.download(FLAGS.dataset, FLAGS.data_dir)

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)

  num_gpus = flags_core.get_num_gpus(FLAGS)
  batch_size = distribution_utils.per_device_batch_size(
      int(FLAGS.batch_size), num_gpus)
  total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals

  eval_per_user = rconst.NUM_EVAL_NEGATIVES + 1
  eval_batch_size = int(FLAGS.eval_batch_size or
                        max([FLAGS.batch_size, eval_per_user]))
  if eval_batch_size % eval_per_user:
    eval_batch_size = eval_batch_size // eval_per_user * eval_per_user
    tf.logging.warning(
        "eval examples per user does not evenly divide eval_batch_size. "
        "Overriding to {}".format(eval_batch_size))

  if FLAGS.use_synthetic_data:
    ncf_dataset = None
    cleanup_fn = lambda: None
    num_users, num_items = data_preprocessing.DATASET_TO_NUM_USERS_AND_ITEMS[
        FLAGS.dataset]
    num_train_steps = data_preprocessing.SYNTHETIC_BATCHES_PER_EPOCH
    num_eval_steps = data_preprocessing.SYNTHETIC_BATCHES_PER_EPOCH
  else:
    ncf_dataset, cleanup_fn = data_preprocessing.instantiate_pipeline(
        dataset=FLAGS.dataset, data_dir=FLAGS.data_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_neg=FLAGS.num_neg,
        epochs_per_cycle=FLAGS.epochs_between_evals,
        num_cycles=total_training_cycle,
        match_mlperf=FLAGS.ml_perf,
        deterministic=FLAGS.seed is not None,
        use_subprocess=FLAGS.use_subprocess,
        cache_id=FLAGS.cache_id)
    num_users = ncf_dataset.num_users
    num_items = ncf_dataset.num_items
    num_train_steps = int(np.ceil(
        FLAGS.epochs_between_evals * ncf_dataset.num_train_positives *
        (1 + FLAGS.num_neg) / FLAGS.batch_size))
    num_eval_steps = int(np.ceil((1 + rconst.NUM_EVAL_NEGATIVES) *
                                 ncf_dataset.num_users / eval_batch_size))

  model_helpers.apply_clean(flags.FLAGS)

  params = {
      "use_seed": FLAGS.seed is not None,
      "hash_pipeline": FLAGS.hash_pipeline,
      "batch_size": batch_size,
      "eval_batch_size": eval_batch_size,
      "learning_rate": FLAGS.learning_rate,
      "num_users": num_users,
      "num_items": num_items,
      "mf_dim": FLAGS.num_factors,
      "model_layers": [int(layer) for layer in FLAGS.layers],
      "mf_regularization": FLAGS.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in FLAGS.mlp_regularization],
      "num_neg": FLAGS.num_neg,
      "use_tpu": FLAGS.tpu is not None,
      "tpu": FLAGS.tpu,
      "tpu_zone": FLAGS.tpu_zone,
      "tpu_gcp_project": FLAGS.tpu_gcp_project,
      "beta1": FLAGS.beta1,
      "beta2": FLAGS.beta2,
      "epsilon": FLAGS.epsilon,
      "match_mlperf": FLAGS.ml_perf,
      "use_xla_for_gpu": FLAGS.use_xla_for_gpu,
      "use_estimator": FLAGS.use_estimator,
  }

  train_input_fn, _, _= data_preprocessing.make_input_fn(
    ncf_dataset=ncf_dataset, is_training=True)

  user_input = tf.keras.layers.Input(
    shape=(1,), batch_size=FLAGS.batch_size, name="user_id", dtype=tf.int32)
  item_input = tf.keras.layers.Input(
    shape=(1,), batch_size=FLAGS.batch_size, name="item_id", dtype=tf.int32)

  base_model = neumf_model.construct_model(user_input, item_input, params)
  keras_model = _logitfy([user_input, item_input], base_model)

  keras_model.summary()

  tf.logging.info("Using Keras instead of Estimator")

  def softmax_crossentropy_with_logits(y_true, y_pred):
    """A loss function replicating tf's sparse_softmax_cross_entropy
    Args:
      y_true: True labels. Tensor of shape [batch_size,]
      y_pred: Predictions. Tensor of shape [batch_size, num_classes]
    """
    y_true = tf.cast(y_true, tf.int32)
    return tf.losses.sparse_softmax_cross_entropy(
      labels=tf.reshape(y_true, [FLAGS.batch_size,]),
      logits=tf.reshape(y_pred, [FLAGS.batch_size, 2]))

  opt = neumf_model.get_optimizer(params)
  strategy = distribution_utils.get_distribution_strategy(num_gpus=1)

  keras_model.compile(loss=softmax_crossentropy_with_logits,
                optimizer=opt,
                metrics=['accuracy'],
                distribute=None)

  total_examples = 1000210
  steps_per_epoch = total_examples // FLAGS.batch_size

  train_input_dataset = train_input_fn(params)

  keras_model.fit(train_input_dataset.repeat(),
            epochs=FLAGS.train_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[],
            verbose=0)

  print("Fit done")

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  ncf_main.define_ncf_flags()
  absl_app.run(main)
