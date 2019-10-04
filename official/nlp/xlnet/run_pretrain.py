# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""XLNet classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: disable=unused-import
from official.nlp import xlnet_config
from official.nlp import xlnet_modeling as modeling
from official.nlp.xlnet import common_flags
from official.nlp.xlnet import data_utils
from official.nlp.xlnet import optimization
from official.nlp.xlnet import training_utils
from official.utils.misc import tpu_lib

flags.DEFINE_integer(
    "mask_alpha", default=6, help="How many tokens to form a group.")
flags.DEFINE_integer(
    "mask_beta", default=1, help="How many tokens to mask within each group.")
flags.DEFINE_integer(
    "num_predict",
    default=None,
    help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer("perm_size", 0, help="Window size of permutation.")

FLAGS = flags.FLAGS


def get_pretrainxlnet_model(model_config, run_config):
  return modeling.PretrainingXLNetModel(
      use_proj=True,
      xlnet_config=model_config,
      run_config=run_config,
      name="model")


def main(unused_argv):
  del unused_argv
  num_hosts = 1
  if FLAGS.strategy_type == "mirror":
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == "tpu":
    cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    topology = FLAGS.tpu_topology.split("x")
    total_num_core = 2 * int(topology[0]) * int(topology[1])
    num_hosts = total_num_core // FLAGS.num_core_per_host
  else:
    raise ValueError("The distribution strategy type is not supported: %s" %
                     FLAGS.strategy_type)
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)
    logging.info("***** Number of hosts used : %d", num_hosts)
  train_input_fn = functools.partial(
      data_utils.get_pretrain_input_data, FLAGS.train_batch_size, FLAGS.seq_len,
      strategy, FLAGS.train_tfrecord_path, FLAGS.reuse_len, FLAGS.perm_size,
      FLAGS.mask_alpha, FLAGS.mask_beta, FLAGS.num_predict, FLAGS.bi_data,
      FLAGS.uncased, num_hosts)

  total_training_steps = FLAGS.train_steps
  steps_per_epoch = int(FLAGS.train_data_size / FLAGS.train_batch_size)
  steps_per_loop = FLAGS.iterations

  optimizer, learning_rate_fn = optimization.create_optimizer(
      init_lr=FLAGS.learning_rate,
      num_train_steps=total_training_steps,
      num_warmup_steps=FLAGS.warmup_steps,
      min_lr_ratio=FLAGS.min_lr_ratio,
      adam_epsilon=FLAGS.adam_epsilon,
      weight_decay_rate=FLAGS.weight_decay_rate)

  model_config = xlnet_config.XLNetConfig(FLAGS)
  run_config = xlnet_config.create_run_config(True, False, FLAGS)
  input_meta_data = {}
  input_meta_data["d_model"] = FLAGS.d_model
  input_meta_data["mem_len"] = FLAGS.mem_len
  input_meta_data["batch_size_per_core"] = int(FLAGS.train_batch_size /
                                               strategy.num_replicas_in_sync)
  input_meta_data["n_layer"] = FLAGS.n_layer
  input_meta_data["lr_layer_decay_rate"] = FLAGS.lr_layer_decay_rate
  model_fn = functools.partial(get_pretrainxlnet_model, model_config,
                               run_config)

  training_utils.train(
      strategy=strategy,
      model_fn=model_fn,
      input_meta_data=input_meta_data,
      eval_fn=None,
      metric_fn=None,
      train_input_fn=train_input_fn,
      test_input_fn=None,
      init_checkpoint=FLAGS.init_checkpoint,
      total_training_steps=total_training_steps,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      optimizer=optimizer,
      learning_rate_fn=learning_rate_fn,
      model_dir=FLAGS.model_dir,
      save_steps=FLAGS.save_steps)


if __name__ == "__main__":
  assert tf.version.VERSION.startswith('2.')
  app.run(main)
