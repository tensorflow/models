# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""XLNet pretraining runner in tf2.0."""

import functools
import os

# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: disable=unused-import
from official.common import distribute_utils
from official.nlp.xlnet import common_flags
from official.nlp.xlnet import data_utils
from official.nlp.xlnet import optimization
from official.nlp.xlnet import training_utils
from official.nlp.xlnet import xlnet_config
from official.nlp.xlnet import xlnet_modeling as modeling

flags.DEFINE_integer(
    "num_predict",
    default=None,
    help="Number of tokens to predict in partial prediction.")

# FLAGS for pretrain input preprocessing
flags.DEFINE_integer("perm_size", 0, help="Window size of permutation.")
flags.DEFINE_float("leak_ratio", default=0.1,
                   help="Percent of masked tokens that are leaked.")

flags.DEFINE_enum("sample_strategy", default="token_span",
                  enum_values=["single_token", "whole_word", "token_span",
                               "word_span"],
                  help="Stragey used to sample prediction targets.")
flags.DEFINE_integer("max_num_tokens", default=5,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_integer("min_num_tokens", default=1,
                     help="Minimum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")

flags.DEFINE_integer("max_num_words", default=5,
                     help="Maximum number of whole words to sample in a span."
                     "Effective when word_span strategy is used.")
flags.DEFINE_integer("min_num_words", default=1,
                     help="Minimum number of whole words to sample in a span."
                     "Effective when word_span strategy is used.")
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
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.strategy_type,
      tpu_address=FLAGS.tpu)
  if FLAGS.strategy_type == "tpu":
    num_hosts = strategy.extended.num_hosts
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)
    logging.info("***** Number of hosts used : %d", num_hosts)
  online_masking_config = data_utils.OnlineMaskingConfig(
      sample_strategy=FLAGS.sample_strategy,
      max_num_tokens=FLAGS.max_num_tokens,
      min_num_tokens=FLAGS.min_num_tokens,
      max_num_words=FLAGS.max_num_words,
      min_num_words=FLAGS.min_num_words)

  train_input_fn = functools.partial(
      data_utils.get_pretrain_input_data, FLAGS.train_batch_size, FLAGS.seq_len,
      strategy, FLAGS.train_tfrecord_path, FLAGS.reuse_len, FLAGS.perm_size,
      FLAGS.leak_ratio, FLAGS.num_predict, FLAGS.uncased, online_masking_config,
      num_hosts)

  total_training_steps = FLAGS.train_steps

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

  model = training_utils.train(
      strategy=strategy,
      model_fn=model_fn,
      input_meta_data=input_meta_data,
      eval_fn=None,
      metric_fn=None,
      train_input_fn=train_input_fn,
      init_checkpoint=FLAGS.init_checkpoint,
      init_from_transformerxl=FLAGS.init_from_transformerxl,
      total_training_steps=total_training_steps,
      steps_per_loop=steps_per_loop,
      optimizer=optimizer,
      learning_rate_fn=learning_rate_fn,
      model_dir=FLAGS.model_dir,
      save_steps=FLAGS.save_steps)

  # Export transformer-xl model checkpoint to be used in finetuning.
  checkpoint = tf.train.Checkpoint(transformer_xl=model.transformerxl_model)
  saved_path = checkpoint.save(
      os.path.join(FLAGS.model_dir, "pretrained/transformer_xl.ckpt"))
  logging.info("Exporting the transformer-xl model as a new TF checkpoint: %s",
               saved_path)


if __name__ == "__main__":
  app.run(main)
