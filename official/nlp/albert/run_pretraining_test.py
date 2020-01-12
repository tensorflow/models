# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3
"""Run a small test for ALBERT pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import modeling
import run_pretraining
import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver


flags = tf.flags

FLAGS = flags.FLAGS


def _add_float32_feature(example, feature, shape, minval, maxval):
  values = np.random.random_sample(shape)
  values = (maxval - minval) * values + minval
  example.features.feature[feature].float_list.value.extend(values)


def _add_int64_feature(example, feature, shape, minval, maxval):
  values = np.random.randint(low=minval, high=maxval+1, size=shape)
  example.features.feature[feature].int64_list.value.extend(values)


def _make_dummy_input_files(num_files, num_per_file, vocab_size):
  for i in range(num_files):
    filename = os.path.join(FLAGS.output_dir, "input%d.tfrecord" % i)
    with tf.io.TFRecordWriter(filename) as writer:
      for _ in range(num_per_file):
        example = tf.train.Example()
        _add_int64_feature(example, "input_ids", [FLAGS.max_seq_length],
                           minval=0, maxval=vocab_size-1)
        _add_int64_feature(example, "input_mask", [FLAGS.max_seq_length],
                           minval=0, maxval=1)
        _add_int64_feature(example, "segment_ids", [FLAGS.max_seq_length],
                           minval=0, maxval=0)
        _add_int64_feature(example, "next_sentence_labels", [1], minval=0,
                           maxval=1)
        _add_int64_feature(example, "token_boundary", [FLAGS.max_seq_length],
                           minval=0, maxval=vocab_size)
        _add_int64_feature(example, "masked_lm_positions",
                           [FLAGS.max_predictions_per_seq], minval=0,
                           maxval=FLAGS.max_seq_length-1)
        _add_int64_feature(example, "masked_lm_ids",
                           [FLAGS.max_predictions_per_seq], minval=0,
                           maxval=vocab_size-1)
        _add_float32_feature(example, "masked_lm_weights",
                             [FLAGS.max_predictions_per_seq], minval=0,
                             maxval=1)
        record = example.SerializeToString()
        writer.write(record)
    yield filename


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  albert_config = modeling.AlbertConfig(
      100,
      embedding_size=7,
      hidden_size=26,
      num_hidden_layers=3,
      num_hidden_groups=1,
      num_attention_heads=13,
      intermediate_size=29,
      inner_group_num=1,
      down_scale_factor=1,
      hidden_act="gelu",
      hidden_dropout_prob=0,
      attention_probs_dropout_prob=0,
      max_position_embeddings=512,
      type_vocab_size=2,
      initializer_range=0.02)

  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Create some dummy input files instead of reading from actual data.
  input_files = list(_make_dummy_input_files(2, 5, 100))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = run_pretraining.model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = run_pretraining.input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    global_step = -1
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    writer = tf.io.gfile.GFile(output_eval_file, "w")
    eval_input_fn = run_pretraining.input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)
    while global_step < FLAGS.num_train_steps:
      if estimator.latest_checkpoint() is None:
        tf.logging.info("No checkpoint found yet. Sleeping.")
        time.sleep(1)
      else:
        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
        global_step = result["global_step"]
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
