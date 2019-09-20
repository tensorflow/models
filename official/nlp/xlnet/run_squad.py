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
"""XLNet SQUAD finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import functools
import json
import os
import pickle

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
from official.nlp.xlnet import squad_utils
from official.nlp.xlnet import training_utils
from official.utils.misc import tpu_lib

flags.DEFINE_string(
    "test_feature_path", default=None, help="Path to feature of test set.")
flags.DEFINE_integer("query_len", default=64, help="Max query length.")
flags.DEFINE_integer("start_n_top", default=5, help="Beam size for span start.")
flags.DEFINE_integer("end_n_top", default=5, help="Beam size for span end.")
flags.DEFINE_string(
    "predict_dir", default=None, help="Path to write predictions.")
flags.DEFINE_string(
    "predict_file", default=None, help="Path to json file of test set.")
flags.DEFINE_integer(
    "n_best_size", default=5, help="n best size for predictions.")
flags.DEFINE_integer("max_answer_length", default=64, help="Max answer length.")

FLAGS = flags.FLAGS


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               input_ids,
               input_mask,
               p_mask,
               segment_ids,
               paragraph_len,
               cls_index,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index
    self.tok_end_to_orig_index = tok_end_to_orig_index
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.p_mask = p_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.cls_index = cls_index
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def get_primary_cpu_task(use_remote_tpu=False):
  """Returns primary CPU task to which input pipeline Ops are put."""

  # Remote Eager Borg job configures the TPU worker with job name 'worker'.
  return "/job:worker" if use_remote_tpu else ""


# pylint: disable=unused-argument
def run_evaluation(strategy,
                   test_input_fn,
                   eval_steps,
                   input_meta_data,
                   model,
                   step,
                   eval_summary_writer=None):
  """Run evaluation for SQUAD task.

  Args:
    strategy: distribution strategy.
    test_input_fn: input function for evaluation data.
    eval_steps: total number of evaluation steps.
    input_meta_data: input meta data.
    model: keras model object.
    step: current training step.
    eval_summary_writer: summary writer used to record evaluation metrics.

  """

  def _test_step_fn(inputs):
    """Replicated validation step."""

    inputs["mems"] = None
    res = model(inputs, training=False)
    return res, inputs["unique_ids"]

  @tf.function
  def _run_evaluation(test_iterator):
    """Runs validation steps."""
    res, unique_ids = strategy.experimental_run_v2(
        _test_step_fn, args=(next(test_iterator),))
    return res, unique_ids

  # pylint: disable=protected-access
  test_iterator = data_utils._get_input_iterator(test_input_fn, strategy)
  # pylint: enable=protected-access
  cur_results = []
  eval_examples = squad_utils.read_squad_examples(
      input_meta_data["predict_file"], is_training=False)
  with tf.io.gfile.GFile(input_meta_data["predict_file"]) as f:
    orig_data = json.load(f)["data"]

  for _ in range(eval_steps):
    results, unique_ids = _run_evaluation(test_iterator)
    unique_ids = strategy.experimental_local_results(unique_ids)

    for result_key in results:
      results[result_key] = (
          strategy.experimental_local_results(results[result_key]))
    for core_i in range(strategy.num_replicas_in_sync):
      bsz = int(input_meta_data["test_batch_size"] /
                strategy.num_replicas_in_sync)
      for j in range(bsz):
        result = {}
        for result_key in results:
          result[result_key] = results[result_key][core_i].numpy()[j]
        result["unique_ids"] = unique_ids[core_i].numpy()[j]
        # We appended a fake example into dev set to make data size can be
        # divided by test_batch_size. Ignores this fake example during
        # evaluation.
        if result["unique_ids"] == 1000012047:
          continue
        unique_id = int(result["unique_ids"])

        start_top_log_probs = ([
            float(x) for x in result["start_top_log_probs"].flat
        ])
        start_top_index = [int(x) for x in result["start_top_index"].flat]
        end_top_log_probs = ([
            float(x) for x in result["end_top_log_probs"].flat
        ])
        end_top_index = [int(x) for x in result["end_top_index"].flat]

        cls_logits = float(result["cls_logits"].flat[0])
        cur_results.append(
            squad_utils.RawResult(
                unique_id=unique_id,
                start_top_log_probs=start_top_log_probs,
                start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs,
                end_top_index=end_top_index,
                cls_logits=cls_logits))
        if len(cur_results) % 1000 == 0:
          logging.info("Processing example: %d", len(cur_results))

  output_prediction_file = os.path.join(input_meta_data["predict_dir"],
                                        "predictions.json")
  output_nbest_file = os.path.join(input_meta_data["predict_dir"],
                                   "nbest_predictions.json")
  output_null_log_odds_file = os.path.join(input_meta_data["predict_dir"],
                                           "null_odds.json")

  ret = squad_utils.write_predictions(
      eval_examples, input_meta_data["eval_features"], cur_results,
      input_meta_data["n_best_size"], input_meta_data["max_answer_length"],
      output_prediction_file, output_nbest_file, output_null_log_odds_file,
      orig_data, input_meta_data["start_n_top"], input_meta_data["end_n_top"])

  # Log current result

  log_str = "Result | "
  for key, val in ret.items():
    log_str += "{} {} | ".format(key, val)
  logging.info(log_str)
  if eval_summary_writer:
    with eval_summary_writer.as_default():
      tf.summary.scalar("best_f1", ret["best_f1"], step=step)
      tf.summary.scalar("best_exact", ret["best_exact"], step=step)
      eval_summary_writer.flush()


def get_qaxlnet_model(model_config, run_config, start_n_top, end_n_top):
  model = modeling.QAXLNetModel(
      model_config,
      run_config,
      start_n_top=start_n_top,
      end_n_top=end_n_top,
      name="model")
  return model


def main(unused_argv):
  del unused_argv
  use_remote_tpu = False
  if FLAGS.strategy_type == "mirror":
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == "tpu":
    # Initialize TPU System.
    cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    use_remote_tpu = True
  else:
    raise ValueError("The distribution strategy type is not supported: %s" %
                     FLAGS.strategy_type)
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)
  train_input_fn = functools.partial(data_utils.get_squad_input_data,
                                     FLAGS.train_batch_size, FLAGS.seq_len,
                                     FLAGS.query_len, strategy, True,
                                     FLAGS.train_tfrecord_path)

  test_input_fn = functools.partial(data_utils.get_squad_input_data,
                                    FLAGS.test_batch_size, FLAGS.seq_len,
                                    FLAGS.query_len, strategy, False,
                                    FLAGS.test_tfrecord_path)

  total_training_steps = FLAGS.train_steps
  steps_per_epoch = int(FLAGS.train_data_size / FLAGS.train_batch_size)
  steps_per_loop = FLAGS.iterations
  eval_steps = int(FLAGS.test_data_size / FLAGS.test_batch_size)

  optimizer, learning_rate_fn = optimization.create_optimizer(
      FLAGS.learning_rate,
      total_training_steps,
      FLAGS.warmup_steps,
      adam_epsilon=FLAGS.adam_epsilon)
  model_config = xlnet_config.XLNetConfig(FLAGS)
  run_config = xlnet_config.create_run_config(True, False, FLAGS)
  input_meta_data = {}
  input_meta_data["start_n_top"] = FLAGS.start_n_top
  input_meta_data["end_n_top"] = FLAGS.end_n_top
  input_meta_data["lr_layer_decay_rate"] = FLAGS.lr_layer_decay_rate
  input_meta_data["predict_dir"] = FLAGS.predict_dir
  input_meta_data["predict_file"] = FLAGS.predict_file
  input_meta_data["n_best_size"] = FLAGS.n_best_size
  input_meta_data["max_answer_length"] = FLAGS.max_answer_length
  input_meta_data["test_feature_path"] = FLAGS.test_feature_path
  input_meta_data["test_batch_size"] = FLAGS.test_batch_size
  input_meta_data["batch_size_per_core"] = int(FLAGS.train_batch_size /
                                               strategy.num_replicas_in_sync)
  input_meta_data["mem_len"] = FLAGS.mem_len
  model_fn = functools.partial(get_qaxlnet_model, model_config, run_config,
                               FLAGS.start_n_top, FLAGS.end_n_top)

  logging.info("start reading pickle file...")
  with tf.io.gfile.GFile(input_meta_data["test_feature_path"], "rb") as f:
    eval_features = pickle.load(f)

  logging.info("finishing reading pickle file...")
  input_meta_data["eval_features"] = eval_features
  eval_fn = functools.partial(run_evaluation, strategy, test_input_fn,
                              eval_steps, input_meta_data)

  with tf.device(get_primary_cpu_task(use_remote_tpu)):
    training_utils.train(
        strategy=strategy,
        model_fn=model_fn,
        input_meta_data=input_meta_data,
        eval_fn=eval_fn,
        metric_fn=None,
        train_input_fn=train_input_fn,
        test_input_fn=test_input_fn,
        init_checkpoint=FLAGS.init_checkpoint,
        total_training_steps=total_training_steps,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        optimizer=optimizer,
        learning_rate_fn=learning_rate_fn,
        model_dir=FLAGS.model_dir)


if __name__ == "__main__":
  assert tf.version.VERSION.startswith('2.')
  app.run(main)
