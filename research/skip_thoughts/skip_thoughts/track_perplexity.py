# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tracks training progress via per-word perplexity.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import math
import os.path
import time


import numpy as np
import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", None,
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", None, "Directory to write event logs to.")

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 50000,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 100,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_model(sess, losses, weights, num_batches, global_step,
                   summary_writer, summary_op):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    losses: A Tensor of any shape; the target cross entropy losses for the
      current batch.
    weights: A Tensor of weights corresponding to losses.
    num_batches: Integer; the number of evaluation batches.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
  """
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)

  start_time = time.time()
  sum_losses = 0.0
  sum_weights = 0.0
  for i in range(num_batches):
    batch_losses, batch_weights = sess.run([losses, weights])
    sum_losses += np.sum(batch_losses * batch_weights)
    sum_weights += np.sum(batch_weights)
    if not i % 100:
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_batches)
  eval_time = time.time() - start_time

  perplexity = math.exp(sum_losses / sum_weights)
  tf.logging.info("Perplexity = %f (%.2f sec)", perplexity, eval_time)

  # Log perplexity to the SummaryWriter.
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = perplexity
  value.tag = "perplexity"
  summary_writer.add_summary(summary, global_step)

  # Write the Events file to the eval directory.
  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(model, losses, weights, saver, summary_writer, summary_op):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of SkipThoughtsModel; the model to evaluate.
    losses: Tensor; the target cross entropy losses for the current batch.
    weights: A Tensor of weights corresponding to losses.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
  """
  model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    os.path.basename(model_path), global_step)
    if global_step < FLAGS.min_global_step:
      tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                      FLAGS.min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_eval_batches = int(
        math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

    # Run evaluation on the latest checkpoint.
    try:
      evaluate_model(sess, losses, weights, num_eval_batches, global_step,
                     summary_writer, summary_op)
    except tf.InvalidArgumentError:
      tf.logging.error(
          "Evaluation raised InvalidArgumentError (e.g. due to Nans).")
    finally:
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(unused_argv):
  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.checkpoint_dir:
    raise ValueError("--checkpoint_dir is required.")
  if not FLAGS.eval_dir:
    raise ValueError("--eval_dir is required.")

  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.model_config(
        input_file_pattern=FLAGS.input_file_pattern,
        input_queue_capacity=FLAGS.num_eval_examples,
        shuffle_input_data=False)
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="eval")
    model.build()

    losses = tf.concat(model.target_cross_entropy_losses, 0)
    weights = tf.concat(model.target_cross_entropy_loss_weights, 0)

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(eval_dir)

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model, losses, weights, saver, summary_writer, summary_op)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


if __name__ == "__main__":
  tf.app.run()
