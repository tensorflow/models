# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time


import numpy as np
import tensorflow as tf

from im2txt import configuration
from im2txt import show_and_tell_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 5000,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_model(sess, model, global_step, summary_writer, summary_op):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
  """
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)

  # Compute perplexity over the entire dataset.
  num_eval_batches = int(
      math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

  start_time = time.time()
  sum_losses = 0.
  sum_weights = 0.
  for i in xrange(num_eval_batches):
    cross_entropy_losses, weights = sess.run([
        model.target_cross_entropy_losses,
        model.target_cross_entropy_loss_weights
    ])
    sum_losses += np.sum(cross_entropy_losses * weights)
    sum_weights += np.sum(weights)
    if not i % 100:
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_eval_batches)
  eval_time = time.time() - start_time

  perplexity = math.exp(sum_losses / sum_weights)
  tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)

  # Log perplexity to the FileWriter.
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = perplexity
  value.tag = "Perplexity"
  summary_writer.add_summary(summary, global_step)

  # Write the Events file to the eval directory.
  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(model, saver, summary_writer, summary_op):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
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

    # Run evaluation on the latest checkpoint.
    try:
      evaluate_model(
          sess=sess,
          model=model,
          global_step=global_step,
          summary_writer=summary_writer,
          summary_op=summary_op)
    except Exception, e:  # pylint: disable=broad-except
      tf.logging.error("Evaluation failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="eval")
    model.build()

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
      run_once(model, saver, summary_writer, summary_op)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.eval_dir, "--eval_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
