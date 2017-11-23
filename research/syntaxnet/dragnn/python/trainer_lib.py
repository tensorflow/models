# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utility functions to build DRAGNN MasterSpecs and schedule model training.

Provides functions to finish a MasterSpec, building required lexicons for it and
adding them as resources, as well as setting features sizes.
"""

import random


import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile

flags = tf.app.flags
FLAGS = flags.FLAGS


def calculate_component_accuracies(eval_res_values):
  """Transforms the DRAGNN eval_res output to float accuracies of components."""
  # The structure of eval_res_values is
  # [comp1_total, comp1_correct, comp2_total, comp2_correct, ...]
  return [
      1.0 * eval_res_values[2 * i + 1] / eval_res_values[2 * i]
      if eval_res_values[2 * i] > 0 else float('nan')
      for i in xrange(len(eval_res_values) // 2)
  ]


def write_summary(summary_writer, label, value, step):
  """Write a summary for a certain evaluation."""
  summary = Summary(value=[Summary.Value(tag=label, simple_value=float(value))])
  summary_writer.add_summary(summary, step)
  summary_writer.flush()


def annotate_dataset(sess, annotator, eval_corpus):
  """Annotate eval_corpus given a model."""
  batch_size = min(len(eval_corpus), 1024)
  processed = []
  tf.logging.info('Annotating datset: %d examples', len(eval_corpus))
  for start in range(0, len(eval_corpus), batch_size):
    end = min(start + batch_size, len(eval_corpus))
    serialized_annotations = sess.run(
        annotator['annotations'],
        feed_dict={annotator['input_batch']: eval_corpus[start:end]})
    assert len(serialized_annotations) == end - start
    processed.extend(serialized_annotations)
  tf.logging.info('Done. Produced %d annotations', len(processed))
  return processed


def get_summary_writer(tensorboard_dir):
  """Creates a directory for writing summaries and returns a writer."""
  tf.logging.info('TensorBoard directory: %s', tensorboard_dir)
  tf.logging.info('Deleting prior data if exists...')
  try:
    gfile.DeleteRecursively(tensorboard_dir)
  except errors.OpError as err:
    tf.logging.error('Directory did not exist? Error: %s', err)
  tf.logging.info('Deleted! Creating the directory again...')
  gfile.MakeDirs(tensorboard_dir)
  tf.logging.info('Created! Instatiating SummaryWriter...')
  summary_writer = tf.summary.FileWriter(tensorboard_dir)
  return summary_writer


def run_training_step(sess, trainer, train_corpus, batch_size):
  """Runs a single iteration of train_op on a randomly sampled batch."""
  batch = random.sample(train_corpus, batch_size)
  sess.run(trainer['run'], feed_dict={trainer['input_batch']: batch})


def run_training(sess, trainers, annotator, evaluator, pretrain_steps,
                 train_steps, train_corpus, eval_corpus, eval_gold,
                 batch_size, summary_writer, report_every, saver,
                 checkpoint_filename, checkpoint_stats=None):
  """Runs multi-task DRAGNN training on a single corpus.

  Arguments:
    sess: TF session to use.
    trainers: List of training ops to use.
    annotator: Annotation op.
    evaluator: Function taking two serialized corpora and returning a dict of
      scalar summaries representing evaluation metrics. The 'eval_metric'
      summary will be used for early stopping.
    pretrain_steps: List of the no. of pre-training steps for each train op.
    train_steps: List of the total no. of steps for each train op.
    train_corpus: Training corpus to use.
    eval_corpus: Holdout Corpus for early stoping.
    eval_gold: Reference of eval_corpus for computing accuracy.
      eval_corpus and eval_gold are allowed to be the same if eval_corpus
      already contains gold annotation.
      Note for segmentation eval_corpus and eval_gold are always different since
      eval_corpus contains sentences whose tokens are utf8-characters while
      eval_gold's tokens are gold words.
    batch_size: How many examples to send to the train op at a time.
    summary_writer: TF SummaryWriter to use to write summaries.
    report_every: How often to compute summaries (in steps).
    saver: TF saver op to save variables.
    checkpoint_filename: File to save checkpoints to.
    checkpoint_stats: Stats of checkpoint.
  """
  random.seed(0x31337)

  if not checkpoint_stats:
    checkpoint_stats = [0] * (len(train_steps) + 1)
  tf.logging.info('Determining the training schedule...')
  target_for_step = []
  for target_idx in xrange(len(pretrain_steps)):
    target_for_step += [target_idx] * pretrain_steps[target_idx]
  while sum(train_steps) > 0:
    step = random.randint(0, sum(train_steps) - 1)
    cumulative_steps = 0
    for target_idx in xrange(len(train_steps)):
      cumulative_steps += train_steps[target_idx]
      if step < cumulative_steps:
        break
    assert train_steps[target_idx] > 0
    train_steps[target_idx] -= 1
    target_for_step.append(target_idx)
  tf.logging.info('Training schedule defined!')

  best_eval_metric = -1.0
  tf.logging.info('Starting training...')
  actual_step = sum(checkpoint_stats[1:])
  for step, target_idx in enumerate(target_for_step):
    run_training_step(sess, trainers[target_idx], train_corpus, batch_size)
    checkpoint_stats[target_idx + 1] += 1
    if step % 100 == 0:
      tf.logging.info('training step: %d, actual: %d', step, actual_step + step)
    if step % report_every == 0:
      tf.logging.info('finished step: %d, actual: %d', step, actual_step + step)

      annotated = annotate_dataset(sess, annotator, eval_corpus)
      summaries = evaluator(eval_gold, annotated)
      for label, metric in summaries.iteritems():
        write_summary(summary_writer, label, metric, actual_step + step)
      eval_metric = summaries['eval_metric']
      tf.logging.info('Current eval metric: %.2f', eval_metric)
      if best_eval_metric < eval_metric:
        tf.logging.info('Updating best eval to %.2f, saving checkpoint.',
                        eval_metric)
        best_eval_metric = eval_metric
        saver.save(sess, checkpoint_filename)

        with gfile.GFile('%s.stats' % checkpoint_filename, 'w') as f:
          stats_str = ','.join([str(x) for x in checkpoint_stats])
          f.write(stats_str)
          tf.logging.info('Writing stats: %s', stats_str)

  tf.logging.info('Finished training!')
