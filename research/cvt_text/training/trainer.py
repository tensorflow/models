# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Runs training for CVT text models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import time
import numpy as np
import tensorflow as tf

from base import utils
from model import multitask_model
from task_specific import task_definitions


class Trainer(object):
  def __init__(self, config):
    self._config = config
    self.tasks = [task_definitions.get_task(self._config, task_name)
                  for task_name in self._config.task_names]

    utils.log('Loading Pretrained Embeddings')
    pretrained_embeddings = utils.load_cpickle(self._config.word_embeddings)

    utils.log('Building Model')
    self._model = multitask_model.Model(
        self._config, pretrained_embeddings, self.tasks)
    utils.log()

  def train(self, sess, progress, summary_writer):
    heading = lambda s: utils.heading(s, '(' + self._config.model_name + ')')
    trained_on_sentences = 0
    start_time = time.time()
    unsupervised_loss_total, unsupervised_loss_count = 0, 0
    supervised_loss_total, supervised_loss_count = 0, 0
    for mb in self._get_training_mbs(progress.unlabeled_data_reader):
      if mb.task_name != 'unlabeled':
        loss = self._model.train_labeled(sess, mb)
        supervised_loss_total += loss
        supervised_loss_count += 1

      if mb.task_name == 'unlabeled':
        self._model.run_teacher(sess, mb)
        loss = self._model.train_unlabeled(sess, mb)
        unsupervised_loss_total += loss
        unsupervised_loss_count += 1
        mb.teacher_predictions.clear()

      trained_on_sentences += mb.size
      global_step = self._model.get_global_step(sess)

      if global_step % self._config.print_every == 0:
        utils.log('step {:} - '
                  'supervised loss: {:.2f} - '
                  'unsupervised loss: {:.2f} - '
                  '{:.1f} sentences per second'.format(
            global_step,
            supervised_loss_total / max(1, supervised_loss_count),
            unsupervised_loss_total / max(1, unsupervised_loss_count),
            trained_on_sentences / (time.time() - start_time)))
        unsupervised_loss_total, unsupervised_loss_count = 0, 0
        supervised_loss_total, supervised_loss_count = 0, 0

      if global_step % self._config.eval_dev_every == 0:
        heading('EVAL ON DEV')
        self.evaluate_all_tasks(sess, summary_writer, progress.history)
        progress.save_if_best_dev_model(sess, global_step)
        utils.log()

      if global_step % self._config.eval_train_every == 0:
        heading('EVAL ON TRAIN')
        self.evaluate_all_tasks(sess, summary_writer, progress.history, True)
        utils.log()

      if global_step % self._config.save_model_every == 0:
        heading('CHECKPOINTING MODEL')
        progress.write(sess, global_step)
        utils.log()

  def evaluate_all_tasks(self, sess, summary_writer, history, train_set=False):
    for task in self.tasks:
      results = self._evaluate_task(sess, task, summary_writer, train_set)
      if history is not None:
        results.append(('step', self._model.get_global_step(sess)))
        history.append(results)
    if history is not None:
      utils.write_cpickle(history, self._config.history_file)

  def _evaluate_task(self, sess, task, summary_writer, train_set):
    scorer = task.get_scorer()
    data = task.train_set if train_set else task.val_set
    for i, mb in enumerate(data.get_minibatches(self._config.test_batch_size)):
      loss, batch_preds = self._model.test(sess, mb)
      scorer.update(mb.examples, batch_preds, loss)

    results = scorer.get_results(task.name +
                                 ('_train_' if train_set else '_dev_'))
    utils.log(task.name.upper() + ': ' + scorer.results_str())
    write_summary(summary_writer, results,
                  global_step=self._model.get_global_step(sess))
    return results

  def _get_training_mbs(self, unlabeled_data_reader):
    datasets = [task.train_set for task in self.tasks]
    weights = [np.sqrt(dataset.size) for dataset in datasets]
    thresholds = np.cumsum([w / np.sum(weights) for w in weights])

    labeled_mbs = [dataset.endless_minibatches(self._config.train_batch_size)
                   for dataset in datasets]
    unlabeled_mbs = unlabeled_data_reader.endless_minibatches()
    while True:
      dataset_ind = bisect.bisect(thresholds, np.random.random())
      yield next(labeled_mbs[dataset_ind])
      if self._config.is_semisup:
        yield next(unlabeled_mbs)


def write_summary(writer, results, global_step):
  for k, v in results:
    if 'f1' in k or 'acc' in k or 'loss' in k:
      writer.add_summary(tf.Summary(
          value=[tf.Summary.Value(tag=k, simple_value=v)]), global_step)
  writer.flush()
