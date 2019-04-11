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

"""
Tracks and saves training progress (models and other data such as the current
location in the lm1b corpus) for later reloading.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base import utils
from corpus_processing import unlabeled_data


class TrainingProgress(object):
  def __init__(self, config, sess, checkpoint_saver, best_model_saver,
              restore_if_possible=True):
    self.config = config
    self.checkpoint_saver = checkpoint_saver
    self.best_model_saver = best_model_saver

    tf.gfile.MakeDirs(config.checkpoints_dir)
    if restore_if_possible and tf.gfile.Exists(config.progress):
      history, current_file, current_line = utils.load_cpickle(
          config.progress, memoized=False)
      self.history = history
      self.unlabeled_data_reader = unlabeled_data.UnlabeledDataReader(
          config, current_file, current_line)
      utils.log("Continuing from global step", dict(self.history[-1])["step"],
                "(lm1b file {:}, line {:})".format(current_file, current_line))
      self.checkpoint_saver.restore(sess, tf.train.latest_checkpoint(
          self.config.checkpoints_dir))
    else:
      utils.log("No previous checkpoint found - starting from scratch")
      self.history = []
      self.unlabeled_data_reader = (
          unlabeled_data.UnlabeledDataReader(config))

  def write(self, sess, global_step):
    self.checkpoint_saver.save(sess, self.config.checkpoint,
                               global_step=global_step)
    utils.write_cpickle(
        (self.history, self.unlabeled_data_reader.current_file,
         self.unlabeled_data_reader.current_line),
        self.config.progress)

  def save_if_best_dev_model(self, sess, global_step):
    best_avg_score = 0
    for i, results in enumerate(self.history):
      if any("train" in metric for metric, value in results):
        continue
      total, count = 0, 0
      for metric, value in results:
        if "f1" in metric or "las" in metric or "accuracy" in metric:
          total += value
          count += 1
      avg_score = total / count
      if avg_score >= best_avg_score:
        best_avg_score = avg_score
        if i == len(self.history) - 1:
          utils.log("New best model! Saving...")
          self.best_model_saver.save(sess, self.config.best_model_checkpoint,
                                     global_step=global_step)
