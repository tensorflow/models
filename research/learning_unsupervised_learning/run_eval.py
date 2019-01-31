# Copyright 2018 Google, Inc. All Rights Reserved.
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
""" Script that iteratively applies the unsupervised update rule and evaluates the

meta-objective performance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

from learning_unsupervised_learning import evaluation
from learning_unsupervised_learning import datasets
from learning_unsupervised_learning import architectures
from learning_unsupervised_learning import summary_utils
from learning_unsupervised_learning import meta_objective

import tensorflow as tf
import sonnet as snt

from tensorflow.contrib.framework.python.framework import checkpoint_utils

flags.DEFINE_string("checkpoint_dir", None, "Dir to load pretrained update rule from")
flags.DEFINE_string("train_log_dir", None, "Training log directory")

FLAGS = flags.FLAGS


def train(train_log_dir, checkpoint_dir, eval_every_n_steps=10, num_steps=3000):
  dataset_fn = datasets.mnist.TinyMnist
  w_learner_fn = architectures.more_local_weight_update.MoreLocalWeightUpdateWLearner
  theta_process_fn = architectures.more_local_weight_update.MoreLocalWeightUpdateProcess

  meta_objectives = []
  meta_objectives.append(
      meta_objective.linear_regression.LinearRegressionMetaObjective)
  meta_objectives.append(meta_objective.sklearn.LogisticRegression)

  checkpoint_vars, train_one_step_op, (
      base_model, dataset) = evaluation.construct_evaluation_graph(
          theta_process_fn=theta_process_fn,
          w_learner_fn=w_learner_fn,
          dataset_fn=dataset_fn,
          meta_objectives=meta_objectives)
  batch = dataset()
  pre_logit, outputs = base_model(batch)

  global_step = tf.train.get_or_create_global_step()
  var_list = list(
      snt.get_variables_in_module(base_model, tf.GraphKeys.TRAINABLE_VARIABLES))

  tf.logging.info("all vars")
  for v in tf.all_variables():
    tf.logging.info("   %s" % str(v))
  global_step = tf.train.get_global_step()
  accumulate_global_step = global_step.assign_add(1)
  reset_global_step = global_step.assign(0)

  train_op = tf.group(
      train_one_step_op, accumulate_global_step, name="train_op")

  summary_op = tf.summary.merge_all()

  file_writer = summary_utils.LoggingFileWriter(train_log_dir, regexes=[".*"])
  if checkpoint_dir:
    str_var_list = checkpoint_utils.list_variables(checkpoint_dir)
    name_to_v_map = {v.op.name: v for v in tf.all_variables()}
    var_list = [
        name_to_v_map[vn] for vn, _ in str_var_list if vn in name_to_v_map
    ]
    saver = tf.train.Saver(var_list)
    missed_variables = [
        v.op.name for v in set(
            snt.get_variables_in_scope("LocalWeightUpdateProcess",
                                       tf.GraphKeys.GLOBAL_VARIABLES)) -
        set(var_list)
    ]
    assert len(missed_variables) == 0, "Missed a theta variable."

  hooks = []

  with tf.train.SingularMonitoredSession(master="", hooks=hooks) as sess:

    # global step should be restored from the evals job checkpoint or zero for fresh.
    step = sess.run(global_step)

    if step == 0 and checkpoint_dir:
      tf.logging.info("force restore")
      saver.restore(sess, checkpoint_dir)
      tf.logging.info("force restore done")
      sess.run(reset_global_step)
      step = sess.run(global_step)

    while step < num_steps:
      if step % eval_every_n_steps == 0:
        s, _, step = sess.run([summary_op, train_op, global_step])
        file_writer.add_summary(s, step)
      else:
        _, step = sess.run([train_op, global_step])


def main(argv):
  train(FLAGS.train_log_dir, FLAGS.checkpoint_dir)


if __name__ == "__main__":
  app.run(main)
