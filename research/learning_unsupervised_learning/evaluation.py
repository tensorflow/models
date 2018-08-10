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


"""Evaluation job.

This sits on the side and performs evaluation on a saved model.
This is a separate process for ease of use and stability of numbers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from learning_unsupervised_learning import utils


def construct_evaluation_graph(theta_process_fn=None,
                               w_learner_fn=None,
                               dataset_fn=None,
                               meta_objectives=None,
                              ):
  """Construct the evaluation graph.
  """
  if meta_objectives is None:
    meta_objectives = []

  tf.train.create_global_step()

  local_device = ""
  remote_device = ""

  meta_opt = theta_process_fn(
      remote_device=remote_device, local_device=local_device)

  base_model = w_learner_fn(
      remote_device=remote_device, local_device=local_device)

  train_dataset = dataset_fn(device=local_device)

  # construct variables
  x, outputs = base_model(train_dataset())
  initial_state = base_model.initial_state(meta_opt, max_steps=10)
  next_state = base_model.compute_next_state(outputs, meta_opt, initial_state)
  with utils.state_barrier_context(next_state):
    train_one_step_op = meta_opt.assign_state(base_model, next_state)

  meta_objs = []
  for meta_obj_fn in meta_objectives:
    meta_obj = meta_obj_fn(local_device="", remote_device="")
    meta_objs.append(meta_obj)
    J = meta_obj(train_dataset, lambda x: base_model(x)[0])
    tf.summary.scalar(str(meta_obj.__class__.__name__)+"_J", tf.reduce_mean(J))

  # TODO(lmetz) this is kinda error prone.
  # We should share the construction of the global variables across train and
  # make sure both sets of savable variables are the same
  checkpoint_vars = meta_opt.remote_variables() + [tf.train.get_global_step()]
  for meta_obj in meta_objs:
    checkpoint_vars.extend(meta_obj.remote_variables())

  return checkpoint_vars, train_one_step_op, (base_model, train_dataset)
