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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from learning_unsupervised_learning import optimizers
from learning_unsupervised_learning import utils
from learning_unsupervised_learning import summary_utils
from learning_unsupervised_learning import variable_replace

class MultiTrialMetaObjective(snt.AbstractModule):
  def __init__(self, samples_per_class, averages, **kwargs):
    self.samples_per_class = samples_per_class
    self.averages = averages
    self.dataset_map = {}

    super(MultiTrialMetaObjective,
          self).__init__(**kwargs)

  def _build(self, dataset, feature_transformer):
    if self.samples_per_class is not None:
      if dataset not in self.dataset_map:
        # datasets are outside of frames from while loops
        with tf.control_dependencies(None):
          self.dataset_map[dataset] = utils.sample_n_per_class(
              dataset, self.samples_per_class)

      dataset = self.dataset_map[dataset]

    stats = collections.defaultdict(list)
    losses = []
    # TODO(lmetz) move this to ingraph control flow?
    for _ in xrange(self.averages):
      loss, stat = self._build_once(dataset, feature_transformer)
      losses.append(loss)
      for k, v in stat.items():
        stats[k].append(v)
    stats = {k: tf.add_n(v) / float(len(v)) for k, v in stats.items()}

    for k, v in stats.items():
      tf.summary.scalar(k, v)

    return tf.add_n(losses) / float(len(losses))

  def local_variables(self):
    """List of variables that need to be updated for each evaluation.

    These variables should not be stored on a parameter server and
    should be reset every computation of a meta_objective loss.

    Returns:
      vars: list of tf.Variable
    """
    return list(
        snt.get_variables_in_module(self, tf.GraphKeys.TRAINABLE_VARIABLES))

  def remote_variables(self):
    return []
