# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Class for variational inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class VariationalInference(object):
  """VariationalInference class."""

  def __init__(self, model, variational, data):
    """Initializes the VariationalInference class.

    Args:
      model: the probability model. an object with a log_prob and sample method.
      variational: the variational family for the model. an object with
          log_prob and sampling methods.
      data: the observations we use to fit the model.
    """
    self.model = model
    self.variational = variational
    self.data = data

  def build_graph(self):
    """Builds the graph for variational inference."""
    q_samples = self.variational.sample
    log_p = self.model.log_prob(q_samples, self.data['x'])
    log_q = self.variational.log_prob(q_samples)
    elbo = log_p - log_q
    if elbo.get_shape().ndims > 1:
      # first dimension is samples, second is batch_size
      self.scalar_elbo = tf.reduce_mean(tf.reduce_mean(elbo, 0), 0)
    else:
      self.scalar_elbo = tf.reduce_sum(elbo, 0)
    self.elbo = elbo
    self.log_p = log_p
    self.log_q = log_q
