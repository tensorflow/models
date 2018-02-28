# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""A trainable optimizer that learns a single global learning rate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from learned_optimizer.optimizer import trainable_optimizer


class GlobalLearningRate(trainable_optimizer.TrainableOptimizer):
  """Optimizes for a single global learning rate."""

  def __init__(self, initial_rate=1e-3, **kwargs):
    """Initializes the global learning rate."""
    with tf.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
      initializer = tf.constant_initializer(initial_rate)
      self.learning_rate = tf.get_variable("global_learning_rate", shape=(),
                                           initializer=initializer)
    super(GlobalLearningRate, self).__init__("GLR", [], **kwargs)

  def _compute_update(self, param, grad, state):
    return param - tf.scalar_mul(self.learning_rate, grad), state

