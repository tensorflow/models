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

"""A trainable optimizer that learns a learning rate schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from learned_optimizer.optimizer import trainable_optimizer


class LearningRateSchedule(trainable_optimizer.TrainableOptimizer):
  """Learns a learning rate schedule over a fixed number of iterations."""

  def __init__(self, initial_rate=0.0, n_steps=1000, **kwargs):
    """Initializes the learning rates."""
    self.max_index = tf.constant(n_steps-1, dtype=tf.int32)

    with tf.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
      initializer = tf.constant_initializer(initial_rate)
      self.learning_rates = tf.get_variable("learning_rates",
                                            shape=([n_steps,]),
                                            initializer=initializer)

    super(LearningRateSchedule, self).__init__("LRS", ["itr"], **kwargs)

  def _initialize_state(self, var):
    """Return a dictionary mapping names of state variables to their values."""
    return {
        "itr": tf.constant(0, dtype=tf.int32),
    }

  def _compute_update(self, param, grad, state):
    """Compute updates of parameters."""

    # get the learning rate at the current index, if the index
    # is greater than the number of available learning rates,
    # use the last one
    index = tf.minimum(state["itr"], self.max_index)
    learning_rate = tf.gather(self.learning_rates, index)

    # update the parameters: parameter - learning_rate * gradient
    updated_param = param - tf.scalar_mul(learning_rate, grad)

    return updated_param, {"itr": state["itr"] + 1}
