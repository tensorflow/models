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

"""Tests for fivo.models.srnn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fivo.models import base
from fivo.test_utils import create_srnn


class SrnnTest(tf.test.TestCase):

  def test_srnn_normal_emission(self):
    self.run_srnn(base.ConditionalNormalDistribution, [-5.947752, -1.182961])

  def test_srnn_bernoulli_emission(self):
    self.run_srnn(base.ConditionalBernoulliDistribution, [-2.566631, -2.479234])

  def run_srnn(self, generative_class, gt_log_alpha):
    """Tests the SRNN.

    All test values are 'golden values' derived by running the code and copying
    the output.

    Args:
      generative_class: The class of the generative distribution to use.
      gt_log_alpha: The ground-truth value of log alpha.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      batch_size = 2
      model, inputs, targets, _ = create_srnn(generative_class=generative_class,
                                              batch_size=batch_size,
                                              data_lengths=(1, 1),
                                              random_seed=1234)
      zero_state = model.zero_state(batch_size=batch_size, dtype=tf.float32)
      model.set_observations([inputs, targets], tf.convert_to_tensor([1, 1]))
      model_out = model.propose_and_weight(zero_state, 0)
      sess.run(tf.global_variables_initializer())
      log_alpha, state = sess.run(model_out)
      self.assertAllClose(
          state.latent_encoded,
          [[0.591787, 1.310583], [-1.523136, 0.953918]])
      self.assertAllClose(state.rnn_out,
                          [[0.041675, -0.056038, -0.001823, 0.005224],
                           [0.042925, -0.044619, 0.021401, 0.016998]])
      self.assertAllClose(log_alpha, gt_log_alpha)

  def test_srnn_with_tilt_normal_emission(self):
    self.run_srnn_with_tilt(base.ConditionalNormalDistribution, [-9.13577, -4.56725])


  def test_srnn_with_tilt_bernoulli_emission(self):
    self.run_srnn_with_tilt(base.ConditionalBernoulliDistribution, [-4.617461, -5.079248])

  def run_srnn_with_tilt(self, generative_class, gt_log_alpha):
    """Tests the SRNN with a tilting function.

    All test values are 'golden values' derived by running the code and copying
    the output.

    Args:
      generative_class: The class of the generative distribution to use.
      gt_log_alpha: The ground-truth value of log alpha.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      batch_size = 2
      model, inputs, targets, _ = create_srnn(generative_class=generative_class,
                                              batch_size=batch_size,
                                              data_lengths=(3, 2),
                                              random_seed=1234,
                                              use_tilt=True)
      zero_state = model.zero_state(batch_size=batch_size, dtype=tf.float32)
      model.set_observations([inputs, targets], tf.convert_to_tensor([3, 2]))
      model_out = model.propose_and_weight(zero_state, 0)
      sess.run(tf.global_variables_initializer())
      log_alpha, state = sess.run(model_out)
      self.assertAllClose(
          state.latent_encoded,
          [[0.591787, 1.310583], [-1.523136, 0.953918]])
      self.assertAllClose(state.rnn_out,
                          [[0.041675, -0.056038, -0.001823, 0.005224],
                           [0.042925, -0.044619, 0.021401, 0.016998]])
      self.assertAllClose(log_alpha, gt_log_alpha)

if __name__ == "__main__":
  tf.test.main()
