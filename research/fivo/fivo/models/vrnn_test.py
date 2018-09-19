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

"""Tests for fivo.models.vrnn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from fivo.models import base
from fivo.test_utils import create_vrnn


class VrnnTest(tf.test.TestCase):

  def test_vrnn_normal_emission(self):
    self.run_vrnn(base.ConditionalNormalDistribution, [-4.509767, -3.242221])

  def test_vrnn_bernoulli_emission(self):
    self.run_vrnn(base.ConditionalBernoulliDistribution, [-2.63812733, -2.02216434]),

  def run_vrnn(self, generative_class, gt_log_p_x_given_z):
    """Tests the VRNN.

    All test values are 'golden values' derived by running the code and copying
    the output.

    Args:
      generative_class: The class of the generative distribution to use.
      gt_log_p_x_given_z: The ground-truth value of log p(x|z).
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      batch_size = 2
      model, inputs, targets, _ = create_vrnn(generative_class=generative_class,
                                              batch_size=batch_size,
                                              data_lengths=(1, 1),
                                              random_seed=1234)
      zero_state = model.zero_state(batch_size=batch_size, dtype=tf.float32)
      model.set_observations([inputs, targets], tf.convert_to_tensor([1, 1]))
      model_out = model.propose_and_weight(zero_state, 0)
      sess.run(tf.global_variables_initializer())
      log_alpha, state = sess.run(model_out)
      rnn_state, latent_state, rnn_out = state
      self.assertAllClose(
          rnn_state.c,
          [[-0.15014534, 0.0143046, 0.00160489, -0.12899463],
           [-0.25015137, 0.09377634, -0.05000039, -0.17123522]])
      self.assertAllClose(
          rnn_state.h,
          [[-0.06842659, 0.00760155, 0.00096106, -0.05434214],
           [-0.1109542, 0.0441804, -0.03121299, -0.07882939]]
      )
      self.assertAllClose(
          latent_state,
          [[0.025241, 0.122011, 1.066661, 0.316209, -0.25369, 0.108215,
            -1.501128, -0.440111, -0.40447, -0.156649, 1.206028],
           [0.066824, 0.519937, 0.610973, 0.977739, -0.121889, -0.223429,
            -0.32687, -0.578763, -0.56965, 0.751886, 0.681606]]
      )
      self.assertAllClose(rnn_out, [[-0.068427, 0.007602, 0.000961, -0.054342],
                                    [-0.110954, 0.04418, -0.031213, -0.078829]])
      gt_log_q_z = [-8.0895052, -6.75819111]
      gt_log_p_z = [-7.246827, -6.512877]
      gt_log_alpha = (np.array(gt_log_p_z) +
                      np.array(gt_log_p_x_given_z) -
                      np.array(gt_log_q_z))
      self.assertAllClose(log_alpha, gt_log_alpha)

  def test_vrnn_with_tilt_normal_emission(self):
    self.run_vrnn_with_tilt(base.ConditionalNormalDistribution, [-5.198263, -6.31686])

  def test_vrnn_with_tilt_bernoulli_emission(self):
    self.run_vrnn_with_tilt(base.ConditionalBernoulliDistribution, [-4.66985, -3.802245])

  def run_vrnn_with_tilt(self, generative_class, gt_log_alpha):
    """Tests the VRNN with a tilting function.

    All test values are 'golden values' derived by running the code and copying
    the output.

    Args:
      generative_class: The class of the generative distribution to use.
      gt_log_alpha: The ground-truth value of log alpha.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      batch_size = 2
      model, inputs, targets, _ = create_vrnn(generative_class=generative_class,
                                              batch_size=batch_size,
                                              data_lengths=(3, 2),
                                              random_seed=1234,
                                              use_tilt=True)
      zero_state = model.zero_state(batch_size=batch_size, dtype=tf.float32)
      model.set_observations([inputs, targets], tf.convert_to_tensor([3, 2]))
      model_out = model.propose_and_weight(zero_state, 0)
      sess.run(tf.global_variables_initializer())
      log_alpha, state = sess.run(model_out)
      rnn_state, latent_state, rnn_out = state
      self.assertAllClose(
          rnn_state.c,
          [[-0.15014534, 0.0143046, 0.00160489, -0.12899463],
           [-0.25015137, 0.09377634, -0.05000039, -0.17123522]])
      self.assertAllClose(
          rnn_state.h,
          [[-0.06842659, 0.00760155, 0.00096106, -0.05434214],
           [-0.1109542, 0.0441804, -0.03121299, -0.07882939]]
      )
      self.assertAllClose(
          latent_state,
          [[0.025241, 0.122011, 1.066661, 0.316209, -0.25369, 0.108215,
            -1.501128, -0.440111, -0.40447, -0.156649, 1.206028],
           [0.066824, 0.519937, 0.610973, 0.977739, -0.121889, -0.223429,
            -0.32687, -0.578763, -0.56965, 0.751886, 0.681606]]
      )
      self.assertAllClose(rnn_out, [[-0.068427, 0.007602, 0.000961, -0.054342],
                                    [-0.110954, 0.04418, -0.031213, -0.078829]])
      self.assertAllClose(log_alpha, gt_log_alpha)

if __name__ == "__main__":
  tf.test.main()
