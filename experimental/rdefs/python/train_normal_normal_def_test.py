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
"""Tests for train_normal_normal_def."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

import train_normal_normal_def

FLAGS = tf.flags.FLAGS


class TrainGaussianDefTest(tf.test.TestCase):

  def testBernoulliDegenerateSolution(self):
    """Test whether we recover bernoulli parameter 0 if we feed in zeros.

    In this case, the model is a gaussian-bernoulli factor model, with one
    time step (i.e. no recurrence)
    """
    tf.set_random_seed(1322423)
    np.random.seed(1423234)
    hparams = tf.HParams(
        dataset='alternating',
        z_dim=1,
        timestep_dim=1,
        n_timesteps=1,
        batch_size=1,
        samples_to_save=1,
        learning_rate=0.05,
        n_examples=1,
        momentum=0.0,
        p_z_sigma=1.,
        p_w_mu_sigma=1.,
        p_w_sigma_sigma=1.,
        init_q_sigma_scale=0.1,
        fixed_p_z_sigma=True,
        fixed_q_z_sigma=False,
        fixed_q_w_mu_sigma=False,
        fixed_q_w_sigma_sigma=False,
        fixed_q_w_0_sigma=False,
        dtype='float32',
        trainer='local')

    tmp_dir = tf.test.get_temp_dir()

    vi, sess = train_normal_normal_def.run_training(
        hparams, tmp_dir, 6000, None, trainer='local')

    zero = np.array(0.)
    zero = np.reshape(zero, (1, 1, 1, 1))
    p_list = []
    for _ in range(100):
      bernoulli_p = sess.run(vi.model.p_x_zw_bernoulli_p,
                             {vi.data['x']: zero,
                              vi.data['x_indexes']: np.reshape(zero, (1,))})
      p_list.append(bernoulli_p)

    mean_p = np.mean(p_list)
    self.assertAllClose(mean_p, 0., rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
  tf.test.main()
