# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.projects.simclr.losses import contrastive_losses


class ContrastiveLossesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(1.0, 0.5)
  def test_contrastive_loss_computation(self, temperature):
    batch_size = 2
    project_dim = 16
    projection_norm = False

    p_1_arr = np.random.rand(batch_size, project_dim)
    p_1 = tf.constant(p_1_arr, dtype=tf.float32)
    p_2_arr = np.random.rand(batch_size, project_dim)
    p_2 = tf.constant(p_2_arr, dtype=tf.float32)

    losses_obj = contrastive_losses.ContrastiveLoss(
        projection_norm=projection_norm,
        temperature=temperature)
    comp_contrastive_loss = losses_obj(
        projection1=p_1,
        projection2=p_2)

    def _exp_sim(p1, p2):
      return np.exp(np.matmul(p1, p2) / temperature)

    l11 = - np.log(
        _exp_sim(p_1_arr[0], p_2_arr[0]) /
        (_exp_sim(p_1_arr[0], p_1_arr[1])
         + _exp_sim(p_1_arr[0], p_2_arr[1])
         + _exp_sim(p_1_arr[0], p_2_arr[0]))
    ) - np.log(
        _exp_sim(p_1_arr[0], p_2_arr[0]) /
        (_exp_sim(p_2_arr[0], p_2_arr[1])
         + _exp_sim(p_2_arr[0], p_1_arr[1])
         + _exp_sim(p_1_arr[0], p_2_arr[0]))
    )

    l22 = - np.log(
        _exp_sim(p_1_arr[1], p_2_arr[1]) /
        (_exp_sim(p_1_arr[1], p_1_arr[0])
         + _exp_sim(p_1_arr[1], p_2_arr[0])
         + _exp_sim(p_1_arr[1], p_2_arr[1]))
    ) - np.log(
        _exp_sim(p_1_arr[1], p_2_arr[1]) /
        (_exp_sim(p_2_arr[1], p_2_arr[0])
         + _exp_sim(p_2_arr[1], p_1_arr[0])
         + _exp_sim(p_1_arr[1], p_2_arr[1]))
    )

    exp_contrastive_loss = (l11 + l22) / 2.0

    self.assertAlmostEqual(comp_contrastive_loss[0].numpy(),
                           exp_contrastive_loss, places=5)


if __name__ == '__main__':
  tf.test.main()
