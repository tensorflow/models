# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for multitask.task_sampler."""
import tensorflow as tf

from official.modeling.multitask import configs
from official.modeling.multitask import task_sampler as sampler


class TaskSamplerTest(tf.test.TestCase):

  def setUp(self):
    super(TaskSamplerTest, self).setUp()
    self._task_weights = {'A': 1.0, 'B': 2.0, 'C': 3.0}

  def test_uniform_sample_distribution(self):
    uniform_sampler = sampler.get_task_sampler(
        configs.TaskSamplingConfig(type='uniform'), self._task_weights)
    for step in range(5):
      cumulative_distribution = uniform_sampler.task_cumulative_distribution(
          tf.constant(step, dtype=tf.int64))
      self.assertAllClose([0.333333, 0.666666, 1.0],
                          cumulative_distribution.numpy())

  def test_proportional_sample_distribution(self):
    prop_sampler = sampler.get_task_sampler(
        configs.TaskSamplingConfig(
            type='proportional',
            proportional=configs.ProportionalSampleConfig(alpha=2.0)),
        self._task_weights)
    # CucmulativeOf(Normalize([1.0^2, 2.0^2, 3.0^2]))
    for step in range(5):
      cumulative_distribution = prop_sampler.task_cumulative_distribution(
          tf.constant(step, dtype=tf.int64))
      self.assertAllClose([0.07142857, 0.35714286, 1.0],
                          cumulative_distribution.numpy())

  def test_annealing_sample_distribution(self):
    num_epoch = 3
    step_per_epoch = 6
    annel_sampler = sampler.get_task_sampler(
        configs.TaskSamplingConfig(
            type='annealing',
            annealing=configs.AnnealingSampleConfig(
                steps_per_epoch=step_per_epoch,
                total_steps=step_per_epoch * num_epoch)), self._task_weights)

    global_step = tf.Variable(
        0, dtype=tf.int64, name='global_step', trainable=False)
    expected_cumulative_epochs = [[0.12056106, 0.4387236, 1.0],
                                  [0.16666667, 0.5, 1.0],
                                  [0.22477472, 0.5654695, 1.0]]
    for epoch in range(num_epoch):
      for _ in range(step_per_epoch):
        cumulative_distribution = annel_sampler.task_cumulative_distribution(
            tf.constant(global_step, dtype=tf.int64))
        global_step.assign_add(1)
        self.assertAllClose(expected_cumulative_epochs[epoch],
                            cumulative_distribution.numpy())


if __name__ == '__main__':
  tf.test.main()
