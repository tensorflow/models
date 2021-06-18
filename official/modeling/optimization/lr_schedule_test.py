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

"""Tests for lr_schedule."""
from absl.testing import parameterized
import tensorflow as tf

from official.modeling.optimization import lr_schedule


class PowerAndLinearDecayTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='power_only',
          init_lr=1.0,
          power=-1.0,
          linear_decay_fraction=0.0,
          total_decay_steps=100,
          offset=0,
          expected=[[0, 1.0], [1, 1.0], [40, 1. / 40.], [60, 1. / 60],
                    [100, 1. / 100]]),
      dict(
          testcase_name='linear_only',
          init_lr=1.0,
          power=0.0,
          linear_decay_fraction=1.0,
          total_decay_steps=100,
          offset=0,
          expected=[[0, 1.0], [1, 0.99], [40, 0.6], [60, 0.4], [100, 0.0]]),
      dict(
          testcase_name='general',
          init_lr=1.0,
          power=-1.0,
          linear_decay_fraction=0.5,
          total_decay_steps=100,
          offset=0,
          expected=[[0, 1.0], [1, 1.0], [40, 1. / 40.],
                    [60, 1. / 60. * 0.8], [100, 0.0]]),
      dict(
          testcase_name='offset',
          init_lr=1.0,
          power=-1.0,
          linear_decay_fraction=0.5,
          total_decay_steps=100,
          offset=90,
          expected=[[0, 1.0], [90, 1.0], [91, 1.0], [130, 1. / 40.],
                    [150, 1. / 60. * 0.8], [190, 0.0], [200, 0.0]]),
  )
  def test_power_linear_lr_schedule(self, init_lr, power, linear_decay_fraction,
                                    total_decay_steps, offset, expected):
    lr = lr_schedule.PowerAndLinearDecay(
        initial_learning_rate=init_lr,
        power=power,
        linear_decay_fraction=linear_decay_fraction,
        total_decay_steps=total_decay_steps,
        offset=offset)
    for step, value in expected:
      self.assertAlmostEqual(lr(step).numpy(), value)


if __name__ == '__main__':
  tf.test.main()
