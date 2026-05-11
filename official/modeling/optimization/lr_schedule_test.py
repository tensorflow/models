# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf, tf_keras

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


class OffsetLearningRateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(class_name=lr_schedule.PiecewiseConstantDecayWithOffset),
      dict(class_name=lr_schedule.PolynomialDecayWithOffset),
      dict(class_name=lr_schedule.ExponentialDecayWithOffset),
      dict(class_name=lr_schedule.CosineDecayWithOffset),
      dict(class_name=lr_schedule.CosineDecayRestartsWithOffset),
  )
  def test_generated_docstring(self, class_name):
    self.assertNotEmpty(class_name.__init__.__doc__)

  @parameterized.parameters(
      dict(
          class_name=lr_schedule.PiecewiseConstantDecayWithOffset,
          kwarg=dict(boundaries=[50, 80], values=[1.0, 0.5, 0.1])),
      dict(
          class_name=lr_schedule.PolynomialDecayWithOffset,
          kwarg=dict(initial_learning_rate=1.0, decay_steps=100)),
      dict(
          class_name=lr_schedule.ExponentialDecayWithOffset,
          kwarg=dict(
              initial_learning_rate=1.0, decay_steps=100, decay_rate=0.5)),
      dict(
          class_name=lr_schedule.CosineDecayWithOffset,
          kwarg=dict(initial_learning_rate=1.0, decay_steps=100)),
  )
  def test_offset(self, class_name, kwarg):
    offset = 10
    offset_lr = class_name(offset=offset, **kwarg)
    base_lr = class_name.base_lr_class(**kwarg)
    self.assertIsInstance(offset_lr, class_name)
    for step in range(10, 101, 10):
      self.assertEqual(offset_lr(step), base_lr(step - offset))

  @parameterized.named_parameters(
      dict(testcase_name='period_1', start=5, end=15, offset_shift=0),
      dict(testcase_name='period_2', start=15, end=25, offset_shift=10),
      dict(testcase_name='period_3', start=25, end=35, offset_shift=20),
  )
  def test_cosine_decay_restarts_with_offset(self, start, end, offset_shift):
    offset = 5
    m_mul = 0.5
    first_decay_steps = 10

    restarts_lr = lr_schedule.CosineDecayRestartsWithOffset(
        offset=offset,
        initial_learning_rate=1.0,
        first_decay_steps=first_decay_steps,
        t_mul=1.0,
        m_mul=m_mul,
    )

    # The equivalent cosine decay with offset.
    period_start = offset + offset_shift
    initial_learning_rate = m_mul ** (period_start // first_decay_steps)
    decay_lr = lr_schedule.CosineDecayWithOffset(
        offset=period_start,
        initial_learning_rate=initial_learning_rate,
        decay_steps=first_decay_steps,
    )

    for step in range(start, end):
      self.assertAlmostEqual(
          restarts_lr(step).numpy(), decay_lr(step).numpy(), places=5
      )


if __name__ == '__main__':
  tf.test.main()
