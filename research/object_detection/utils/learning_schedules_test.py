# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.utils.learning_schedules."""
import tensorflow as tf

from object_detection.utils import learning_schedules


class LearningSchedulesTest(tf.test.TestCase):

  def testExponentialDecayWithBurnin(self):
    global_step = tf.placeholder(tf.int32, [])
    learning_rate_base = 1.0
    learning_rate_decay_steps = 3
    learning_rate_decay_factor = .1
    burnin_learning_rate = .5
    burnin_steps = 2
    exp_rates = [.5, .5, 1, .1, .1, .1, .01, .01]
    learning_rate = learning_schedules.exponential_decay_with_burnin(
        global_step, learning_rate_base, learning_rate_decay_steps,
        learning_rate_decay_factor, burnin_learning_rate, burnin_steps)
    with self.test_session() as sess:
      output_rates = []
      for input_global_step in range(8):
        output_rate = sess.run(learning_rate,
                               feed_dict={global_step: input_global_step})
        output_rates.append(output_rate)
      self.assertAllClose(output_rates, exp_rates)

  def testCosineDecayWithWarmup(self):
    global_step = tf.placeholder(tf.int32, [])
    learning_rate_base = 1.0
    total_steps = 100
    warmup_learning_rate = 0.1
    warmup_steps = 9
    input_global_steps = [0, 4, 8, 9, 100]
    exp_rates = [0.1, 0.5, 0.9, 1.0, 0]
    learning_rate = learning_schedules.cosine_decay_with_warmup(
        global_step, learning_rate_base, total_steps,
        warmup_learning_rate, warmup_steps)
    with self.test_session() as sess:
      output_rates = []
      for input_global_step in input_global_steps:
        output_rate = sess.run(learning_rate,
                               feed_dict={global_step: input_global_step})
        output_rates.append(output_rate)
      self.assertAllClose(output_rates, exp_rates)

  def testManualStepping(self):
    global_step = tf.placeholder(tf.int64, [])
    boundaries = [2, 3, 7]
    rates = [1.0, 2.0, 3.0, 4.0]
    exp_rates = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    learning_rate = learning_schedules.manual_stepping(global_step, boundaries,
                                                       rates)
    with self.test_session() as sess:
      output_rates = []
      for input_global_step in range(10):
        output_rate = sess.run(learning_rate,
                               feed_dict={global_step: input_global_step})
        output_rates.append(output_rate)
      self.assertAllClose(output_rates, exp_rates)

if __name__ == '__main__':
  tf.test.main()
