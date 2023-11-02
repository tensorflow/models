# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for learning_rate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, tf_keras

from official.legacy.image_classification import learning_rate


class LearningRateTests(tf.test.TestCase):

  def test_warmup_decay(self):
    """Basic computational test for warmup decay."""
    initial_lr = 0.01
    decay_steps = 100
    decay_rate = 0.01
    warmup_steps = 10

    base_lr = tf_keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    lr = learning_rate.WarmupDecaySchedule(
        lr_schedule=base_lr, warmup_steps=warmup_steps)

    for step in range(warmup_steps - 1):
      config = lr.get_config()
      self.assertEqual(config['warmup_steps'], warmup_steps)
      self.assertAllClose(
          self.evaluate(lr(step)), step / warmup_steps * initial_lr)

  def test_cosine_decay_with_warmup(self):
    """Basic computational test for cosine decay with warmup."""
    expected_lrs = [0.0, 0.1, 0.05, 0.0]

    lr = learning_rate.CosineDecayWithWarmup(
        batch_size=256, total_steps=3, warmup_steps=1)

    for step in [0, 1, 2, 3]:
      self.assertAllClose(lr(step), expected_lrs[step])


if __name__ == '__main__':
  tf.test.main()
