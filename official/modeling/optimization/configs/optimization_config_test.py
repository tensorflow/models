# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for optimization_config.py."""

import tensorflow as tf, tf_keras

from official.modeling.optimization.configs import learning_rate_config as lr_cfg
from official.modeling.optimization.configs import optimization_config
from official.modeling.optimization.configs import optimizer_config as opt_cfg


class OptimizerConfigTest(tf.test.TestCase):

  def test_no_optimizer(self):
    optimizer = optimization_config.OptimizationConfig({}).optimizer.get()
    self.assertIsNone(optimizer)

  def test_no_lr_schedule(self):
    lr = optimization_config.OptimizationConfig({}).learning_rate.get()
    self.assertIsNone(lr)

  def test_no_warmup_schedule(self):
    warmup = optimization_config.OptimizationConfig({}).warmup.get()
    self.assertIsNone(warmup)

  def test_config(self):
    opt_config = optimization_config.OptimizationConfig({
        'optimizer': {
            'type': 'sgd',
            'sgd': {}  # default config
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {}
        },
        'warmup': {
            'type': 'linear'
        }
    })
    self.assertEqual(opt_config.optimizer.get(), opt_cfg.SGDConfig())
    self.assertEqual(opt_config.learning_rate.get(),
                     lr_cfg.PolynomialLrConfig())
    self.assertEqual(opt_config.warmup.get(), lr_cfg.LinearWarmupConfig())


if __name__ == '__main__':
  tf.test.main()
