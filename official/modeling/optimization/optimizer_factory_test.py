# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for optimizer_factory.py."""

import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers

from official.modeling.optimization import optimizer_factory
from official.modeling.optimization.configs import optimization_config
from official.nlp import optimization as nlp_optimization


class OptimizerFactoryTest(tf.test.TestCase):

  def test_sgd_optimizer(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        }
    }
    expected_optimizer_config = {
        'name': 'SGD',
        'learning_rate': 0.1,
        'decay': 0.0,
        'momentum': 0.9,
        'nesterov': False
    }
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(lr)

    self.assertIsInstance(optimizer, tf.keras.optimizers.SGD)
    self.assertEqual(expected_optimizer_config, optimizer.get_config())

  def test_adam_optimizer(self):

    # Define adam optimizer with default values.
    params = {
        'optimizer': {
            'type': 'adam'
        }
    }
    expected_optimizer_config = tf.keras.optimizers.Adam().get_config()

    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(lr)

    self.assertIsInstance(optimizer, tf.keras.optimizers.Adam)
    self.assertEqual(expected_optimizer_config, optimizer.get_config())

  def test_adam_weight_decay_optimizer(self):
    params = {
        'optimizer': {
            'type': 'adamw'
        }
    }
    expected_optimizer_config = nlp_optimization.AdamWeightDecay().get_config()
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(lr)

    self.assertIsInstance(optimizer, nlp_optimization.AdamWeightDecay)
    self.assertEqual(expected_optimizer_config, optimizer.get_config())

  def test_lamb_optimizer(self):
    params = {
        'optimizer': {
            'type': 'lamb'
        }
    }
    expected_optimizer_config = tfa_optimizers.LAMB().get_config()
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(lr)

    self.assertIsInstance(optimizer, tfa_optimizers.LAMB)
    self.assertEqual(expected_optimizer_config, optimizer.get_config())

  def test_stepwise_lr_schedule(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'stepwise',
            'stepwise': {'boundaries': [10000, 20000],
                         'values': [0.1, 0.01, 0.001]}
        }
    }
    expected_lr_step_values = [
        [0, 0.1],
        [5000, 0.1],
        [10000, 0.1],
        [10001, 0.01],
        [20000, 0.01],
        [20001, 0.001]
    ]
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()

    for step, value in expected_lr_step_values:
      self.assertAlmostEqual(lr(step).numpy(), value)

  def test_stepwise_lr_with_warmup_schedule(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'stepwise',
            'stepwise': {'boundaries': [10000, 20000],
                         'values': [0.1, 0.01, 0.001]}
        },
        'warmup': {
            'type': 'linear',
            'linear': {'warmup_steps': 500, 'warmup_learning_rate': 0.01}
        }
    }
    expected_lr_step_values = [
        [0, 0.01],
        [250, 0.055],
        [500, 0.1],
        [5500, 0.1],
        [10000, 0.1],
        [10001, 0.01],
        [20000, 0.01],
        [20001, 0.001]
    ]
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()

    for step, value in expected_lr_step_values:
      self.assertAlmostEqual(lr(step).numpy(), value)

  def test_exponential_lr_schedule(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'exponential',
            'exponential': {
                'initial_learning_rate': 0.1,
                'decay_steps': 1000,
                'decay_rate': 0.96,
                'staircase': True
            }
        }
    }
    expected_lr_step_values = [
        [0, 0.1],
        [999, 0.1],
        [1000, 0.096],
        [1999, 0.096],
        [2000, 0.09216],
    ]
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()

    for step, value in expected_lr_step_values:
      self.assertAlmostEqual(lr(step).numpy(), value)

  def test_polynomial_lr_schedule(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 0.1,
                'decay_steps': 1000,
                'end_learning_rate': 0.001
            }
        }
    }

    expected_lr_step_values = [[0, 0.1], [500, 0.0505], [1000, 0.001]]
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()

    for step, value in expected_lr_step_values:
      self.assertAlmostEqual(lr(step).numpy(), value)

  def test_constant_lr_with_warmup_schedule(self):
    params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'warmup': {
            'type': 'linear',
            'linear': {
                'warmup_steps': 500,
                'warmup_learning_rate': 0.01
            }
        }
    }

    expected_lr_step_values = [[0, 0.01], [250, 0.055], [500, 0.1], [5000, 0.1],
                               [10000, 0.1], [20000, 0.1]]
    opt_config = optimization_config.OptimizationConfig(params)
    opt_factory = optimizer_factory.OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()

    for step, value in expected_lr_step_values:
      self.assertAlmostEqual(lr(step).numpy(), value)


if __name__ == '__main__':
  tf.test.main()
