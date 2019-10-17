# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for learning_rates.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v2 as tf

from official.vision.detection.modeling import learning_rates
from official.modeling.hyperparams import params_dict


class StepLearningRateWithLinearWarmupTest(tf.test.TestCase):

  def test_step_learning_rate_with_linear_warmup(self):
    params = params_dict.ParamsDict({
        'type': 'step',
        'init_learning_rate': 0.2,
        'warmup_learning_rate': 0.1,
        'warmup_steps': 100,
        'learning_rate_levels': [0.02, 0.002],
        'learning_rate_steps': [200, 400],
    })
    learning_rate_fn = learning_rates.learning_rate_generator(params)
    lr = learning_rate_fn(0).numpy()
    self.assertAlmostEqual(0.1, lr)
    lr = learning_rate_fn(50).numpy()
    self.assertAlmostEqual(0.15, lr)
    lr = learning_rate_fn(100).numpy()
    self.assertAlmostEqual(0.2, lr)
    lr = learning_rate_fn(150).numpy()
    self.assertAlmostEqual(0.2, lr)
    lr = learning_rate_fn(200).numpy()
    self.assertAlmostEqual(0.02, lr)
    lr = learning_rate_fn(300).numpy()
    self.assertAlmostEqual(0.02, lr)
    lr = learning_rate_fn(400).numpy()
    self.assertAlmostEqual(0.002, lr)
    lr = learning_rate_fn(500).numpy()
    self.assertAlmostEqual(0.002, lr)
    lr = learning_rate_fn(600).numpy()
    self.assertAlmostEqual(0.002, lr)


class CosinLearningRateWithLinearWarmupTest(tf.test.TestCase):

  def test_cosine_learning_rate_with_linear_warmup(self):
    params = params_dict.ParamsDict({
        'type': 'cosine',
        'init_learning_rate': 0.2,
        'warmup_learning_rate': 0.1,
        'warmup_steps': 100,
        'total_steps': 1100,
    })
    learning_rate_fn = learning_rates.learning_rate_generator(params)
    lr = learning_rate_fn(0).numpy()
    self.assertAlmostEqual(0.1, lr)
    lr = learning_rate_fn(50).numpy()
    self.assertAlmostEqual(0.15, lr)
    lr = learning_rate_fn(100).numpy()
    self.assertAlmostEqual(0.2, lr)
    lr = learning_rate_fn(350).numpy()
    self.assertAlmostEqual(0.17071067811865476, lr)
    lr = learning_rate_fn(600).numpy()
    self.assertAlmostEqual(0.1, lr)
    lr = learning_rate_fn(850).numpy()
    self.assertAlmostEqual(0.029289321881345254, lr)
    lr = learning_rate_fn(1100).numpy()
    self.assertAlmostEqual(0.0, lr)


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  logging.set_verbosity(logging.INFO)
  tf.test.main()
