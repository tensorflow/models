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

# Lint as: python3
"""Tests for teams_experiments."""

from absl.testing import parameterized
import tensorflow as tf

# pylint: disable=unused-import
from official.common import registry_imports
# pylint: enable=unused-import
from official.core import config_definitions as cfg
from official.core import exp_factory


class TeamsExperimentsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('teams/pretraining',))
  def test_teams_experiments(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task.train_data, cfg.DataConfig)


if __name__ == '__main__':
  tf.test.main()
