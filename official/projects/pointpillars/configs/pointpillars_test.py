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

"""Tests for pointpillars."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.pointpillars.configs import pointpillars as exp_cfg


class PointPillarsConfigTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('pointpillars_baseline',),
  )
  def test_configs(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.PointPillarsTask)
    self.assertIsInstance(config.task.model, exp_cfg.PointPillarsModel)
    self.assertIsInstance(config.task.train_data, exp_cfg.DataConfig)
    self.assertIsInstance(config.task.validation_data, exp_cfg.DataConfig)
    self.assertIsInstance(config.task.losses, exp_cfg.Losses)
    self.assertGreater(config.task.model.image.height, 0)
    self.assertGreater(config.task.model.image.width, 0)
    self.assertLen(config.task.model.head.attribute_heads, 3)
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()


if __name__ == '__main__':
  tf.test.main()
