# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Unit tests for DLRM config."""

from absl.testing import parameterized
import tensorflow as tf

from official.recommendation.ranking.configs import config


class ConfigTest(tf.test.TestCase, parameterized.TestCase):

  def test_configs(self):
    criteo_config = config.default_config()
    self.assertIsInstance(criteo_config, config.Config)
    self.assertIsInstance(criteo_config.task, config.Task)
    self.assertIsInstance(criteo_config.task.model, config.ModelConfig)
    self.assertIsInstance(criteo_config.task.train_data,
                          config.DataConfig)
    self.assertIsInstance(criteo_config.task.validation_data,
                          config.DataConfig)
    criteo_config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      criteo_config.validate()


if __name__ == '__main__':
  tf.test.main()
