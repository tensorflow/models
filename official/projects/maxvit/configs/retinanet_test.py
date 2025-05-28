# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for retinanet."""
# pylint: disable=unused-import

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.maxvit.configs import retinanet
from official.vision.configs import retinanet as exp_cfg


class RetinaNetConfigTest(tf.test.TestCase, parameterized.TestCase):

  def test_retinanet_configs(self):
    config = exp_factory.get_exp_config('retinanet_maxvit_coco')
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.RetinaNetTask)
    self.assertIsInstance(config.task.model, exp_cfg.RetinaNet)
    self.assertIsInstance(
        config.task.model.backbone.maxvit, retinanet.backbones.MaxViT
    )
    self.assertIsInstance(config.task.train_data, exp_cfg.DataConfig)
    config.validate()
    config.task.train_data.is_training = None
    with self.assertRaisesRegex(KeyError, 'Found inconsistency between key'):
      config.validate()

if __name__ == '__main__':
  tf.test.main()
