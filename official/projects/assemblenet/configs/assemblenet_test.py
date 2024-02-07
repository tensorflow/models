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

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.assemblenet.configs import assemblenet
from official.vision.configs import video_classification as exp_cfg


class AssemblenetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('assemblenet50_kinetics600',),)
  def test_assemblenet_configs(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.VideoClassificationTask)
    self.assertIsInstance(config.task.model, assemblenet.AssembleNetModel)
    self.assertIsInstance(config.task.train_data, exp_cfg.DataConfig)
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()

  def test_configs_conversion(self):
    blocks = assemblenet.flat_lists_to_blocks(assemblenet.asn50_structure,
                                              assemblenet.asn_structure_weights)
    re_structure, re_weights = assemblenet.blocks_to_flat_lists(blocks)
    self.assertAllEqual(
        re_structure, assemblenet.asn50_structure, msg='asn50_structure')
    self.assertAllEqual(
        re_weights,
        assemblenet.asn_structure_weights,
        msg='asn_structure_weights')


if __name__ == '__main__':
  tf.test.main()
