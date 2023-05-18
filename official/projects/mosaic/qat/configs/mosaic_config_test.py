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

"""Tests for mosaic."""
# pylint: disable=unused-import
from absl.testing import parameterized
import tensorflow as tf

from official import vision
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.mosaic.configs import mosaic_config as exp_cfg
from official.projects.mosaic.qat.configs import mosaic_config as qat_exp_cfg
from official.projects.qat.vision.configs import common


class MosaicConfigTest(tf.test.TestCase, parameterized.TestCase):

  def test_mosaic_configs(self):
    config = exp_factory.get_exp_config('mosaic_mnv35_cityscapes_qat')
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task,
                          qat_exp_cfg.MosaicSemanticSegmentationTask)
    self.assertIsInstance(config.task.model,
                          exp_cfg.MosaicSemanticSegmentationModel)
    self.assertIsInstance(config.task.quantization, common.Quantization)
    config.validate()
    config.task.train_data.is_training = None
    with self.assertRaisesRegex(KeyError, 'Found inconsistency between key'):
      config.validate()


if __name__ == '__main__':
  tf.test.main()
