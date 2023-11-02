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

"""Tests for multitask_config."""

import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.modeling.multitask import configs as multitask_configs
from official.projects.simclr.configs import multitask_config as simclr_multitask_config
from official.projects.simclr.configs import simclr as exp_cfg


class MultitaskConfigTest(tf.test.TestCase):

  def test_simclr_configs(self):
    config = exp_factory.get_exp_config('multitask_simclr')
    self.assertIsInstance(config, multitask_configs.MultiTaskExperimentConfig)
    self.assertIsInstance(config.task.model,
                          simclr_multitask_config.SimCLRMTModelConfig)
    self.assertIsInstance(config.task.task_routines[0].task_config,
                          exp_cfg.SimCLRPretrainTask)
    self.assertIsInstance(config.task.task_routines[1].task_config,
                          exp_cfg.SimCLRFinetuneTask)


if __name__ == '__main__':
  tf.test.main()
