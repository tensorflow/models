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

"""Tests for multitask_model."""

import os.path

import tensorflow as tf

from official.projects.simclr.configs import multitask_config
from official.projects.simclr.modeling import multitask_model
from official.projects.simclr.modeling import simclr_model


class MultitaskModelTest(tf.test.TestCase):

  def test_initialize_model_success(self):
    ckpt_dir = self.get_temp_dir()
    config = multitask_config.SimCLRMTModelConfig(
        input_size=[64, 64, 3],
        heads=(multitask_config.SimCLRMTHeadConfig(
            mode=simclr_model.PRETRAIN, task_name='pretrain_simclr'),
               multitask_config.SimCLRMTHeadConfig(
                   mode=simclr_model.FINETUNE, task_name='finetune_simclr')))
    model = multitask_model.SimCLRMTModel(config)
    self.assertIn('pretrain_simclr', model.sub_tasks)
    self.assertIn('finetune_simclr', model.sub_tasks)
    ckpt = tf.train.Checkpoint(backbone=model._backbone)
    ckpt.save(os.path.join(ckpt_dir, 'ckpt'))
    model.initialize()


if __name__ == '__main__':
  tf.test.main()
