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

import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.maxvit.configs import image_classification  # pylint:disable=unused-import
from official.vision.configs import image_classification as img_cls_config


class MaxViTImageClassificationConfigTest(tf.test.TestCase):

  def test_maxvit_build_model(self):
    config = exp_factory.get_exp_config('maxvit_imagenet')

    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(
        config.task, img_cls_config.ImageClassificationTask
    )
    self.assertIsInstance(
        config.task.model, img_cls_config.ImageClassificationModel
    )
    self.assertIsInstance(
        config.task.train_data, img_cls_config.DataConfig
    )
    config.validate()
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()


if __name__ == '__main__':
  tf.test.main()
