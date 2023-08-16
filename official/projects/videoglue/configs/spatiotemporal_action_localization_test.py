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

"""Tests for spatiotemporal_action_localization."""

import tensorflow as tf

from official.projects.videoglue.configs import spatiotemporal_action_localization as stal


class SpatiotemporalActionLocalizationTest(tf.test.TestCase):

  def test_spatiotemporal_action_localization_config(self):
    config = (
        stal.spatiotemporal_action_localization())

    self.assertIsInstance(
        config.task,
        stal.SpatiotemporalActionLocalizationTask)
    self.assertIsInstance(
        config.task.model,
        stal.VideoActionTransformerModel)

  def test_spatiotemporal_action_localization_vit12_config(self):
    config = (
        stal.spatiotemporal_action_localization_vit12())

    self.assertIsInstance(
        config.task,
        stal.SpatiotemporalActionLocalizationTask)
    self.assertEqual(
        config.trainer.optimizer_config.optimizer.type, 'vit_adamw')

  def test_spatiotemporal_action_localization_vit16_config(self):
    config = (
        stal.spatiotemporal_action_localization_vit16())

    self.assertIsInstance(
        config.task,
        stal.SpatiotemporalActionLocalizationTask)
    self.assertEqual(
        config.trainer.optimizer_config.optimizer.type, 'vit_adamw')


if __name__ == '__main__':
  tf.test.main()
