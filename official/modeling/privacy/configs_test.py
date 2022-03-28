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

"""Tests for configs."""

import tensorflow as tf
from official.modeling.privacy import configs


class ConfigsTest(tf.test.TestCase):

  def test_clipping_norm_default(self):
    clipping_norm = configs.DifferentialPrivacyConfig().clipping_norm
    self.assertEqual(100000000.0, clipping_norm)

  def test_noise_multiplier_default(self):
    noise_multiplier = configs.DifferentialPrivacyConfig().noise_multiplier
    self.assertEqual(0.0, noise_multiplier)

  def test_config(self):
    dp_config = configs.DifferentialPrivacyConfig(
        clipping_norm=1.0,
        noise_multiplier=1.0,
    )
    self.assertEqual(1.0, dp_config.clipping_norm)
    self.assertEqual(1.0, dp_config.noise_multiplier)


if __name__ == '__main__':
  tf.test.main()
