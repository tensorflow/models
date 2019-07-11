# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for object detection model library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection import model_runconfig


class ModelRunConfigTest(tf.test.TestCase):

  def test_create_runconfig(self):
    """Tests creation of a plain RunConfig"""
    model_dir = 'dir'

    config = model_runconfig.create_runconfig(model_dir)

    self.assertEqual(config.model_dir, model_dir)

  def test_create_runconfig_with_valid_overrides(self):
    """Tests creation of a RunConfig with some valid overrides"""
    overrides = dict(protocol='grpc+verbs',
                     tf_random_seed=5,
                     save_checkpoints_secs=10)
    override_string = self._to_override_string(overrides)

    config = model_runconfig.create_runconfig('dir', override_string)

    for k, v in overrides.items():
      self.assertEqual(getattr(config, k), v)

  def test_create_runconfig_with_inconsistent_overrides(self):
    """Tests that overrides do not allow to create an inconsistent RunConfig"""

    overrides = dict(save_checkpoints_steps=100,
                     save_checkpoints_secs=10)
    override_string = self._to_override_string(overrides)

    with self.assertRaises(ValueError):
      model_runconfig.create_runconfig('dir', override_string)

  def test_create_runconfig_with_inconsistent_types_overrides(self):
    """Tests that overrides with inconsistent types are not allowed"""
    overrides = dict(save_summary_steps='str')
    override_string = self._to_override_string(overrides)

    with self.assertRaises(ValueError):
      model_runconfig.create_runconfig('dir', override_string)

  def test_create_runconfig_with_empty_override(self):
    """Tests creation of a RunConfig with empty overrides"""
    model_dir = 'dir'
    config = model_runconfig.create_runconfig(model_dir, '')

    self.assertEqual(config.model_dir, model_dir)

  @staticmethod
  def _to_override_string(overrides):
    return ','.join(['{}={}'.format(k, v) for k, v in overrides.items()])


if __name__ == '__main__':
  tf.test.main()
