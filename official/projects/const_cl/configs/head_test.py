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

"""Tests for head."""

import tensorflow as tf, tf_keras
from official.projects.const_cl.configs import head as head_cfg


class HeadTest(tf.test.TestCase):

  def test_mlp_head_valid(self):
    config = head_cfg.MLP(
        num_hidden_channels=128,
        num_hidden_layers=4,
        num_output_channels=1280,
        use_sync_bn=True,
        norm_momentum=0.99,
        norm_epsilon=1e-5,
        activation='relu')
    config.validate()

  def test_instance_reconstructor_head_valid(self):
    config = head_cfg.InstanceReconstructor(
        num_output_channels=1280,
        layer_norm_epsilon=1e-12,
        activation='relu')
    config.validate()

  def test_action_transformer_head_valid(self):
    config = head_cfg.ActionTransformer(
        activation='relu',
        tx_activation='relu')
    config.validate()


if __name__ == '__main__':
  tf.test.main()
