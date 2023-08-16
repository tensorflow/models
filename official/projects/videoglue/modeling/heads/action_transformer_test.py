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

"""Tests for action_transformer."""
import tensorflow as tf

from official.projects.videoglue.modeling.heads import action_transformer


class ActionTransformerTest(tf.test.TestCase):

  def test_action_transformer_head_construction(self):
    head = action_transformer.ActionTransformerHead(
        num_hidden_layers=1,
        num_hidden_channels=1024,
        use_sync_bn=False,
        num_classes=80,
        # parameters for TxDecoder
        num_tx_channels=128,
        num_tx_layers=3,
        num_tx_heads=3,
        use_positional_embedding=True)

    inputs = {
        'features': tf.ones([2, 4, 16, 16, 128]),
        'instances_position': tf.random.uniform([2, 6, 4]),
    }

    outputs = head(inputs, training=False)
    self.assertAllEqual(outputs.shape, [2, 6, 80])

  def test_action_transformer_linear_head_construction(self):
    head = action_transformer.ActionTransformerHead(
        num_hidden_layers=0,
        num_hidden_channels=1024,
        use_sync_bn=False,
        num_classes=80,
        dropout_rate=0.5,
        # parameters for TxDecoder
        num_tx_channels=128,
        num_tx_layers=0,
        num_tx_heads=3,
        attention_dropout_rate=0.2,
        use_positional_embedding=False)

    inputs = {
        'features': tf.ones([2, 4, 16, 16, 128]),
        'instances_position': tf.random.uniform([2, 6, 4]),
    }

    outputs = head(inputs, training=False)
    self.assertAllEqual(outputs.shape, [2, 6, 80])
    trainable_weight_names = [w.name for w in head.weights]
    expected_weight_names = [
        'action_transformer_head/mlp/dense/kernel:0',
        'action_transformer_head/mlp/dense/bias:0'
    ]
    self.assertCountEqual(trainable_weight_names, expected_weight_names)


if __name__ == '__main__':
  tf.test.main()
