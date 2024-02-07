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

"""Tests for simple."""

import numpy as np
import tensorflow as tf, tf_keras
from official.projects.const_cl.modeling.heads import simple


class SimpleTest(tf.test.TestCase):

  def test_mlp_construction(self):
    mlp_head = simple.MLP(
        num_hidden_layers=3,
        num_hidden_channels=128,
        num_output_channels=56,
        use_sync_bn=False,
        activation='relu')
    inputs = tf.zeros([2, 512])
    outputs = mlp_head(inputs, training=False)

    num_params = np.sum(
        [np.prod(v.get_shape()) for v in mlp_head.trainable_weights])
    self.assertEqual(num_params, 106296)
    self.assertAllEqual(outputs.shape, [2, 56])


if __name__ == '__main__':
  tf.test.main()
