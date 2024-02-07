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

"""Tests for dbof."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling.backbones import dbof


class DbofTest(parameterized.TestCase, tf.test.TestCase):
  """Class for testing nn_layers."""

  @parameterized.product(
      pooling_method=["average", "max", "swap"],
      use_context_gate_cluster_layer=[True, False],
      context_gate_cluster_bottleneck_size=[0, 8],
  )
  def test_dbof_backbone(
      self,
      pooling_method,
      use_context_gate_cluster_layer,
      context_gate_cluster_bottleneck_size,
  ):
    """Test for creation of a context gate layer."""

    model_cfg = yt8m_cfg.DbofModel(
        cluster_size=30,
        hidden_size=20,
        pooling_method=pooling_method,
        use_context_gate_cluster_layer=use_context_gate_cluster_layer,
        context_gate_cluster_bottleneck_size=context_gate_cluster_bottleneck_size,
    )
    backbone = dbof.Dbof(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, 32]),
        params=model_cfg,
    )

    inputs = tf.ones([2, 24, 32], dtype=tf.float32)
    outputs = backbone(inputs, num_frames=tf.constant([24, 16]))
    self.assertAllEqual(outputs.shape.as_list(), [2, 20])


if __name__ == "__main__":
  tf.test.main()
