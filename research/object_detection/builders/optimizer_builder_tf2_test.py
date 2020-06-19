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

"""Tests for optimizer_builder."""
import unittest
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import optimizer_builder
from object_detection.protos import optimizer_pb2
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class OptimizerBuilderV2Test(tf.test.TestCase):
  """Test building optimizers in V2 mode."""

  def testBuildRMSPropOptimizer(self):
    optimizer_text_proto = """
      rms_prop_optimizer: {
        learning_rate: {
          exponential_decay_learning_rate {
            initial_learning_rate: 0.004
            decay_steps: 800720
            decay_factor: 0.95
          }
        }
        momentum_optimizer_value: 0.9
        decay: 0.9
        epsilon: 1.0
      }
      use_moving_average: false
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    optimizer, _ = optimizer_builder.build(optimizer_proto)
    self.assertIsInstance(optimizer, tf.keras.optimizers.RMSprop)

  def testBuildMomentumOptimizer(self):
    optimizer_text_proto = """
      momentum_optimizer: {
        learning_rate: {
          constant_learning_rate {
            learning_rate: 0.001
          }
        }
        momentum_optimizer_value: 0.99
      }
      use_moving_average: false
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    optimizer, _ = optimizer_builder.build(optimizer_proto)
    self.assertIsInstance(optimizer, tf.keras.optimizers.SGD)

  def testBuildAdamOptimizer(self):
    optimizer_text_proto = """
      adam_optimizer: {
        learning_rate: {
          constant_learning_rate {
            learning_rate: 0.002
          }
        }
      }
      use_moving_average: false
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    optimizer, _ = optimizer_builder.build(optimizer_proto)
    self.assertIsInstance(optimizer, tf.keras.optimizers.Adam)

  def testMovingAverageOptimizerUnsupported(self):
    optimizer_text_proto = """
      adam_optimizer: {
        learning_rate: {
          constant_learning_rate {
            learning_rate: 0.002
          }
        }
      }
      use_moving_average: True
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    with self.assertRaises(ValueError):
      optimizer_builder.build(optimizer_proto)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
