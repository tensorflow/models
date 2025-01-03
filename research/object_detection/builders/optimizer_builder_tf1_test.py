# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import six
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import optimizer_builder
from object_detection.protos import optimizer_pb2
from object_detection.utils import tf_version

# pylint: disable=g-import-not-at-top
if tf_version.is_tf1():
  from tensorflow.contrib import opt as contrib_opt
# pylint: enable=g-import-not-at-top


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class LearningRateBuilderTest(tf.test.TestCase):

  def testBuildConstantLearningRate(self):
    learning_rate_text_proto = """
      constant_learning_rate {
        learning_rate: 0.004
      }
    """
    learning_rate_proto = optimizer_pb2.LearningRate()
    text_format.Merge(learning_rate_text_proto, learning_rate_proto)
    learning_rate = optimizer_builder._create_learning_rate(
        learning_rate_proto)
    self.assertTrue(
        six.ensure_str(learning_rate.op.name).endswith('learning_rate'))
    with self.test_session():
      learning_rate_out = learning_rate.eval()
    self.assertAlmostEqual(learning_rate_out, 0.004)

  def testBuildExponentialDecayLearningRate(self):
    learning_rate_text_proto = """
      exponential_decay_learning_rate {
        initial_learning_rate: 0.004
        decay_steps: 99999
        decay_factor: 0.85
        staircase: false
      }
    """
    learning_rate_proto = optimizer_pb2.LearningRate()
    text_format.Merge(learning_rate_text_proto, learning_rate_proto)
    learning_rate = optimizer_builder._create_learning_rate(
        learning_rate_proto)
    self.assertTrue(
        six.ensure_str(learning_rate.op.name).endswith('learning_rate'))
    self.assertIsInstance(learning_rate, tf.Tensor)

  def testBuildManualStepLearningRate(self):
    learning_rate_text_proto = """
      manual_step_learning_rate {
        initial_learning_rate: 0.002
        schedule {
          step: 100
          learning_rate: 0.006
        }
        schedule {
          step: 90000
          learning_rate: 0.00006
        }
        warmup: true
      }
    """
    learning_rate_proto = optimizer_pb2.LearningRate()
    text_format.Merge(learning_rate_text_proto, learning_rate_proto)
    learning_rate = optimizer_builder._create_learning_rate(
        learning_rate_proto)
    self.assertIsInstance(learning_rate, tf.Tensor)

  def testBuildCosineDecayLearningRate(self):
    learning_rate_text_proto = """
      cosine_decay_learning_rate {
        learning_rate_base: 0.002
        total_steps: 20000
        warmup_learning_rate: 0.0001
        warmup_steps: 1000
        hold_base_rate_steps: 20000
      }
    """
    learning_rate_proto = optimizer_pb2.LearningRate()
    text_format.Merge(learning_rate_text_proto, learning_rate_proto)
    learning_rate = optimizer_builder._create_learning_rate(
        learning_rate_proto)
    self.assertIsInstance(learning_rate, tf.Tensor)

  def testRaiseErrorOnEmptyLearningRate(self):
    learning_rate_text_proto = """
    """
    learning_rate_proto = optimizer_pb2.LearningRate()
    text_format.Merge(learning_rate_text_proto, learning_rate_proto)
    with self.assertRaises(ValueError):
      optimizer_builder._create_learning_rate(learning_rate_proto)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class OptimizerBuilderTest(tf.test.TestCase):

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
    self.assertIsInstance(optimizer, tf.train.RMSPropOptimizer)

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
    self.assertIsInstance(optimizer, tf.train.MomentumOptimizer)

  def testBuildAdamOptimizer(self):
    optimizer_text_proto = """
      adam_optimizer: {
        epsilon: 1e-6
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
    self.assertIsInstance(optimizer, tf.train.AdamOptimizer)

  def testBuildMovingAverageOptimizer(self):
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
    optimizer, _ = optimizer_builder.build(optimizer_proto)
    self.assertIsInstance(optimizer, contrib_opt.MovingAverageOptimizer)

  def testBuildMovingAverageOptimizerWithNonDefaultDecay(self):
    optimizer_text_proto = """
      adam_optimizer: {
        learning_rate: {
          constant_learning_rate {
            learning_rate: 0.002
          }
        }
      }
      use_moving_average: True
      moving_average_decay: 0.2
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    optimizer, _ = optimizer_builder.build(optimizer_proto)
    self.assertIsInstance(optimizer, contrib_opt.MovingAverageOptimizer)
    # TODO(rathodv): Find a way to not depend on the private members.
    self.assertAlmostEqual(optimizer._ema._decay, 0.2)

  def testBuildEmptyOptimizer(self):
    optimizer_text_proto = """
    """
    optimizer_proto = optimizer_pb2.Optimizer()
    text_format.Merge(optimizer_text_proto, optimizer_proto)
    with self.assertRaises(ValueError):
      optimizer_builder.build(optimizer_proto)


if __name__ == '__main__':
  tf.test.main()
