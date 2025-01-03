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
"""Tests for lstm_object_detection.models.mobilenet_defs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from lstm_object_detection.models import mobilenet_defs
from nets import mobilenet_v1
from nets.mobilenet import mobilenet_v2


class MobilenetV1DefsTest(tf.test.TestCase):

  def test_mobilenet_v1_lite_def(self):
    net, _ = mobilenet_v1.mobilenet_v1_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        final_endpoint='Conv2d_13_pointwise',
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v1_lite_def(1.0),
        use_explicit_padding=True,
        scope='MobilenetV1')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 1024])

  def test_mobilenet_v1_lite_def_depthmultiplier_half(self):
    net, _ = mobilenet_v1.mobilenet_v1_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        final_endpoint='Conv2d_13_pointwise',
        min_depth=8,
        depth_multiplier=0.5,
        conv_defs=mobilenet_defs.mobilenet_v1_lite_def(0.5),
        use_explicit_padding=True,
        scope='MobilenetV1')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 1024])

  def test_mobilenet_v1_lite_def_depthmultiplier_2x(self):
    net, _ = mobilenet_v1.mobilenet_v1_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        final_endpoint='Conv2d_13_pointwise',
        min_depth=8,
        depth_multiplier=2.0,
        conv_defs=mobilenet_defs.mobilenet_v1_lite_def(2.0),
        use_explicit_padding=True,
        scope='MobilenetV1')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 1024])

  def test_mobilenet_v1_lite_def_low_res(self):
    net, _ = mobilenet_v1.mobilenet_v1_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        final_endpoint='Conv2d_13_pointwise',
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v1_lite_def(1.0, low_res=True),
        use_explicit_padding=True,
        scope='MobilenetV1')
    self.assertEqual(net.get_shape().as_list(), [10, 20, 20, 1024])


class MobilenetV2DefsTest(tf.test.TestCase):

  def test_mobilenet_v2_lite_def(self):
    net, features = mobilenet_v2.mobilenet_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v2_lite_def(),
        use_explicit_padding=True,
        scope='MobilenetV2')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 320])
    self._assert_contains_op('MobilenetV2/expanded_conv_16/project/Identity')
    self.assertEqual(
        features['layer_3/expansion_output'].get_shape().as_list(),
        [10, 160, 160, 96])
    self.assertEqual(
        features['layer_4/expansion_output'].get_shape().as_list(),
        [10, 80, 80, 144])

  def test_mobilenet_v2_lite_def_is_quantized(self):
    net, _ = mobilenet_v2.mobilenet_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v2_lite_def(is_quantized=True),
        use_explicit_padding=True,
        scope='MobilenetV2')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 320])
    self._assert_contains_op('MobilenetV2/expanded_conv_16/project/Relu6')

  def test_mobilenet_v2_lite_def_low_res(self):
    net, _ = mobilenet_v2.mobilenet_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v2_lite_def(low_res=True),
        use_explicit_padding=True,
        scope='MobilenetV2')
    self.assertEqual(net.get_shape().as_list(), [10, 20, 20, 320])

  def test_mobilenet_v2_lite_def_reduced(self):
    net, features = mobilenet_v2.mobilenet_base(
        tf.placeholder(tf.float32, (10, 320, 320, 3)),
        min_depth=8,
        depth_multiplier=1.0,
        conv_defs=mobilenet_defs.mobilenet_v2_lite_def(reduced=True),
        use_explicit_padding=True,
        scope='MobilenetV2')
    self.assertEqual(net.get_shape().as_list(), [10, 10, 10, 320])
    self.assertEqual(
        features['layer_3/expansion_output'].get_shape().as_list(),
        [10, 160, 160, 48])
    self.assertEqual(
        features['layer_4/expansion_output'].get_shape().as_list(),
        [10, 80, 80, 72])

  def _assert_contains_op(self, op_name):
    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertIn(op_name, op_names)


if __name__ == '__main__':
  tf.test.main()
