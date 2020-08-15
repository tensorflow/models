# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for mobilenet_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from six.moves import range
import tensorflow.compat.v1 as tf
import tf_slim as slim
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2


def find_ops(optype):
  """Find ops of a given type in graphdef or a graph.

  Args:
    optype: operation type (e.g. Conv2D)
  Returns:
     List of operations.
  """
  gd = tf.get_default_graph()
  return [var for var in gd.get_operations() if var.type == optype]


class MobilenetV2Test(tf.test.TestCase):

  def testCreation(self):
    spec = dict(mobilenet_v2.V2_DEF)
    _, ep = mobilenet.mobilenet(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=spec)
    num_convs = len(find_ops('Conv2D'))

    # This is mostly a sanity test. No deep reason for these particular
    # constants.
    #
    # All but first 2 and last one have  two convolutions, and there is one
    # extra conv that is not in the spec. (logits)
    self.assertEqual(num_convs, len(spec['spec']) * 2 - 2)
    # Check that depthwise are exposed.
    for i in range(2, 17):
      self.assertIn('layer_%d/depthwise_output' % i, ep)

  def testCreationNoClasses(self):
    spec = copy.deepcopy(mobilenet_v2.V2_DEF)
    net, ep = mobilenet.mobilenet(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=spec,
        num_classes=None)
    self.assertIs(net, ep['global_pool'])

  def testImageSizes(self):
    for input_size, output_size in [(224, 7), (192, 6), (160, 5),
                                    (128, 4), (96, 3)]:
      tf.reset_default_graph()
      _, ep = mobilenet_v2.mobilenet(
          tf.placeholder(tf.float32, (10, input_size, input_size, 3)))

      self.assertEqual(ep['layer_18/output'].get_shape().as_list()[1:3],
                       [output_size] * 2)

  def testWithSplits(self):
    spec = copy.deepcopy(mobilenet_v2.V2_DEF)
    spec['overrides'] = {
        (ops.expanded_conv,): dict(split_expansion=2),
    }
    _, _ = mobilenet.mobilenet(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=spec)
    num_convs = len(find_ops('Conv2D'))
    # All but 3 op has 3 conv operatore, the remainign 3 have one
    # and there is one unaccounted.
    self.assertEqual(num_convs, len(spec['spec']) * 3 - 5)

  def testWithOutputStride8(self):
    out, _ = mobilenet.mobilenet_base(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=mobilenet_v2.V2_DEF,
        output_stride=8,
        scope='MobilenetV2')
    self.assertEqual(out.get_shape().as_list()[1:3], [28, 28])

  def testDivisibleBy(self):
    tf.reset_default_graph()
    mobilenet_v2.mobilenet(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=mobilenet_v2.V2_DEF,
        divisible_by=16,
        min_depth=32)
    s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
    s = set(s)
    self.assertSameElements([32, 64, 96, 160, 192, 320, 384, 576, 960, 1280,
                             1001], s)

  def testDivisibleByWithArgScope(self):
    tf.reset_default_graph()
    # Verifies that depth_multiplier arg scope actually works
    # if no default min_depth is provided.
    with slim.arg_scope((mobilenet.depth_multiplier,), min_depth=32):
      mobilenet_v2.mobilenet(
          tf.placeholder(tf.float32, (10, 224, 224, 2)),
          conv_defs=mobilenet_v2.V2_DEF,
          depth_multiplier=0.1)
      s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
      s = set(s)
      self.assertSameElements(s, [32, 192, 128, 1001])

  def testFineGrained(self):
    tf.reset_default_graph()
    # Verifies that depth_multiplier arg scope actually works
    # if no default min_depth is provided.

    mobilenet_v2.mobilenet(
        tf.placeholder(tf.float32, (10, 224, 224, 2)),
        conv_defs=mobilenet_v2.V2_DEF,
        depth_multiplier=0.01,
        finegrain_classification_mode=True)
    s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
    s = set(s)
    # All convolutions will be 8->48, except for the last one.
    self.assertSameElements(s, [8, 48, 1001, 1280])

  def testMobilenetBase(self):
    tf.reset_default_graph()
    # Verifies that mobilenet_base returns pre-pooling layer.
    with slim.arg_scope((mobilenet.depth_multiplier,), min_depth=32):
      net, _ = mobilenet_v2.mobilenet_base(
          tf.placeholder(tf.float32, (10, 224, 224, 16)),
          conv_defs=mobilenet_v2.V2_DEF,
          depth_multiplier=0.1)
      self.assertEqual(net.get_shape().as_list(), [10, 7, 7, 128])

  def testWithOutputStride16(self):
    tf.reset_default_graph()
    out, _ = mobilenet.mobilenet_base(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=mobilenet_v2.V2_DEF,
        output_stride=16)
    self.assertEqual(out.get_shape().as_list()[1:3], [14, 14])

  def testMultiplier(self):
    op = mobilenet.op
    new_def = copy.deepcopy(mobilenet_v2.V2_DEF)

    def inverse_multiplier(output_params, multiplier):
      output_params['num_outputs'] = int(
          output_params['num_outputs'] / multiplier)

    new_def['spec'][0] = op(
        slim.conv2d,
        kernel_size=(3, 3),
        multiplier_func=inverse_multiplier,
        num_outputs=16)
    _ = mobilenet_v2.mobilenet_base(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=new_def,
        depth_multiplier=0.1)
    s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
    # Expect first layer to be 160 (16 / 0.1), and other layers
    # their max(original size * 0.1, 8)
    self.assertEqual([160, 8, 48, 8, 48], s[:5])

  def testWithOutputStride8AndExplicitPadding(self):
    tf.reset_default_graph()
    out, _ = mobilenet.mobilenet_base(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=mobilenet_v2.V2_DEF,
        output_stride=8,
        use_explicit_padding=True,
        scope='MobilenetV2')
    self.assertEqual(out.get_shape().as_list()[1:3], [28, 28])

  def testWithOutputStride16AndExplicitPadding(self):
    tf.reset_default_graph()
    out, _ = mobilenet.mobilenet_base(
        tf.placeholder(tf.float32, (10, 224, 224, 16)),
        conv_defs=mobilenet_v2.V2_DEF,
        output_stride=16,
        use_explicit_padding=True)
    self.assertEqual(out.get_shape().as_list()[1:3], [14, 14])

  def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
    sc = mobilenet.training_scope(is_training=None)
    self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
        slim.batch_norm)])

  def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
    sc = mobilenet.training_scope(is_training=False)
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
    sc = mobilenet.training_scope(is_training=True)
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
    sc = mobilenet.training_scope()
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])


if __name__ == '__main__':
  tf.test.main()
