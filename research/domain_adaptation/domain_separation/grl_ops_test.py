# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Tests for grl_ops."""

#from models.domain_adaptation.domain_separation import grl_op_grads  # pylint: disable=unused-import
#from models.domain_adaptation.domain_separation import grl_op_shapes  # pylint: disable=unused-import
import tensorflow as tf

import grl_op_grads
import grl_ops

FLAGS = tf.app.flags.FLAGS


class GRLOpsTest(tf.test.TestCase):

  def testGradientReversalOp(self):
    with tf.Graph().as_default():
      with self.test_session():
        # Test that in forward prop, gradient reversal op acts as the
        # identity operation.
        examples = tf.constant([5.0, 4.0, 3.0, 2.0, 1.0])
        output = grl_ops.gradient_reversal(examples)
        expected_output = examples
        self.assertAllEqual(output.eval(), expected_output.eval())

        # Test that shape inference works as expected.
        self.assertAllEqual(output.get_shape(), expected_output.get_shape())

        # Test that in backward prop, gradient reversal op multiplies
        # gradients by -1.
        examples = tf.constant([[1.0]])
        w = tf.get_variable(name='w', shape=[1, 1])
        b = tf.get_variable(name='b', shape=[1])
        init_op = tf.global_variables_initializer()
        init_op.run()
        features = tf.nn.xw_plus_b(examples, w, b)
        # Construct two outputs: features layer passes directly to output1, but
        # features layer passes through a gradient reversal layer before
        # reaching output2.
        output1 = features
        output2 = grl_ops.gradient_reversal(features)
        gold = tf.constant([1.0])
        loss1 = gold - output1
        loss2 = gold - output2
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        grads_and_vars_1 = opt.compute_gradients(loss1,
                                                 tf.trainable_variables())
        grads_and_vars_2 = opt.compute_gradients(loss2,
                                                 tf.trainable_variables())
        self.assertAllEqual(len(grads_and_vars_1), len(grads_and_vars_2))
        for i in range(len(grads_and_vars_1)):
          g1 = grads_and_vars_1[i][0]
          g2 = grads_and_vars_2[i][0]
          # Verify that gradients of loss1 are the negative of gradients of
          # loss2.
          self.assertAllEqual(tf.negative(g1).eval(), g2.eval())

if __name__ == '__main__':
  tf.test.main()
