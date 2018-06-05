# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Tests for op_regularizer_manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from morph_net.framework import op_regularizer_manager as orm
from morph_net.testing import op_regularizer_stub

layers = tf.contrib.layers


def _get_op(name):
  return tf.get_default_graph().get_operation_by_name(name)


class TestOpRegularizerManager(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(12)
    np.random.seed(665544)

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with tf.contrib.framework.arg_scope([layers.conv2d], **params) as sc:
      return sc

  @parameterized.named_parameters(('Batch_no_par1', True, False, 'conv1'),
                                  ('Batch_par1', True, True, 'conv1'),
                                  ('NoBatch_no_par1', False, False, 'conv1'),
                                  ('NoBatch_par2', False, True, 'conv2'),
                                  ('Batch_no_par2', True, False, 'conv2'),
                                  ('Batch_par2', True, True, 'conv2'),
                                  ('Batch_par3', True, True, 'conv3'),
                                  ('NoBatch_par3', False, True, 'conv3'),
                                  ('NoBatch_no_par3', False, False, 'conv3'))
  def testSimpleOpGetRegularizer(self, use_batch_norm, use_partitioner, scope):
    # Tests the alive patern of the conv and relu ops.
    # use_batch_norm: A Boolean. Inidcats if batch norm should be used.
    # use_partitioner: A Boolean. Inidcats if a fixed_size_partitioner should be
    #   used.
    # scope: A String. with the scope to test.
    sc = self._batch_norm_scope() if use_batch_norm else []
    partitioner = tf.fixed_size_partitioner(2) if use_partitioner else None
    with tf.contrib.framework.arg_scope(sc):
      with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
        final_op = op_regularizer_stub.build_model()

    op_reg_manager = orm.OpRegularizerManager([final_op],
                                              op_regularizer_stub.MOCK_REG_DICT)
    expected_alive = op_regularizer_stub.expected_alive()
    with self.test_session():
      conv_reg = op_reg_manager.get_regularizer(_get_op(scope + '/Conv2D'))
      self.assertAllEqual(expected_alive[scope],
                          conv_reg.alive_vector.eval())

      relu_reg = op_reg_manager.get_regularizer(_get_op(scope +  '/Relu'))
      self.assertAllEqual(expected_alive[scope],
                          relu_reg.alive_vector.eval())

  @parameterized.named_parameters(('Batch_no_par', True, False),
                                  ('Batch_par', True, True),
                                  ('NoBatch_no_par', False, False),
                                  ('NoBatch_par', False, True))
  def testConcatOpGetRegularizer(self, use_batch_norm, use_partitioner):
    sc = self._batch_norm_scope() if use_batch_norm else []
    partitioner = tf.fixed_size_partitioner(2) if use_partitioner else None
    with tf.contrib.framework.arg_scope(sc):
      with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
        final_op = op_regularizer_stub.build_model()
    op_reg_manager = orm.OpRegularizerManager([final_op],
                                              op_regularizer_stub.MOCK_REG_DICT)
    expected_alive = op_regularizer_stub.expected_alive()

    expected = np.logical_or(expected_alive['conv4'],
                             expected_alive['concat'])
    with self.test_session():
      conv_reg = op_reg_manager.get_regularizer(_get_op('conv4/Conv2D'))
      self.assertAllEqual(expected, conv_reg.alive_vector.eval())

      relu_reg = op_reg_manager.get_regularizer(_get_op('conv4/Relu'))
      self.assertAllEqual(expected, relu_reg.alive_vector.eval())

  @parameterized.named_parameters(('Concat_5', True, 5),
                                  ('Concat_7', True, 7),
                                  ('Add_6', False, 6))
  def testGetRegularizerForConcatWithNone(self, test_concat, depth):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv2')
    other_input = tf.add(
        tf.identity(tf.constant(3.0, shape=[1, 17, 19, depth])), 3.0)
    # other_input has None as regularizer.
    concat = tf.concat([other_input, conv2], 3)
    output = tf.add(concat, concat, name='output_out')
    op = concat.op if test_concat else output.op
    op_reg_manager = orm.OpRegularizerManager([output.op],
                                              op_regularizer_stub.MOCK_REG_DICT)
    expected_alive = op_regularizer_stub.expected_alive()

    with self.test_session():
      alive = op_reg_manager.get_regularizer(op).alive_vector.eval()
      self.assertAllEqual([True] * depth, alive[:depth])
      self.assertAllEqual(expected_alive['conv2'], alive[depth:])

  @parameterized.named_parameters(('add', tf.add),
                                  ('div', tf.divide),
                                  ('mul', tf.multiply),
                                  ('max', tf.maximum),
                                  ('min', tf.minimum),
                                  ('l2', tf.squared_difference))
  def testGroupingOps(self, tested_op):
    th, size = 0.5, 11
    image = tf.constant(0.5, shape=[1, 17, 19, 3])

    conv1 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv1')
    conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv2')
    res = tested_op(conv1, conv2)
    reg = {'conv1': np.random.random(size), 'conv2': np.random.random(size)}

    def regularizer(conv_op, manager=None):
      del manager  # unused
      for prefix in ['conv1', 'conv2']:
        if conv_op.name.startswith(prefix):
          return op_regularizer_stub.OpRegularizerStub(
              reg[prefix], reg[prefix] > th)

    op_reg_manager = orm.OpRegularizerManager([res.op], {'Conv2D': regularizer})
    with self.test_session():
      alive = op_reg_manager.get_regularizer(res.op).alive_vector.eval()
      self.assertAllEqual(alive,
                          np.logical_or(reg['conv1'] > th, reg['conv2'] > th))

if __name__ == '__main__':
  tf.test.main()
