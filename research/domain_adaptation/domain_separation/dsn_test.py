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
"""Tests for DSN model assembly functions."""

import numpy as np
import tensorflow as tf

import dsn


class HelperFunctionsTest(tf.test.TestCase):

  def testBasicDomainSeparationStartPoint(self):
    with self.test_session() as sess:
      # Test for when global_step < domain_separation_startpoint
      step = tf.contrib.slim.get_or_create_global_step()
      sess.run(tf.global_variables_initializer())  # global_step = 0
      params = {'domain_separation_startpoint': 2}
      weight = dsn.dsn_loss_coefficient(params)
      weight_np = sess.run(weight)
      self.assertAlmostEqual(weight_np, 1e-10)

      step_op = tf.assign_add(step, 1)
      step_np = sess.run(step_op)  # global_step = 1
      weight = dsn.dsn_loss_coefficient(params)
      weight_np = sess.run(weight)
      self.assertAlmostEqual(weight_np, 1e-10)

      # Test for when global_step >= domain_separation_startpoint
      step_np = sess.run(step_op)  # global_step = 2
      tf.logging.info(step_np)
      weight = dsn.dsn_loss_coefficient(params)
      weight_np = sess.run(weight)
      self.assertAlmostEqual(weight_np, 1.0)


class DsnModelAssemblyTest(tf.test.TestCase):

  def _testBuildDefaultModel(self):
    images = tf.to_float(np.random.rand(32, 28, 28, 1))
    labels = {}
    labels['classes'] = tf.one_hot(
        tf.to_int32(np.random.randint(0, 9, (32))), 10)

    params = {
        'use_separation': True,
        'layers_to_regularize': 'fc3',
        'weight_decay': 0.0,
        'ps_tasks': 1,
        'domain_separation_startpoint': 1,
        'alpha_weight': 1,
        'beta_weight': 1,
        'gamma_weight': 1,
        'recon_loss_name': 'sum_of_squares',
        'decoder_name': 'small_decoder',
        'encoder_name': 'default_encoder',
    }
    return images, labels, params

  def testBuildModelDann(self):
    images, labels, params = self._testBuildDefaultModel()

    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'dann_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 6)

  def testBuildModelDannSumOfPairwiseSquares(self):
    images, labels, params = self._testBuildDefaultModel()

    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'dann_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 6)

  def testBuildModelDannMultiPSTasks(self):
    images, labels, params = self._testBuildDefaultModel()
    params['ps_tasks'] = 10
    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'dann_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 6)

  def testBuildModelMmd(self):
    images, labels, params = self._testBuildDefaultModel()

    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'mmd_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 6)

  def testBuildModelCorr(self):
    images, labels, params = self._testBuildDefaultModel()

    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'correlation_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 6)

  def testBuildModelNoDomainAdaptation(self):
    images, labels, params = self._testBuildDefaultModel()
    params['use_separation'] = False
    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels, 'none',
                       params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
      self.assertEqual(len(loss_tensors), 1)
      self.assertEqual(len(tf.contrib.losses.get_regularization_losses()), 0)

  def testBuildModelNoAdaptationWeightDecay(self):
    images, labels, params = self._testBuildDefaultModel()
    params['use_separation'] = False
    params['weight_decay'] = 1e-5
    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels, 'none',
                       params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
      self.assertEqual(len(loss_tensors), 1)
      self.assertTrue(len(tf.contrib.losses.get_regularization_losses()) >= 1)

  def testBuildModelNoSeparation(self):
    images, labels, params = self._testBuildDefaultModel()
    params['use_separation'] = False
    with self.test_session():
      dsn.create_model(images, labels,
                       tf.cast(tf.ones([32,]), tf.bool), images, labels,
                       'dann_loss', params, 'dann_mnist')
      loss_tensors = tf.contrib.losses.get_losses()
    self.assertEqual(len(loss_tensors), 2)


if __name__ == '__main__':
  tf.test.main()
