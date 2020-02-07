# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for model_deploy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim

from deployment import model_deploy

slim = contrib_slim


class DeploymentConfigTest(tf.test.TestCase):

  def testDefaults(self):
    deploy_config = model_deploy.DeploymentConfig()

    self.assertEqual(slim.get_variables(), [])
    self.assertEqual(deploy_config.caching_device(), None)
    self.assertDeviceEqual(deploy_config.clone_device(0), 'GPU:0')
    self.assertEqual(deploy_config.clone_scope(0), '')
    self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

  def testCPUonly(self):
    deploy_config = model_deploy.DeploymentConfig(clone_on_cpu=True)

    self.assertEqual(deploy_config.caching_device(), None)
    self.assertDeviceEqual(deploy_config.clone_device(0), 'CPU:0')
    self.assertEqual(deploy_config.clone_scope(0), '')
    self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

  def testMultiGPU(self):
    deploy_config = model_deploy.DeploymentConfig(num_clones=2)

    self.assertEqual(deploy_config.caching_device(), None)
    self.assertDeviceEqual(deploy_config.clone_device(0), 'GPU:0')
    self.assertDeviceEqual(deploy_config.clone_device(1), 'GPU:1')
    self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
    self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
    self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
    self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

  def testPS(self):
    deploy_config = model_deploy.DeploymentConfig(num_clones=1, num_ps_tasks=1)

    self.assertDeviceEqual(deploy_config.clone_device(0),
                           '/job:worker/device:GPU:0')
    self.assertEqual(deploy_config.clone_scope(0), '')
    self.assertDeviceEqual(deploy_config.optimizer_device(),
                           '/job:worker/device:CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(),
                           '/job:worker/device:CPU:0')
    with tf.device(deploy_config.variables_device()):
      a = tf.Variable(0)
      b = tf.Variable(0)
      c = tf.no_op()
      d = slim.variable('a', [],
                        caching_device=deploy_config.caching_device())
    self.assertDeviceEqual(a.device, '/job:ps/task:0/device:CPU:0')
    self.assertDeviceEqual(a.device, a.value().device)
    self.assertDeviceEqual(b.device, '/job:ps/task:0/device:CPU:0')
    self.assertDeviceEqual(b.device, b.value().device)
    self.assertDeviceEqual(c.device, '')
    self.assertDeviceEqual(d.device, '/job:ps/task:0/device:CPU:0')
    self.assertDeviceEqual(d.value().device, '')

  def testMultiGPUPS(self):
    deploy_config = model_deploy.DeploymentConfig(num_clones=2, num_ps_tasks=1)

    self.assertEqual(deploy_config.caching_device()(tf.no_op()), '')
    self.assertDeviceEqual(deploy_config.clone_device(0),
                           '/job:worker/device:GPU:0')
    self.assertDeviceEqual(deploy_config.clone_device(1),
                           '/job:worker/device:GPU:1')
    self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
    self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
    self.assertDeviceEqual(deploy_config.optimizer_device(),
                           '/job:worker/device:CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(),
                           '/job:worker/device:CPU:0')

  def testReplicasPS(self):
    deploy_config = model_deploy.DeploymentConfig(num_replicas=2,
                                                  num_ps_tasks=2)

    self.assertDeviceEqual(deploy_config.clone_device(0),
                           '/job:worker/device:GPU:0')
    self.assertEqual(deploy_config.clone_scope(0), '')
    self.assertDeviceEqual(deploy_config.optimizer_device(),
                           '/job:worker/device:CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(),
                           '/job:worker/device:CPU:0')

  def testReplicasMultiGPUPS(self):
    deploy_config = model_deploy.DeploymentConfig(num_replicas=2,
                                                  num_clones=2,
                                                  num_ps_tasks=2)
    self.assertDeviceEqual(deploy_config.clone_device(0),
                           '/job:worker/device:GPU:0')
    self.assertDeviceEqual(deploy_config.clone_device(1),
                           '/job:worker/device:GPU:1')
    self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
    self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
    self.assertDeviceEqual(deploy_config.optimizer_device(),
                           '/job:worker/device:CPU:0')
    self.assertDeviceEqual(deploy_config.inputs_device(),
                           '/job:worker/device:CPU:0')

  def testVariablesPS(self):
    deploy_config = model_deploy.DeploymentConfig(num_ps_tasks=2)

    with tf.device(deploy_config.variables_device()):
      a = tf.Variable(0)
      b = tf.Variable(0)
      c = tf.no_op()
      d = slim.variable('a', [],
                        caching_device=deploy_config.caching_device())

    self.assertDeviceEqual(a.device, '/job:ps/task:0/device:CPU:0')
    self.assertDeviceEqual(a.device, a.value().device)
    self.assertDeviceEqual(b.device, '/job:ps/task:1/device:CPU:0')
    self.assertDeviceEqual(b.device, b.value().device)
    self.assertDeviceEqual(c.device, '')
    self.assertDeviceEqual(d.device, '/job:ps/task:0/device:CPU:0')
    self.assertDeviceEqual(d.value().device, '')


def LogisticClassifier(inputs, labels, scope=None, reuse=None):
  with tf.variable_scope(scope, 'LogisticClassifier', [inputs, labels],
                         reuse=reuse):
    predictions = slim.fully_connected(inputs, 1, activation_fn=tf.sigmoid,
                                       scope='fully_connected')
    slim.losses.log_loss(predictions, labels)
    return predictions


def BatchNormClassifier(inputs, labels, scope=None, reuse=None):
  with tf.variable_scope(scope, 'BatchNormClassifier', [inputs, labels],
                         reuse=reuse):
    inputs = slim.batch_norm(inputs, decay=0.1, fused=True)
    predictions = slim.fully_connected(inputs, 1,
                                       activation_fn=tf.sigmoid,
                                       scope='fully_connected')
    slim.losses.log_loss(predictions, labels)
    return predictions


class CreatecloneTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)
    self._logdir = self.get_temp_dir()

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testCreateLogisticClassifier(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = LogisticClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      clone = clones[0]
      self.assertEqual(len(slim.get_variables()), 2)
      for v in slim.get_variables():
        self.assertDeviceEqual(v.device, 'CPU:0')
        self.assertDeviceEqual(v.value().device, 'CPU:0')
      self.assertEqual(clone.outputs.op.name,
                       'LogisticClassifier/fully_connected/Sigmoid')
      self.assertEqual(clone.scope, '')
      self.assertDeviceEqual(clone.device, 'GPU:0')
      self.assertEqual(len(slim.losses.get_losses()), 1)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(update_ops, [])

  def testCreateSingleclone(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      clone = clones[0]
      self.assertEqual(len(slim.get_variables()), 5)
      for v in slim.get_variables():
        self.assertDeviceEqual(v.device, 'CPU:0')
        self.assertDeviceEqual(v.value().device, 'CPU:0')
      self.assertEqual(clone.outputs.op.name,
                       'BatchNormClassifier/fully_connected/Sigmoid')
      self.assertEqual(clone.scope, '')
      self.assertDeviceEqual(clone.device, 'GPU:0')
      self.assertEqual(len(slim.losses.get_losses()), 1)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), 2)

  def testCreateMulticlone(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      num_clones = 4
      deploy_config = model_deploy.DeploymentConfig(num_clones=num_clones)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(slim.get_variables()), 5)
      for v in slim.get_variables():
        self.assertDeviceEqual(v.device, 'CPU:0')
        self.assertDeviceEqual(v.value().device, 'CPU:0')
      self.assertEqual(len(clones), num_clones)
      for i, clone in enumerate(clones):
        self.assertEqual(
            clone.outputs.op.name,
            'clone_%d/BatchNormClassifier/fully_connected/Sigmoid' % i)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone.scope)
        self.assertEqual(len(update_ops), 2)
        self.assertEqual(clone.scope, 'clone_%d/' % i)
        self.assertDeviceEqual(clone.device, 'GPU:%d' % i)

  def testCreateOnecloneWithPS(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1,
                                                    num_ps_tasks=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(clones), 1)
      clone = clones[0]
      self.assertEqual(clone.outputs.op.name,
                       'BatchNormClassifier/fully_connected/Sigmoid')
      self.assertDeviceEqual(clone.device, '/job:worker/device:GPU:0')
      self.assertEqual(clone.scope, '')
      self.assertEqual(len(slim.get_variables()), 5)
      for v in slim.get_variables():
        self.assertDeviceEqual(v.device, '/job:ps/task:0/CPU:0')
        self.assertDeviceEqual(v.device, v.value().device)

  def testCreateMulticloneWithPS(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=2,
                                                    num_ps_tasks=2)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(slim.get_variables()), 5)
      for i, v in enumerate(slim.get_variables()):
        t = i % 2
        self.assertDeviceEqual(v.device, '/job:ps/task:%d/device:CPU:0' % t)
        self.assertDeviceEqual(v.device, v.value().device)
      self.assertEqual(len(clones), 2)
      for i, clone in enumerate(clones):
        self.assertEqual(
            clone.outputs.op.name,
            'clone_%d/BatchNormClassifier/fully_connected/Sigmoid' % i)
        self.assertEqual(clone.scope, 'clone_%d/' % i)
        self.assertDeviceEqual(clone.device, '/job:worker/device:GPU:%d' % i)


class OptimizeclonesTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)
    self._logdir = self.get_temp_dir()

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testCreateLogisticClassifier(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = LogisticClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(slim.get_variables()), 2)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(update_ops, [])

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      total_loss, grads_and_vars = model_deploy.optimize_clones(clones,
                                                                optimizer)
      self.assertEqual(len(grads_and_vars), len(tf.trainable_variables()))
      self.assertEqual(total_loss.op.name, 'total_loss')
      for g, v in grads_and_vars:
        self.assertDeviceEqual(g.device, 'GPU:0')
        self.assertDeviceEqual(v.device, 'CPU:0')

  def testCreateSingleclone(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(slim.get_variables()), 5)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), 2)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      total_loss, grads_and_vars = model_deploy.optimize_clones(clones,
                                                                optimizer)
      self.assertEqual(len(grads_and_vars), len(tf.trainable_variables()))
      self.assertEqual(total_loss.op.name, 'total_loss')
      for g, v in grads_and_vars:
        self.assertDeviceEqual(g.device, 'GPU:0')
        self.assertDeviceEqual(v.device, 'CPU:0')

  def testCreateMulticlone(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      clone_args = (tf_inputs, tf_labels)
      num_clones = 4
      deploy_config = model_deploy.DeploymentConfig(num_clones=num_clones)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, clone_args)
      self.assertEqual(len(slim.get_variables()), 5)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), num_clones * 2)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      total_loss, grads_and_vars = model_deploy.optimize_clones(clones,
                                                                optimizer)
      self.assertEqual(len(grads_and_vars), len(tf.trainable_variables()))
      self.assertEqual(total_loss.op.name, 'total_loss')
      for g, v in grads_and_vars:
        self.assertDeviceEqual(g.device, '')
        self.assertDeviceEqual(v.device, 'CPU:0')

  def testCreateMulticloneCPU(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      model_args = (tf_inputs, tf_labels)
      num_clones = 4
      deploy_config = model_deploy.DeploymentConfig(num_clones=num_clones,
                                                    clone_on_cpu=True)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, model_args)
      self.assertEqual(len(slim.get_variables()), 5)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), num_clones * 2)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      total_loss, grads_and_vars = model_deploy.optimize_clones(clones,
                                                                optimizer)
      self.assertEqual(len(grads_and_vars), len(tf.trainable_variables()))
      self.assertEqual(total_loss.op.name, 'total_loss')
      for g, v in grads_and_vars:
        self.assertDeviceEqual(g.device, '')
        self.assertDeviceEqual(v.device, 'CPU:0')

  def testCreateOnecloneWithPS(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      model_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=1,
                                                    num_ps_tasks=1)

      self.assertEqual(slim.get_variables(), [])
      clones = model_deploy.create_clones(deploy_config, model_fn, model_args)
      self.assertEqual(len(slim.get_variables()), 5)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), 2)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      total_loss, grads_and_vars = model_deploy.optimize_clones(clones,
                                                                optimizer)
      self.assertEqual(len(grads_and_vars), len(tf.trainable_variables()))
      self.assertEqual(total_loss.op.name, 'total_loss')
      for g, v in grads_and_vars:
        self.assertDeviceEqual(g.device, '/job:worker/device:GPU:0')
        self.assertDeviceEqual(v.device, '/job:ps/task:0/CPU:0')


class DeployTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)
    self._logdir = self.get_temp_dir()

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _addBesselsCorrection(self, sample_size, expected_var):
    correction_factor = sample_size / (sample_size - 1)
    expected_var *= correction_factor
    return expected_var

  def testLocalTrainOp(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      model_fn = BatchNormClassifier
      model_args = (tf_inputs, tf_labels)
      deploy_config = model_deploy.DeploymentConfig(num_clones=2,
                                                    clone_on_cpu=True)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      self.assertEqual(slim.get_variables(), [])
      model = model_deploy.deploy(deploy_config, model_fn, model_args,
                                  optimizer=optimizer)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEqual(len(update_ops), 4)
      self.assertEqual(len(model.clones), 2)
      self.assertEqual(model.total_loss.op.name, 'total_loss')
      self.assertEqual(model.summary_op.op.name, 'summary_op/summary_op')
      self.assertEqual(model.train_op.op.name, 'train_op')

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        moving_mean = contrib_framework.get_variables_by_name('moving_mean')[0]
        moving_variance = contrib_framework.get_variables_by_name(
            'moving_variance')[0]
        initial_loss = sess.run(model.total_loss)
        initial_mean, initial_variance = sess.run([moving_mean,
                                                   moving_variance])
        self.assertAllClose(initial_mean, [0.0, 0.0, 0.0, 0.0])
        self.assertAllClose(initial_variance, [1.0, 1.0, 1.0, 1.0])
        for _ in range(10):
          sess.run(model.train_op)
        final_loss = sess.run(model.total_loss)
        self.assertLess(final_loss, initial_loss / 5.0)

        final_mean, final_variance = sess.run([moving_mean,
                                               moving_variance])
        expected_mean = np.array([0.125, 0.25, 0.375, 0.25])
        expected_var = np.array([0.109375, 0.1875, 0.234375, 0.1875])
        expected_var = self._addBesselsCorrection(16, expected_var)
        self.assertAllClose(final_mean, expected_mean)
        self.assertAllClose(final_variance, expected_var)

  def testNoSummariesOnGPU(self):
    with tf.Graph().as_default():
      deploy_config = model_deploy.DeploymentConfig(num_clones=2)

      # clone function creates a fully_connected layer with a regularizer loss.
      def ModelFn():
        inputs = tf.constant(1.0, shape=(10, 20), dtype=tf.float32)
        reg = contrib_layers.l2_regularizer(0.001)
        contrib_layers.fully_connected(inputs, 30, weights_regularizer=reg)

      model = model_deploy.deploy(
          deploy_config, ModelFn,
          optimizer=tf.train.GradientDescentOptimizer(1.0))
      # The model summary op should have a few summary inputs and all of them
      # should be on the CPU.
      self.assertTrue(model.summary_op.op.inputs)
      for inp in  model.summary_op.op.inputs:
        self.assertEqual('/device:CPU:0', inp.device)

  def testNoSummariesOnGPUForEvals(self):
    with tf.Graph().as_default():
      deploy_config = model_deploy.DeploymentConfig(num_clones=2)

      # clone function creates a fully_connected layer with a regularizer loss.
      def ModelFn():
        inputs = tf.constant(1.0, shape=(10, 20), dtype=tf.float32)
        reg = contrib_layers.l2_regularizer(0.001)
        contrib_layers.fully_connected(inputs, 30, weights_regularizer=reg)

      # No optimizer here, it's an eval.
      model = model_deploy.deploy(deploy_config, ModelFn)
      # The model summary op should have a few summary inputs and all of them
      # should be on the CPU.
      self.assertTrue(model.summary_op.op.inputs)
      for inp in  model.summary_op.op.inputs:
        self.assertEqual('/device:CPU:0', inp.device)


if __name__ == '__main__':
  tf.test.main()
