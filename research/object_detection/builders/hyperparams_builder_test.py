# Lint as: python2, python3
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

"""Tests object_detection.core.hyperparams_builder."""

import unittest
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.core import freezable_batch_norm
from object_detection.protos import hyperparams_pb2
from object_detection.utils import tf_version


def _get_scope_key(op):
  return getattr(op, '_key_op', str(op))


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only tests.')
class HyperparamsBuilderTest(tf.test.TestCase):

  def test_default_arg_scope_has_conv2d_op(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    self.assertIn(_get_scope_key(slim.conv2d), scope)

  def test_default_arg_scope_has_separable_conv2d_op(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    self.assertIn(_get_scope_key(slim.separable_conv2d), scope)

  def test_default_arg_scope_has_conv2d_transpose_op(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    self.assertIn(_get_scope_key(slim.conv2d_transpose), scope)

  def test_explicit_fc_op_arg_scope_has_fully_connected_op(self):
    conv_hyperparams_text_proto = """
      op: FC
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    self.assertIn(_get_scope_key(slim.fully_connected), scope)

  def test_separable_conv2d_and_conv2d_and_transpose_have_same_parameters(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    kwargs_1, kwargs_2, kwargs_3 = scope.values()
    self.assertDictEqual(kwargs_1, kwargs_2)
    self.assertDictEqual(kwargs_1, kwargs_3)

  def test_return_l1_regularized_weights(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.5
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = list(scope.values())[0]
    regularizer = conv_scope_arguments['weights_regularizer']
    weights = np.array([1., -1, 4., 2.])
    with self.test_session() as sess:
      result = sess.run(regularizer(tf.constant(weights)))
    self.assertAllClose(np.abs(weights).sum() * 0.5, result)

  def test_return_l2_regularizer_weights(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
          weight: 0.42
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]

    regularizer = conv_scope_arguments['weights_regularizer']
    weights = np.array([1., -1, 4., 2.])
    with self.test_session() as sess:
      result = sess.run(regularizer(tf.constant(weights)))
    self.assertAllClose(np.power(weights, 2).sum() / 2.0 * 0.42, result)

  def test_return_non_default_batch_norm_params_with_train_during_train(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: false
        scale: true
        epsilon: 0.03
        train: true
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
    batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
    self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
    self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
    self.assertFalse(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])
    self.assertTrue(batch_norm_params['is_training'])

  def test_return_batch_norm_params_with_notrain_during_eval(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: false
        scale: true
        epsilon: 0.03
        train: true
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=False)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
    batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
    self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
    self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
    self.assertFalse(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])
    self.assertFalse(batch_norm_params['is_training'])

  def test_return_batch_norm_params_with_notrain_when_train_is_false(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: false
        scale: true
        epsilon: 0.03
        train: false
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
    batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
    self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
    self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
    self.assertFalse(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])
    self.assertFalse(batch_norm_params['is_training'])

  def test_do_not_use_batch_norm_if_default(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['normalizer_fn'], None)

  def test_use_none_activation(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: NONE
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['activation_fn'], None)

  def test_use_relu_activation(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['activation_fn'], tf.nn.relu)

  def test_use_relu_6_activation(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['activation_fn'], tf.nn.relu6)

  def test_use_swish_activation(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: SWISH
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    self.assertEqual(conv_scope_arguments['activation_fn'], tf.nn.swish)

  def _assert_variance_in_range(self, initializer, shape, variance,
                                tol=1e-2):
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        var = tf.get_variable(
            name='test',
            shape=shape,
            dtype=tf.float32,
            initializer=initializer)
        sess.run(tf.global_variables_initializer())
        values = sess.run(var)
        self.assertAllClose(np.var(values), variance, tol, tol)

  def test_variance_in_range_with_variance_scaling_initializer_fan_in(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_IN
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 100.)

  def test_variance_in_range_with_variance_scaling_initializer_fan_out(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_OUT
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 40.)

  def test_variance_in_range_with_variance_scaling_initializer_fan_avg(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_AVG
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=4. / (100. + 40.))

  def test_variance_in_range_with_variance_scaling_initializer_uniform(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_IN
          uniform: true
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 100.)

  def test_variance_in_range_with_truncated_normal_initializer(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.8
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=0.49, tol=1e-1)

  def test_variance_in_range_with_random_normal_initializer(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        random_normal_initializer {
          mean: 0.0
          stddev: 0.8
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    scope_fn = hyperparams_builder.build(conv_hyperparams_proto,
                                         is_training=True)
    scope = scope_fn()
    conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
    initializer = conv_scope_arguments['weights_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=0.64, tol=1e-1)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only tests.')
class KerasHyperparamsBuilderTest(tf.test.TestCase):

  def _assert_variance_in_range(self, initializer, shape, variance,
                                tol=1e-2):
    var = tf.Variable(initializer(shape=shape, dtype=tf.float32))
    self.assertAllClose(np.var(var.numpy()), variance, tol, tol)

  def test_return_l1_regularized_weights_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.5
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    regularizer = keras_config.params()['kernel_regularizer']
    weights = np.array([1., -1, 4., 2.])
    result = regularizer(tf.constant(weights)).numpy()
    self.assertAllClose(np.abs(weights).sum() * 0.5, result)

  def test_return_l2_regularizer_weights_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
          weight: 0.42
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    regularizer = keras_config.params()['kernel_regularizer']
    weights = np.array([1., -1, 4., 2.])
    result = regularizer(tf.constant(weights)).numpy()
    self.assertAllClose(np.power(weights, 2).sum() / 2.0 * 0.42, result)

  def test_return_non_default_batch_norm_params_keras(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: false
        scale: true
        epsilon: 0.03
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    self.assertTrue(keras_config.use_batch_norm())
    batch_norm_params = keras_config.batch_norm_params()
    self.assertAlmostEqual(batch_norm_params['momentum'], 0.7)
    self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
    self.assertFalse(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])

    batch_norm_layer = keras_config.build_batch_norm()
    self.assertIsInstance(batch_norm_layer,
                          freezable_batch_norm.FreezableBatchNorm)

  def test_return_non_default_batch_norm_params_keras_override(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: false
        scale: true
        epsilon: 0.03
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    self.assertTrue(keras_config.use_batch_norm())
    batch_norm_params = keras_config.batch_norm_params(momentum=0.4)
    self.assertAlmostEqual(batch_norm_params['momentum'], 0.4)
    self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
    self.assertFalse(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])

  def test_do_not_use_batch_norm_if_default_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    self.assertFalse(keras_config.use_batch_norm())
    self.assertEqual(keras_config.batch_norm_params(), {})

    # The batch norm builder should build an identity Lambda layer
    identity_layer = keras_config.build_batch_norm()
    self.assertIsInstance(identity_layer,
                          tf.keras.layers.Lambda)

  def test_do_not_use_bias_if_batch_norm_center_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: true
        scale: true
        epsilon: 0.03
        train: true
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    self.assertTrue(keras_config.use_batch_norm())
    batch_norm_params = keras_config.batch_norm_params()
    self.assertTrue(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])
    hyperparams = keras_config.params()
    self.assertFalse(hyperparams['use_bias'])

  def test_force_use_bias_if_batch_norm_center_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        decay: 0.7
        center: true
        scale: true
        epsilon: 0.03
        train: true
      }
      force_use_bias: true
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)

    self.assertTrue(keras_config.use_batch_norm())
    batch_norm_params = keras_config.batch_norm_params()
    self.assertTrue(batch_norm_params['center'])
    self.assertTrue(batch_norm_params['scale'])
    hyperparams = keras_config.params()
    self.assertTrue(hyperparams['use_bias'])

  def test_use_none_activation_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: NONE
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    self.assertIsNone(keras_config.params()['activation'])
    self.assertIsNone(
        keras_config.params(include_activation=True)['activation'])
    activation_layer = keras_config.build_activation_layer()
    self.assertIsInstance(activation_layer, tf.keras.layers.Lambda)
    self.assertEqual(activation_layer.function, tf.identity)

  def test_use_relu_activation_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    self.assertIsNone(keras_config.params()['activation'])
    self.assertEqual(
        keras_config.params(include_activation=True)['activation'], tf.nn.relu)
    activation_layer = keras_config.build_activation_layer()
    self.assertIsInstance(activation_layer, tf.keras.layers.Lambda)
    self.assertEqual(activation_layer.function, tf.nn.relu)

  def test_use_relu_6_activation_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    self.assertIsNone(keras_config.params()['activation'])
    self.assertEqual(
        keras_config.params(include_activation=True)['activation'], tf.nn.relu6)
    activation_layer = keras_config.build_activation_layer()
    self.assertIsInstance(activation_layer, tf.keras.layers.Lambda)
    self.assertEqual(activation_layer.function, tf.nn.relu6)

  def test_use_swish_activation_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: SWISH
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    self.assertIsNone(keras_config.params()['activation'])
    self.assertEqual(
        keras_config.params(include_activation=True)['activation'], tf.nn.swish)
    activation_layer = keras_config.build_activation_layer()
    self.assertIsInstance(activation_layer, tf.keras.layers.Lambda)
    self.assertEqual(activation_layer.function, tf.nn.swish)

  def test_override_activation_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    new_params = keras_config.params(activation=tf.nn.relu)
    self.assertEqual(new_params['activation'], tf.nn.relu)

  def test_variance_in_range_with_variance_scaling_initializer_fan_in_keras(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_IN
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 100.)

  def test_variance_in_range_with_variance_scaling_initializer_fan_out_keras(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_OUT
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 40.)

  def test_variance_in_range_with_variance_scaling_initializer_fan_avg_keras(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_AVG
          uniform: false
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=4. / (100. + 40.))

  def test_variance_in_range_with_variance_scaling_initializer_uniform_keras(
      self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        variance_scaling_initializer {
          factor: 2.0
          mode: FAN_IN
          uniform: true
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=2. / 100.)

  def test_variance_in_range_with_truncated_normal_initializer_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.8
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=0.49, tol=1e-1)

  def test_variance_in_range_with_random_normal_initializer_keras(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        random_normal_initializer {
          mean: 0.0
          stddev: 0.8
        }
      }
    """
    conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
    keras_config = hyperparams_builder.KerasLayerHyperparams(
        conv_hyperparams_proto)
    initializer = keras_config.params()['kernel_initializer']
    self._assert_variance_in_range(initializer, shape=[100, 40],
                                   variance=0.64, tol=1e-1)

if __name__ == '__main__':
  tf.test.main()
