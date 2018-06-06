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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import layers

mock = tf.test.mock


def dummy_apply_kernel(kernel_shape, kernel_initializer):
  kernel = tf.get_variable(
      'kernel', shape=kernel_shape, initializer=kernel_initializer)
  return tf.reduce_sum(kernel) + 1.5


class LayersTest(tf.test.TestCase):

  def test_pixel_norm_4d_images_returns_channel_normalized_images(self):
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    with self.test_session(use_gpu=True) as sess:
      output_np = sess.run(layers.pixel_norm(x))

    expected_np = [[[[0.46291006, 0.92582011, 1.38873017],
                     [0.78954202, 0.98692751, 1.18431306]],
                    [[0.87047803, 0.99483204, 1.11918604],
                     [0.90659684, 0.99725652, 1.08791625]]],
                   [[[0., 0., 0.], [-0.46291006, -0.92582011, -1.38873017]],
                    [[0.57735026, -1.15470052, 1.15470052],
                     [0.56195146, 1.40487862, 0.84292722]]]]
    self.assertNDArrayNear(output_np, expected_np, 1.0e-5)

  def test_get_validated_scale_invalid_scale_throws_exception(self):
    with self.assertRaises(ValueError):
      layers._get_validated_scale(0)

  def test_get_validated_scale_float_scale_returns_integer(self):
    self.assertEqual(layers._get_validated_scale(5.5), 5)

  def test_downscale_invalid_scale_throws_exception(self):
    with self.assertRaises(ValueError):
      layers.downscale(tf.constant([]), -1)

  def test_downscale_4d_images_returns_downscaled_images(self):
    x_np = np.array(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=np.float32)
    with self.test_session(use_gpu=True) as sess:
      x1_np, x2_np = sess.run(
          [layers.downscale(tf.constant(x_np), n) for n in [1, 2]])

    expected2_np = [[[[5.5, 6.5, 7.5]]], [[[0.5, 0.25, 0.5]]]]

    self.assertNDArrayNear(x1_np, x_np, 1.0e-5)
    self.assertNDArrayNear(x2_np, expected2_np, 1.0e-5)

  def test_upscale_invalid_scale_throws_exception(self):
    with self.assertRaises(ValueError):
      self.assertRaises(layers.upscale(tf.constant([]), -1))

  def test_upscale_4d_images_returns_upscaled_images(self):
    x_np = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]], dtype=np.float32)
    with self.test_session(use_gpu=True) as sess:
      x1_np, x2_np = sess.run(
          [layers.upscale(tf.constant(x_np), n) for n in [1, 2]])

    expected2_np = [[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
                    [[[4, 5, 6], [4, 5, 6]], [[4, 5, 6], [4, 5, 6]]]]

    self.assertNDArrayNear(x1_np, x_np, 1.0e-5)
    self.assertNDArrayNear(x2_np, expected2_np, 1.0e-5)

  def test_minibatch_mean_stddev_4d_images_returns_scalar(self):
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    with self.test_session(use_gpu=True) as sess:
      output_np = sess.run(layers.minibatch_mean_stddev(x))

    self.assertAlmostEqual(output_np, 3.0416667, 5)

  def test_scalar_concat_invalid_input_throws_exception(self):
    with self.assertRaises(ValueError):
      layers.scalar_concat(tf.constant(1.2), 2.0)

  def test_scalar_concat_4d_images_and_scalar(self):
    x = tf.constant(
        [[[[1, 2], [4, 5]], [[7, 8], [10, 11]]], [[[0, 0], [-1, -2]],
                                                  [[1, -2], [2, 5]]]],
        dtype=tf.float32)
    with self.test_session(use_gpu=True) as sess:
      output_np = sess.run(layers.scalar_concat(x, 7))

    expected_np = [[[[1, 2, 7], [4, 5, 7]], [[7, 8, 7], [10, 11, 7]]],
                   [[[0, 0, 7], [-1, -2, 7]], [[1, -2, 7], [2, 5, 7]]]]

    self.assertNDArrayNear(output_np, expected_np, 1.0e-5)

  def test_he_initializer_scale_slope_linear(self):
    self.assertAlmostEqual(
        layers.he_initializer_scale([3, 4, 5, 6], 1.0), 0.1290994, 5)

  def test_he_initializer_scale_slope_relu(self):
    self.assertAlmostEqual(
        layers.he_initializer_scale([3, 4, 5, 6], 0.0), 0.1825742, 5)

  @mock.patch.object(tf, 'random_normal_initializer', autospec=True)
  @mock.patch.object(tf, 'zeros_initializer', autospec=True)
  def test_custom_layer_impl_with_weight_scaling(
      self, mock_zeros_initializer, mock_random_normal_initializer):
    mock_zeros_initializer.return_value = tf.constant_initializer(1.0)
    mock_random_normal_initializer.return_value = tf.constant_initializer(3.0)
    output = layers._custom_layer_impl(
        apply_kernel=dummy_apply_kernel,
        kernel_shape=(25, 6),
        bias_shape=(),
        activation=lambda x: 2.0 * x,
        he_initializer_slope=1.0,
        use_weight_scaling=True)
    mock_zeros_initializer.assert_called_once_with()
    mock_random_normal_initializer.assert_called_once_with(stddev=1.0)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)

    self.assertAlmostEqual(output_np, 182.6, 3)

  @mock.patch.object(tf, 'random_normal_initializer', autospec=True)
  @mock.patch.object(tf, 'zeros_initializer', autospec=True)
  def test_custom_layer_impl_no_weight_scaling(self, mock_zeros_initializer,
                                               mock_random_normal_initializer):
    mock_zeros_initializer.return_value = tf.constant_initializer(1.0)
    mock_random_normal_initializer.return_value = tf.constant_initializer(3.0)
    output = layers._custom_layer_impl(
        apply_kernel=dummy_apply_kernel,
        kernel_shape=(25, 6),
        bias_shape=(),
        activation=lambda x: 2.0 * x,
        he_initializer_slope=1.0,
        use_weight_scaling=False)
    mock_zeros_initializer.assert_called_once_with()
    mock_random_normal_initializer.assert_called_once_with(stddev=0.2)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)

    self.assertAlmostEqual(output_np, 905.0, 3)

  @mock.patch.object(tf.layers, 'conv2d', autospec=True)
  def test_custom_conv2d_passes_conv2d_options(self, mock_conv2d):
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    layers.custom_conv2d(x, 1, 2)
    mock_conv2d.assert_called_once_with(
        x,
        filters=1,
        kernel_size=[2, 2],
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_initializer=mock.ANY)

  @mock.patch.object(layers, '_custom_layer_impl', autospec=True)
  def test_custom_conv2d_passes_custom_layer_options(self,
                                                     mock_custom_layer_impl):
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    layers.custom_conv2d(x, 1, 2)
    mock_custom_layer_impl.assert_called_once_with(
        mock.ANY,
        kernel_shape=[2, 2, 3, 1],
        bias_shape=(1,),
        activation=None,
        he_initializer_slope=1.0,
        use_weight_scaling=True)

  @mock.patch.object(tf, 'random_normal_initializer', autospec=True)
  @mock.patch.object(tf, 'zeros_initializer', autospec=True)
  def test_custom_conv2d_scalar_kernel_size(self, mock_zeros_initializer,
                                            mock_random_normal_initializer):
    mock_zeros_initializer.return_value = tf.constant_initializer(1.0)
    mock_random_normal_initializer.return_value = tf.constant_initializer(3.0)
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    output = layers.custom_conv2d(x, 1, 2)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)

    expected_np = [[[[68.54998016], [42.56921768]], [[50.36344528],
                                                     [29.57883835]]],
                   [[[5.33012676], [4.46410179]], [[10.52627945],
                                                   [9.66025352]]]]
    self.assertNDArrayNear(output_np, expected_np, 1.0e-5)

  @mock.patch.object(tf, 'random_normal_initializer', autospec=True)
  @mock.patch.object(tf, 'zeros_initializer', autospec=True)
  def test_custom_conv2d_list_kernel_size(self, mock_zeros_initializer,
                                          mock_random_normal_initializer):
    mock_zeros_initializer.return_value = tf.constant_initializer(1.0)
    mock_random_normal_initializer.return_value = tf.constant_initializer(3.0)
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    output = layers.custom_conv2d(x, 1, [2, 3])
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)

    expected_np = [[
        [[56.15432739], [56.15432739]],
        [[41.30508804], [41.30508804]],
    ], [[[4.53553391], [4.53553391]], [[8.7781744], [8.7781744]]]]
    self.assertNDArrayNear(output_np, expected_np, 1.0e-5)

  @mock.patch.object(layers, '_custom_layer_impl', autospec=True)
  def test_custom_dense_passes_custom_layer_options(self,
                                                    mock_custom_layer_impl):
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    layers.custom_dense(x, 3)
    mock_custom_layer_impl.assert_called_once_with(
        mock.ANY,
        kernel_shape=(12, 3),
        bias_shape=(3,),
        activation=None,
        he_initializer_slope=1.0,
        use_weight_scaling=True)

  @mock.patch.object(tf, 'random_normal_initializer', autospec=True)
  @mock.patch.object(tf, 'zeros_initializer', autospec=True)
  def test_custom_dense_output_is_correct(self, mock_zeros_initializer,
                                          mock_random_normal_initializer):
    mock_zeros_initializer.return_value = tf.constant_initializer(1.0)
    mock_random_normal_initializer.return_value = tf.constant_initializer(3.0)
    x = tf.constant(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
         [[[0, 0, 0], [-1, -2, -3]], [[1, -2, 2], [2, 5, 3]]]],
        dtype=tf.float32)
    output = layers.custom_dense(x, 3)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)

    expected_np = [[68.54998016, 68.54998016, 68.54998016],
                   [5.33012676, 5.33012676, 5.33012676]]
    self.assertNDArrayNear(output_np, expected_np, 1.0e-5)


if __name__ == '__main__':
  tf.test.main()
