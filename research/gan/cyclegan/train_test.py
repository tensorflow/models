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
"""Tests for cyclegan.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import numpy as np
import tensorflow as tf

import train

FLAGS = flags.FLAGS
mock = tf.test.mock
tfgan = tf.contrib.gan


def _test_generator(input_images):
  """Simple generator function."""
  return input_images * tf.get_variable('dummy_g', initializer=2.0)


def _test_discriminator(image_batch, unused_conditioning=None):
  """Simple discriminator function."""
  return tf.contrib.layers.flatten(
      image_batch * tf.get_variable('dummy_d', initializer=2.0))


train.networks.generator = _test_generator
train.networks.discriminator = _test_discriminator


class TrainTest(tf.test.TestCase):

  @mock.patch.object(tfgan, 'eval', autospec=True)
  def test_define_model(self, mock_eval):
    FLAGS.batch_size = 2
    images_shape = [FLAGS.batch_size, 4, 4, 3]
    images_x_np = np.zeros(shape=images_shape)
    images_y_np = np.zeros(shape=images_shape)
    images_x = tf.constant(images_x_np, dtype=tf.float32)
    images_y = tf.constant(images_y_np, dtype=tf.float32)

    cyclegan_model = train._define_model(images_x, images_y)
    self.assertIsInstance(cyclegan_model, tfgan.CycleGANModel)
    self.assertShapeEqual(images_x_np, cyclegan_model.reconstructed_x)
    self.assertShapeEqual(images_y_np, cyclegan_model.reconstructed_y)

  @mock.patch.object(train.networks, 'generator', autospec=True)
  @mock.patch.object(train.networks, 'discriminator', autospec=True)
  @mock.patch.object(
      tf.train, 'get_or_create_global_step', autospec=True)
  def test_get_lr(self, mock_get_or_create_global_step,
                  unused_mock_discriminator, unused_mock_generator):
    FLAGS.max_number_of_steps = 10
    base_lr = 0.01
    with self.test_session(use_gpu=True) as sess:
      mock_get_or_create_global_step.return_value = tf.constant(2)
      lr_step2 = sess.run(train._get_lr(base_lr))
      mock_get_or_create_global_step.return_value = tf.constant(9)
      lr_step9 = sess.run(train._get_lr(base_lr))

    self.assertAlmostEqual(base_lr, lr_step2)
    self.assertAlmostEqual(base_lr * 0.2, lr_step9)

  @mock.patch.object(tf.train, 'AdamOptimizer', autospec=True)
  def test_get_optimizer(self, mock_adam_optimizer):
    gen_lr, dis_lr = 0.1, 0.01
    train._get_optimizer(gen_lr=gen_lr, dis_lr=dis_lr)
    mock_adam_optimizer.assert_has_calls([
        mock.call(gen_lr, beta1=mock.ANY, use_locking=True),
        mock.call(dis_lr, beta1=mock.ANY, use_locking=True)
    ])

  @mock.patch.object(tf.summary, 'scalar', autospec=True)
  def test_define_train_ops(self, mock_summary_scalar):
    FLAGS.batch_size = 2
    FLAGS.generator_lr = 0.1
    FLAGS.discriminator_lr = 0.01

    images_shape = [FLAGS.batch_size, 4, 4, 3]
    images_x = tf.zeros(images_shape, dtype=tf.float32)
    images_y = tf.zeros(images_shape, dtype=tf.float32)

    cyclegan_model = train._define_model(images_x, images_y)
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model, cycle_consistency_loss_weight=10.0)
    train_ops = train._define_train_ops(cyclegan_model, cyclegan_loss)

    self.assertIsInstance(train_ops, tfgan.GANTrainOps)
    mock_summary_scalar.assert_has_calls([
        mock.call('generator_lr', mock.ANY),
        mock.call('discriminator_lr', mock.ANY)
    ])

  @mock.patch.object(tf, 'gfile', autospec=True)
  @mock.patch.object(train, 'data_provider', autospec=True)
  @mock.patch.object(train, '_define_model', autospec=True)
  @mock.patch.object(tfgan, 'cyclegan_loss', autospec=True)
  @mock.patch.object(train, '_define_train_ops', autospec=True)
  @mock.patch.object(tfgan, 'gan_train', autospec=True)
  def test_main(self, mock_gan_train, mock_define_train_ops, mock_cyclegan_loss,
                mock_define_model, mock_data_provider, mock_gfile):
    FLAGS.image_set_x_file_pattern = '/tmp/x/*.jpg'
    FLAGS.image_set_y_file_pattern = '/tmp/y/*.jpg'
    FLAGS.batch_size = 3
    FLAGS.patch_size = 8
    FLAGS.generator_lr = 0.02
    FLAGS.discriminator_lr = 0.3
    FLAGS.train_log_dir = '/tmp/foo'
    FLAGS.master = 'master'
    FLAGS.task = 0
    FLAGS.cycle_consistency_loss_weight = 2.0
    FLAGS.max_number_of_steps = 1

    mock_data_provider.provide_custom_datasets.return_value = (tf.zeros(
        [1, 2], dtype=tf.float32), tf.zeros([1, 2], dtype=tf.float32))

    train.main(None)
    mock_data_provider.provide_custom_datasets.assert_called_once_with(
        ['/tmp/x/*.jpg', '/tmp/y/*.jpg'], batch_size=3, patch_size=8)
    mock_define_model.assert_called_once_with(mock.ANY, mock.ANY)
    mock_cyclegan_loss.assert_called_once_with(
        mock_define_model.return_value,
        cycle_consistency_loss_weight=2.0,
        tensor_pool_fn=mock.ANY)
    mock_define_train_ops.assert_called_once_with(
        mock_define_model.return_value, mock_cyclegan_loss.return_value)
    mock_gan_train.assert_called_once_with(
        mock_define_train_ops.return_value,
        '/tmp/foo',
        get_hooks_fn=mock.ANY,
        hooks=mock.ANY,
        master='master',
        is_chief=True)


if __name__ == '__main__':
  tf.test.main()
