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
import networks


def _get_grad_norm(ys, xs):
  """Compute 2-norm of dys / dxs."""
  return tf.sqrt(
      tf.add_n([tf.reduce_sum(tf.square(g)) for g in tf.gradients(ys, xs)]))


def _num_filters_stub(block_id):
  return networks.num_filters(block_id, 8, 1, 8)


class NetworksTest(tf.test.TestCase):

  def test_resolution_schedule_correct(self):
    rs = networks.ResolutionSchedule(
        start_resolutions=[5, 3], scale_base=2, num_resolutions=3)
    self.assertEqual(rs.start_resolutions, (5, 3))
    self.assertEqual(rs.scale_base, 2)
    self.assertEqual(rs.num_resolutions, 3)
    self.assertEqual(rs.final_resolutions, (20, 12))
    self.assertEqual(rs.scale_factor(1), 4)
    self.assertEqual(rs.scale_factor(2), 2)
    self.assertEqual(rs.scale_factor(3), 1)
    with self.assertRaises(ValueError):
      rs.scale_factor(0)
    with self.assertRaises(ValueError):
      rs.scale_factor(4)

  def test_block_name(self):
    self.assertEqual(networks.block_name(10), 'progressive_gan_block_10')

  def test_min_total_num_images(self):
    self.assertEqual(networks.min_total_num_images(7, 8, 4), 52)

  def test_compute_progress(self):
    current_image_id_ph = tf.placeholder(tf.int32, [])
    progress = networks.compute_progress(
        current_image_id_ph,
        stable_stage_num_images=7,
        transition_stage_num_images=8,
        num_blocks=2)
    with self.test_session(use_gpu=True) as sess:
      progress_output = [
          sess.run(progress, feed_dict={current_image_id_ph: current_image_id})
          for current_image_id in [0, 3, 6, 7, 8, 10, 15, 29, 100]
      ]
    self.assertArrayNear(progress_output,
                         [0.0, 0.0, 0.0, 0.0, 0.125, 0.375, 1.0, 1.0, 1.0],
                         1.0e-6)

  def test_generator_alpha(self):
    with self.test_session(use_gpu=True) as sess:
      alpha_fixed_block_id = [
          sess.run(
              networks._generator_alpha(2, tf.constant(progress, tf.float32)))
          for progress in [0, 0.2, 1, 1.2, 2, 2.2, 3]
      ]
      alpha_fixed_progress = [
          sess.run(
              networks._generator_alpha(block_id, tf.constant(1.2, tf.float32)))
          for block_id in range(1, 5)
      ]

    self.assertArrayNear(alpha_fixed_block_id, [0, 0.2, 1, 0.8, 0, 0, 0],
                         1.0e-6)
    self.assertArrayNear(alpha_fixed_progress, [0, 0.8, 0.2, 0], 1.0e-6)

  def test_discriminator_alpha(self):
    with self.test_session(use_gpu=True) as sess:
      alpha_fixed_block_id = [
          sess.run(
              networks._discriminator_alpha(2, tf.constant(
                  progress, tf.float32)))
          for progress in [0, 0.2, 1, 1.2, 2, 2.2, 3]
      ]
      alpha_fixed_progress = [
          sess.run(
              networks._discriminator_alpha(block_id,
                                            tf.constant(1.2, tf.float32)))
          for block_id in range(1, 5)
      ]

    self.assertArrayNear(alpha_fixed_block_id, [1, 1, 1, 0.8, 0, 0, 0], 1.0e-6)
    self.assertArrayNear(alpha_fixed_progress, [0, 0.8, 1, 1], 1.0e-6)

  def test_blend_images_in_stable_stage(self):
    x_np = np.random.normal(size=[2, 8, 8, 3])
    x = tf.constant(x_np, tf.float32)
    x_blend = networks.blend_images(
        x,
        progress=tf.constant(0.0),
        resolution_schedule=networks.ResolutionSchedule(
            scale_base=2, num_resolutions=2),
        num_blocks=2)
    with self.test_session(use_gpu=True) as sess:
      x_blend_np = sess.run(x_blend)
      x_blend_expected_np = sess.run(layers.upscale(layers.downscale(x, 2), 2))
    self.assertNDArrayNear(x_blend_np, x_blend_expected_np, 1.0e-6)

  def test_blend_images_in_transition_stage(self):
    x_np = np.random.normal(size=[2, 8, 8, 3])
    x = tf.constant(x_np, tf.float32)
    x_blend = networks.blend_images(
        x,
        tf.constant(0.2),
        resolution_schedule=networks.ResolutionSchedule(
            scale_base=2, num_resolutions=2),
        num_blocks=2)
    with self.test_session(use_gpu=True) as sess:
      x_blend_np = sess.run(x_blend)
      x_blend_expected_np = 0.8 * sess.run(
          layers.upscale(layers.downscale(x, 2), 2)) + 0.2 * x_np
    self.assertNDArrayNear(x_blend_np, x_blend_expected_np, 1.0e-6)

  def test_num_filters(self):
    self.assertEqual(networks.num_filters(1, 4096, 1, 256), 256)
    self.assertEqual(networks.num_filters(5, 4096, 1, 256), 128)

  def test_generator_grad_norm_progress(self):
    stable_stage_num_images = 2
    transition_stage_num_images = 3

    current_image_id_ph = tf.placeholder(tf.int32, [])
    progress = networks.compute_progress(
        current_image_id_ph,
        stable_stage_num_images,
        transition_stage_num_images,
        num_blocks=3)
    z = tf.random_normal([2, 10], dtype=tf.float32)
    x, _ = networks.generator(
        z, progress, _num_filters_stub,
        networks.ResolutionSchedule(
            start_resolutions=(4, 4), scale_base=2, num_resolutions=3))
    fake_loss = tf.reduce_sum(tf.square(x))
    grad_norms = [
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_1/.*')),
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_2/.*')),
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_3/.*'))
    ]

    grad_norms_output = None
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      x1_np = sess.run(x, feed_dict={current_image_id_ph: 0.12})
      x2_np = sess.run(x, feed_dict={current_image_id_ph: 1.8})
      grad_norms_output = np.array([
          sess.run(grad_norms, feed_dict={current_image_id_ph: i})
          for i in range(15)  # total num of images
      ])

    self.assertEqual((2, 16, 16, 3), x1_np.shape)
    self.assertEqual((2, 16, 16, 3), x2_np.shape)
    # The gradient of block_1 is always on.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 0] > 0), 0,
        'gradient norms {} for block 1 is not always on'.format(
            grad_norms_output[:, 0]))
    # The gradient of block_2 is on after 1 stable stage.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 1] > 0), 3,
        'gradient norms {} for block 2 is not on at step 3'.format(
            grad_norms_output[:, 1]))
    # The gradient of block_3 is on after 2 stable stage + 1 transition stage.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 2] > 0), 8,
        'gradient norms {} for block 3 is not on at step 8'.format(
            grad_norms_output[:, 2]))

  def test_discriminator_grad_norm_progress(self):
    stable_stage_num_images = 2
    transition_stage_num_images = 3

    current_image_id_ph = tf.placeholder(tf.int32, [])
    progress = networks.compute_progress(
        current_image_id_ph,
        stable_stage_num_images,
        transition_stage_num_images,
        num_blocks=3)
    x = tf.random_normal([2, 16, 16, 3])
    logits, _ = networks.discriminator(
        x, progress, _num_filters_stub,
        networks.ResolutionSchedule(
            start_resolutions=(4, 4), scale_base=2, num_resolutions=3))
    fake_loss = tf.reduce_sum(tf.square(logits))
    grad_norms = [
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_1/.*')),
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_2/.*')),
        _get_grad_norm(
            fake_loss, tf.trainable_variables('.*/progressive_gan_block_3/.*'))
    ]

    grad_norms_output = None
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      grad_norms_output = np.array([
          sess.run(grad_norms, feed_dict={current_image_id_ph: i})
          for i in range(15)  # total num of images
      ])

    # The gradient of block_1 is always on.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 0] > 0), 0,
        'gradient norms {} for block 1 is not always on'.format(
            grad_norms_output[:, 0]))
    # The gradient of block_2 is on after 1 stable stage.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 1] > 0), 3,
        'gradient norms {} for block 2 is not on at step 3'.format(
            grad_norms_output[:, 1]))
    # The gradient of block_3 is on after 2 stable stage + 1 transition stage.
    self.assertEqual(
        np.argmax(grad_norms_output[:, 2] > 0), 8,
        'gradient norms {} for block 3 is not on at step 8'.format(
            grad_norms_output[:, 2]))


if __name__ == '__main__':
  tf.test.main()
