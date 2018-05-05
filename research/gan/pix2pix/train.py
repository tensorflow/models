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
"""Trains an image-to-image translation network with an adversarial loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import data_provider
import networks

flags = tf.flags
tfgan = tf.contrib.gan


flags.DEFINE_integer('batch_size', 10, 'The number of images in each batch.')

flags.DEFINE_integer('patch_size', 32, 'The size of the patches to train on.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', '/tmp/pix2pix/',
                    'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 0.00001,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.00001,
                   'The discriminator learning rate.')

flags.DEFINE_integer('max_number_of_steps', 2000000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_float(
    'weight_factor', 0.0,
    'How much to weight the adversarial loss relative to pixel loss.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')


FLAGS = flags.FLAGS


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # Get real and distorted images.
    with tf.device('/cpu:0'), tf.name_scope('inputs'):
      real_images = data_provider.provide_data(
          'train', FLAGS.batch_size, dataset_dir=FLAGS.dataset_dir,
          patch_size=FLAGS.patch_size)
    distorted_images = _distort_images(
        real_images, downscale_size=int(FLAGS.patch_size / 2),
        upscale_size=FLAGS.patch_size)

    # Create a GANModel tuple.
    gan_model = tfgan.gan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        real_data=real_images,
        generator_inputs=distorted_images)
    tfgan.eval.add_image_comparison_summaries(
        gan_model, num_comparisons=3, display_diffs=True)
    tfgan.eval.add_gan_model_image_summaries(gan_model, grid_size=3)

    # Define the GANLoss tuple using standard library functions.
    with tf.name_scope('losses'):
      gan_loss = tfgan.gan_loss(
          gan_model,
          generator_loss_fn=tfgan.losses.least_squares_generator_loss,
          discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)

      # Define the standard L1 pixel loss.
      l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data,
                              ord=1) / FLAGS.patch_size ** 2

      # Modify the loss tuple to include the pixel loss. Add summaries as well.
      gan_loss = tfgan.losses.combine_adversarial_loss(
          gan_loss, gan_model, l1_pixel_loss,
          weight_factor=FLAGS.weight_factor)

    with tf.name_scope('train_ops'):
      # Get the GANTrain ops using the custom optimizers and optional
      # discriminator weight clipping.
      gen_lr, dis_lr = _lr(FLAGS.generator_lr, FLAGS.discriminator_lr)
      gen_opt, dis_opt = _optimizer(gen_lr, dis_lr)
      train_ops = tfgan.gan_train_ops(
          gan_model,
          gan_loss,
          generator_optimizer=gen_opt,
          discriminator_optimizer=dis_opt,
          summarize_gradients=True,
          colocate_gradients_with_ops=True,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
          transform_grads_fn=tf.contrib.training.clip_gradient_norms_fn(1e3))
      tf.summary.scalar('generator_lr', gen_lr)
      tf.summary.scalar('discriminator_lr', dis_lr)

    # Use GAN train step function if using adversarial loss, otherwise
    # only train the generator.
    train_steps = tfgan.GANTrainSteps(
        generator_train_steps=1,
        discriminator_train_steps=int(FLAGS.weight_factor > 0))

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0: return
    tfgan.gan_train(
        train_ops,
        FLAGS.train_log_dir,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        master=FLAGS.master,
        is_chief=FLAGS.task == 0)


def _optimizer(gen_lr, dis_lr):
  kwargs = {'beta1': 0.5, 'beta2': 0.999}
  generator_opt = tf.train.AdamOptimizer(gen_lr, **kwargs)
  discriminator_opt = tf.train.AdamOptimizer(dis_lr, **kwargs)
  return generator_opt, discriminator_opt


def _lr(gen_lr_base, dis_lr_base):
  """Return the generator and discriminator learning rates."""
  gen_lr = tf.train.exponential_decay(
      learning_rate=gen_lr_base,
      global_step=tf.train.get_or_create_global_step(),
      decay_steps=100000,
      decay_rate=0.8,
      staircase=True,)
  dis_lr = dis_lr_base

  return gen_lr, dis_lr


def _distort_images(images, downscale_size, upscale_size):
  downscaled = tf.image.resize_area(images, [downscale_size] * 2)
  upscaled = tf.image.resize_area(downscaled, [upscale_size] * 2)
  return upscaled


if __name__ == '__main__':
  tf.app.run()

