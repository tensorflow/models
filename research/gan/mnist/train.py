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
"""Trains a generator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools


import tensorflow as tf

import data_provider
import networks
import util

flags = tf.flags
tfgan = tf.contrib.gan


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string(
    'gan_type', 'unconditional',
    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer(
    'grid_size', 5, 'Grid size for image visualization.')


flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS


def _learning_rate(gan_type):
  # First is generator learning rate, second is discriminator learning rate.
  return {
      'unconditional': (1e-3, 1e-4),
      'conditional': (1e-5, 1e-4),
      'infogan': (0.001, 9e-5),
  }[gan_type]


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.name_scope('inputs'):
    with tf.device('/cpu:0'):
      images, one_hot_labels, _ = data_provider.provide_data(
          'train', FLAGS.batch_size, FLAGS.dataset_dir, num_threads=4)

  # Define the GANModel tuple. Optionally, condition the GAN on the label or
  # use an InfoGAN to learn a latent representation.
  if FLAGS.gan_type == 'unconditional':
    gan_model = tfgan.gan_model(
        generator_fn=networks.unconditional_generator,
        discriminator_fn=networks.unconditional_discriminator,
        real_data=images,
        generator_inputs=tf.random_normal(
            [FLAGS.batch_size, FLAGS.noise_dims]))
  elif FLAGS.gan_type == 'conditional':
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])
    gan_model = tfgan.gan_model(
        generator_fn=networks.conditional_generator,
        discriminator_fn=networks.conditional_discriminator,
        real_data=images,
        generator_inputs=(noise, one_hot_labels))
  elif FLAGS.gan_type == 'infogan':
    cat_dim, cont_dim = 10, 2
    generator_fn = functools.partial(
        networks.infogan_generator, categorical_dim=cat_dim)
    discriminator_fn = functools.partial(
        networks.infogan_discriminator, categorical_dim=cat_dim,
        continuous_dim=cont_dim)
    unstructured_inputs, structured_inputs = util.get_infogan_noise(
        FLAGS.batch_size, cat_dim, cont_dim, FLAGS.noise_dims)
    gan_model = tfgan.infogan_model(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        real_data=images,
        unstructured_generator_inputs=unstructured_inputs,
        structured_generator_inputs=structured_inputs)
  tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

  # Get the GANLoss tuple. You can pass a custom function, use one of the
  # already-implemented losses from the losses library, or use the defaults.
  with tf.name_scope('loss'):
    mutual_information_penalty_weight = (1.0 if FLAGS.gan_type == 'infogan'
                                         else 0.0)
    gan_loss = tfgan.gan_loss(
        gan_model,
        gradient_penalty_weight=1.0,
        mutual_information_penalty_weight=mutual_information_penalty_weight,
        add_summaries=True)
    tfgan.eval.add_regularization_loss_summaries(gan_model)

  # Get the GANTrain ops using custom optimizers.
  with tf.name_scope('train'):
    gen_lr, dis_lr = _learning_rate(FLAGS.gan_type)
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
        summarize_gradients=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  # Run the alternating training loop. Skip it if no steps should be taken
  # (used for graph construction tests).
  status_message = tf.string_join(
      ['Starting train step: ',
       tf.as_string(tf.train.get_or_create_global_step())],
      name='status_message')
  if FLAGS.max_number_of_steps == 0: return
  tfgan.gan_train(
      train_ops,
      hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
             tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
      logdir=FLAGS.train_log_dir,
      get_hooks_fn=tfgan.get_joint_train_hooks())

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
