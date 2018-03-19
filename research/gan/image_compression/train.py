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
# pylint:disable=line-too-long
"""Trains an image compression network with an adversarial loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import data_provider
import networks
import summaries


tfgan = tf.contrib.gan
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer('patch_size', 32, 'The size of the patches to train on.')

flags.DEFINE_integer('bits_per_patch', 1230,
                     'The number of bits to produce per patch.')

flags.DEFINE_integer('model_depth', 64,
                     'Number of filters for compression model')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', '/tmp/compression/',
                    'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 1e-5,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 1e-6,
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
    'weight_factor', 10000.0,
    'How much to weight the adversarial loss relative to pixel loss.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # Put input pipeline on CPU to reserve GPU for training.
    with tf.name_scope('inputs'), tf.device('/cpu:0'):
      images = data_provider.provide_data(
          'train', FLAGS.batch_size, dataset_dir=FLAGS.dataset_dir,
          patch_size=FLAGS.patch_size)

    # Manually define a GANModel tuple. This is useful when we have custom
    # code to track variables. Note that we could replace all of this with a
    # call to `tfgan.gan_model`, but we don't in order to demonstrate some of
    # TFGAN's flexibility.
    with tf.variable_scope('generator') as gen_scope:
      reconstructions, _, prebinary = networks.compression_model(
          images,
          num_bits=FLAGS.bits_per_patch,
          depth=FLAGS.model_depth)
    gan_model = _get_gan_model(
        generator_inputs=images,
        generated_data=reconstructions,
        real_data=images,
        generator_scope=gen_scope)
    summaries.add_reconstruction_summaries(images, reconstructions, prebinary)
    tfgan.eval.add_gan_model_summaries(gan_model)

    # Define the GANLoss tuple using standard library functions.
    with tf.name_scope('loss'):
      gan_loss = tfgan.gan_loss(
          gan_model,
          generator_loss_fn=tfgan.losses.least_squares_generator_loss,
          discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss,
          add_summaries=FLAGS.weight_factor > 0)

      # Define the standard pixel loss.
      l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data,
                              ord=1)

      # Modify the loss tuple to include the pixel loss. Add summaries as well.
      gan_loss = tfgan.losses.combine_adversarial_loss(
          gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    with tf.name_scope('train_ops'):
      gen_lr, dis_lr = _lr(FLAGS.generator_lr, FLAGS.discriminator_lr)
      gen_opt, dis_opt = _optimizer(gen_lr, dis_lr)
      train_ops = tfgan.gan_train_ops(
          gan_model,
          gan_loss,
          generator_optimizer=gen_opt,
          discriminator_optimizer=dis_opt,
          summarize_gradients=True,
          colocate_gradients_with_ops=True,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      tf.summary.scalar('generator_lr', gen_lr)
      tf.summary.scalar('discriminator_lr', dis_lr)

    # Determine the number of generator vs discriminator steps.
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
        tfgan.get_sequential_train_hooks(train_steps),
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        master=FLAGS.master,
        is_chief=FLAGS.task == 0)


def _optimizer(gen_lr, dis_lr):
  # First is generator optimizer, second is discriminator.
  adam_kwargs = {
      'epsilon': 1e-8,
      'beta1': 0.5,
  }
  return (tf.train.AdamOptimizer(gen_lr, **adam_kwargs),
          tf.train.AdamOptimizer(dis_lr, **adam_kwargs))


def _lr(gen_lr_base, dis_lr_base):
  """Return the generator and discriminator learning rates."""
  gen_lr_kwargs = {
      'decay_steps': 60000,
      'decay_rate': 0.9,
      'staircase': True,
  }
  gen_lr = tf.train.exponential_decay(
      learning_rate=gen_lr_base,
      global_step=tf.train.get_or_create_global_step(),
      **gen_lr_kwargs)
  dis_lr = dis_lr_base

  return gen_lr, dis_lr


def _get_gan_model(generator_inputs, generated_data, real_data,
                   generator_scope):
  """Manually construct and return a GANModel tuple."""
  generator_vars = tf.contrib.framework.get_trainable_variables(generator_scope)

  discriminator_fn = networks.discriminator
  with tf.variable_scope('discriminator') as dis_scope:
    discriminator_gen_outputs = discriminator_fn(generated_data)
  with tf.variable_scope(dis_scope, reuse=True):
    discriminator_real_outputs = discriminator_fn(real_data)
  discriminator_vars = tf.contrib.framework.get_trainable_variables(
      dis_scope)

  # Manually construct GANModel tuple.
  gan_model = tfgan.GANModel(
      generator_inputs=generator_inputs,
      generated_data=generated_data,
      generator_variables=generator_vars,
      generator_scope=generator_scope,
      generator_fn=None,  # not necessary
      real_data=real_data,
      discriminator_real_outputs=discriminator_real_outputs,
      discriminator_gen_outputs=discriminator_gen_outputs,
      discriminator_variables=discriminator_vars,
      discriminator_scope=dis_scope,
      discriminator_fn=discriminator_fn)

  return gan_model


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

