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
"""Trains a generator on CIFAR data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
from absl import logging
import tensorflow as tf

import data_provider
import networks


tfgan = tf.contrib.gan

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', '/tmp/cifar/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 1000000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_boolean(
    'conditional', False,
    'If `True`, set up a conditional GAN. If False, it is unconditional.')

# Sync replicas flags.
flags.DEFINE_boolean(
    'use_sync_replicas', True,
    'If `True`, use sync replicas. Otherwise use async.')

flags.DEFINE_integer(
    'worker_replicas', 10,
    'The number of gradients to collect before updating params. Only used '
    'with sync replicas.')

flags.DEFINE_integer(
    'backup_workers', 1,
    'Number of workers to be kept as backup in the sync replicas case.')


FLAGS = flags.FLAGS


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        images, one_hot_labels, _, _ = data_provider.provide_data(
            FLAGS.batch_size, FLAGS.dataset_dir)

    # Define the GANModel tuple.
    noise = tf.random_normal([FLAGS.batch_size, 64])
    if FLAGS.conditional:
      generator_fn = networks.conditional_generator
      discriminator_fn = networks.conditional_discriminator
      generator_inputs = (noise, one_hot_labels)
    else:
      generator_fn = networks.generator
      discriminator_fn = networks.discriminator
      generator_inputs = noise
    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data=images,
        generator_inputs=generator_inputs)
    tfgan.eval.add_gan_model_image_summaries(gan_model)

    # Get the GANLoss tuple. Use the selected GAN loss functions.
    # (joelshor): Put this block in `with tf.name_scope('loss'):` when
    # cl/171610946 goes into the opensource release.
    gan_loss = tfgan.gan_loss(gan_model,
                              gradient_penalty_weight=1.0,
                              add_summaries=True)

    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    with tf.name_scope('train'):
      gen_lr, dis_lr = _learning_rate()
      gen_opt, dis_opt = _optimizer(gen_lr, dis_lr, FLAGS.use_sync_replicas)
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

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    sync_hooks = ([gen_opt.make_session_run_hook(FLAGS.task == 0),
                   dis_opt.make_session_run_hook(FLAGS.task == 0)]
                  if FLAGS.use_sync_replicas else [])
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0: return
    tfgan.gan_train(
        train_ops,
        hooks=(
            [tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
             tf.train.LoggingTensorHook([status_message], every_n_iter=10)] +
            sync_hooks),
        logdir=FLAGS.train_log_dir,
        master=FLAGS.master,
        is_chief=FLAGS.task == 0)


def _learning_rate():
  generator_lr = tf.train.exponential_decay(
      learning_rate=0.0001,
      global_step=tf.train.get_or_create_global_step(),
      decay_steps=100000,
      decay_rate=0.9,
      staircase=True)
  discriminator_lr = 0.001
  return generator_lr, discriminator_lr


def _optimizer(gen_lr, dis_lr, use_sync_replicas):
  """Get an optimizer, that's optionally synchronous."""
  generator_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
  discriminator_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)

  def _make_sync(opt):
    return tf.train.SyncReplicasOptimizer(
        opt,
        replicas_to_aggregate=FLAGS.worker_replicas-FLAGS.backup_workers,
        total_num_replicas=FLAGS.worker_replicas)
  if use_sync_replicas:
    generator_opt = _make_sync(generator_opt)
    discriminator_opt = _make_sync(discriminator_opt)

  return generator_opt, discriminator_opt


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run()

