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
"""Trains a CycleGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import tensorflow as tf

import data_provider
import networks

tfgan = tf.contrib.gan


flags.DEFINE_string('image_set_x_file_pattern', None,
                    'File pattern of images in image set X')

flags.DEFINE_string('image_set_y_file_pattern', None,
                    'File pattern of images in image set Y')

flags.DEFINE_integer('batch_size', 1, 'The number of images in each batch.')

flags.DEFINE_integer('patch_size', 64, 'The patch size of images.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', '/tmp/cyclegan/',
                    'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 0.0002,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.0001,
                   'The discriminator learning rate.')

flags.DEFINE_integer('max_number_of_steps', 500000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_float('cycle_consistency_loss_weight', 10.0,
                   'The weight of cycle consistency loss')


FLAGS = flags.FLAGS


def _define_model(images_x, images_y):
  """Defines a CycleGAN model that maps between images_x and images_y.

  Args:
    images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
    images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

  Returns:
    A `CycleGANModel` namedtuple.
  """
  cyclegan_model = tfgan.cyclegan_model(
      generator_fn=networks.generator,
      discriminator_fn=networks.discriminator,
      data_x=images_x,
      data_y=images_y)

  # Add summaries for generated images.
  tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

  return cyclegan_model


def _get_lr(base_lr):
  """Returns a learning rate `Tensor`.

  Args:
    base_lr: A scalar float `Tensor` or a Python number.  The base learning
        rate.

  Returns:
    A scalar float `Tensor` of learning rate which equals `base_lr` when the
    global training step is less than FLAGS.max_number_of_steps / 2, afterwards
    it linearly decays to zero.
  """
  global_step = tf.train.get_or_create_global_step()
  lr_constant_steps = FLAGS.max_number_of_steps // 2

  def _lr_decay():
    return tf.train.polynomial_decay(
        learning_rate=base_lr,
        global_step=(global_step - lr_constant_steps),
        decay_steps=(FLAGS.max_number_of_steps - lr_constant_steps),
        end_learning_rate=0.0)

  return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


def _get_optimizer(gen_lr, dis_lr):
  """Returns generator optimizer and discriminator optimizer.

  Args:
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
        rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
        learning rate.

  Returns:
    A tuple of generator optimizer and discriminator optimizer.
  """
  # beta1 follows
  # https://github.com/junyanz/CycleGAN/blob/master/options.lua
  gen_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
  dis_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
  return gen_opt, dis_opt


def _define_train_ops(cyclegan_model, cyclegan_loss):
  """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

  Args:
    cyclegan_model: A `CycleGANModel` namedtuple.
    cyclegan_loss: A `CycleGANLoss` namedtuple containing all losses for
        `cyclegan_model`.

  Returns:
    A `GANTrainOps` namedtuple.
  """
  gen_lr = _get_lr(FLAGS.generator_lr)
  dis_lr = _get_lr(FLAGS.discriminator_lr)
  gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)
  train_ops = tfgan.gan_train_ops(
      cyclegan_model,
      cyclegan_loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      summarize_gradients=True,
      colocate_gradients_with_ops=True,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  tf.summary.scalar('generator_lr', gen_lr)
  tf.summary.scalar('discriminator_lr', dis_lr)
  return train_ops


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    with tf.name_scope('inputs'):
      images_x, images_y = data_provider.provide_custom_datasets(
          [FLAGS.image_set_x_file_pattern, FLAGS.image_set_y_file_pattern],
          batch_size=FLAGS.batch_size,
          patch_size=FLAGS.patch_size)

    # Define CycleGAN model.
    cyclegan_model = _define_model(images_x, images_y)

    # Define CycleGAN loss.
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model,
        cycle_consistency_loss_weight=FLAGS.cycle_consistency_loss_weight,
        tensor_pool_fn=tfgan.features.tensor_pool)

    # Define CycleGAN train ops.
    train_ops = _define_train_ops(cyclegan_model, cyclegan_loss)

    # Training
    train_steps = tfgan.GANTrainSteps(1, 1)
    status_message = tf.string_join(
        [
            'Starting train step: ',
            tf.as_string(tf.train.get_or_create_global_step())
        ],
        name='status_message')
    if not FLAGS.max_number_of_steps:
      return
    tfgan.gan_train(
        train_ops,
        FLAGS.train_log_dir,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
            tf.train.LoggingTensorHook([status_message], every_n_iter=10)
        ],
        master=FLAGS.master,
        is_chief=FLAGS.task == 0)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('image_set_x_file_pattern')
  tf.flags.mark_flag_as_required('image_set_y_file_pattern')
  tf.app.run()
