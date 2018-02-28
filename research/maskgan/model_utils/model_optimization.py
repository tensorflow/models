# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Model optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def create_dis_pretrain_op(hparams, dis_loss, global_step):
  """Create a train op for pretraining."""
  with tf.name_scope('pretrain_generator'):
    optimizer = tf.train.AdamOptimizer(hparams.dis_pretrain_learning_rate)
    dis_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('dis')
    ]
    if FLAGS.dis_update_share_embedding and FLAGS.dis_share_embedding:
      shared_embedding = [
          v for v in tf.trainable_variables()
          if v.op.name == 'gen/decoder/rnn/embedding'
      ][0]
      dis_vars.append(shared_embedding)
    dis_grads = tf.gradients(dis_loss, dis_vars)
    dis_grads_clipped, _ = tf.clip_by_global_norm(dis_grads,
                                                  FLAGS.grad_clipping)
    dis_pretrain_op = optimizer.apply_gradients(
        zip(dis_grads_clipped, dis_vars), global_step=global_step)
    return dis_pretrain_op


def create_gen_pretrain_op(hparams, cross_entropy_loss, global_step):
  """Create a train op for pretraining."""
  with tf.name_scope('pretrain_generator'):
    optimizer = tf.train.AdamOptimizer(hparams.gen_pretrain_learning_rate)
    gen_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('gen')
    ]
    gen_grads = tf.gradients(cross_entropy_loss, gen_vars)
    gen_grads_clipped, _ = tf.clip_by_global_norm(gen_grads,
                                                  FLAGS.grad_clipping)
    gen_pretrain_op = optimizer.apply_gradients(
        zip(gen_grads_clipped, gen_vars), global_step=global_step)
    return gen_pretrain_op


def create_gen_train_op(hparams, learning_rate, gen_loss, global_step, mode):
  """Create Generator train op."""
  del hparams
  with tf.name_scope('train_generator'):
    if FLAGS.generator_optimizer == 'sgd':
      gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.generator_optimizer == 'adam':
      gen_optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
      raise NotImplementedError
    gen_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('gen')
    ]
    print('Optimizing Generator vars.')
    for v in gen_vars:
      print(v)
    if mode == 'MINIMIZE':
      gen_grads = tf.gradients(gen_loss, gen_vars)
    elif mode == 'MAXIMIZE':
      gen_grads = tf.gradients(-gen_loss, gen_vars)
    else:
      raise ValueError("Must be one of 'MINIMIZE' or 'MAXIMIZE'")
    gen_grads_clipped, _ = tf.clip_by_global_norm(gen_grads,
                                                  FLAGS.grad_clipping)
    gen_train_op = gen_optimizer.apply_gradients(
        zip(gen_grads_clipped, gen_vars), global_step=global_step)
    return gen_train_op, gen_grads_clipped, gen_vars


def create_reinforce_gen_train_op(hparams, learning_rate, final_gen_reward,
                                  averages_op, global_step):
  """Create the Generator train_op when using REINFORCE.

  Args:
    hparams:  MaskGAN hyperparameters.
    learning_rate:  tf.Variable scalar learning rate.
    final_gen_objective:  Scalar final REINFORCE objective for the sequence.
    averages_op:  ExponentialMovingAverage apply average op to
      maintain the baseline.
    global_step:  global_step tf.Variable.

  Returns:
    gen_train_op: Generator training op.
  """
  del hparams
  with tf.name_scope('train_generator'):
    if FLAGS.generator_optimizer == 'sgd':
      gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.generator_optimizer == 'adam':
      gen_optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
      raise NotImplementedError
    gen_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('gen')
    ]
    print('\nOptimizing Generator vars:')
    for v in gen_vars:
      print(v)

    # Maximize reward.
    gen_grads = tf.gradients(-final_gen_reward, gen_vars)
    gen_grads_clipped, _ = tf.clip_by_global_norm(gen_grads,
                                                  FLAGS.grad_clipping)
    maximize_op = gen_optimizer.apply_gradients(
        zip(gen_grads_clipped, gen_vars), global_step=global_step)

    # Group maintain averages op.
    if averages_op:
      gen_train_op = tf.group(maximize_op, averages_op)
    else:
      gen_train_op = maximize_op

    return [gen_train_op, gen_grads, gen_vars]


def create_dis_train_op(hparams, dis_loss, global_step):
  """Create Discriminator train op."""
  with tf.name_scope('train_discriminator'):
    dis_optimizer = tf.train.AdamOptimizer(hparams.dis_learning_rate)
    dis_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('dis')
    ]
    if FLAGS.dis_update_share_embedding and FLAGS.dis_share_embedding:
      shared_embedding = [
          v for v in tf.trainable_variables()
          if v.op.name == 'gen/decoder/rnn/embedding'
      ][0]
      dis_vars.append(shared_embedding)
    print('\nOptimizing Discriminator vars:')
    for v in dis_vars:
      print(v)
    dis_grads = tf.gradients(dis_loss, dis_vars)
    dis_grads_clipped, _ = tf.clip_by_global_norm(dis_grads,
                                                  FLAGS.grad_clipping)
    dis_train_op = dis_optimizer.apply_gradients(
        zip(dis_grads_clipped, dis_vars), global_step=global_step)
    return dis_train_op, dis_grads_clipped, dis_vars


def create_critic_train_op(hparams, critic_loss, global_step):
  """Create Discriminator train op."""
  with tf.name_scope('train_critic'):
    critic_optimizer = tf.train.AdamOptimizer(hparams.critic_learning_rate)
    output_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('critic')
    ]

    if FLAGS.critic_update_dis_vars:
      if FLAGS.discriminator_model == 'bidirectional_vd':
        critic_vars = [
            v for v in tf.trainable_variables()
            if v.op.name.startswith('dis/rnn')
        ]
      elif FLAGS.discriminator_model == 'seq2seq_vd':
        critic_vars = [
            v for v in tf.trainable_variables()
            if v.op.name.startswith('dis/decoder/rnn/multi_rnn_cell')
        ]
      critic_vars.extend(output_vars)
    else:
      critic_vars = output_vars
    print('\nOptimizing Critic vars:')
    for v in critic_vars:
      print(v)
    critic_grads = tf.gradients(critic_loss, critic_vars)
    critic_grads_clipped, _ = tf.clip_by_global_norm(critic_grads,
                                                     FLAGS.grad_clipping)
    critic_train_op = critic_optimizer.apply_gradients(
        zip(critic_grads_clipped, critic_vars), global_step=global_step)
    return critic_train_op, critic_grads_clipped, critic_vars
