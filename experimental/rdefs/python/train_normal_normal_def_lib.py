# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Trains a recurrent DEF with gaussian latent variables and gaussian weights.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time



import numpy as np
from scipy.misc import imsave
import tensorflow as tf

from ops import inference
from ops import model_factory
from ops import util

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to write event logs.')
flags.DEFINE_boolean('delete_logdir', True, 'Whether to clear the log dir.')
flags.DEFINE_string('trials_root_dir',
                    '/tmp/logs',
                    'Directory where to write event logs.')
flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 10,
                     'The frequency with which the model is saved, in seconds.')
flags.DEFINE_integer('max_steps', 200000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('print_stats_every', 100, 'print stats every')
flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')
flags.DEFINE_string('trainer', 'supervisor', 'slim/local/supervisor')
flags.DEFINE_integer('samples_to_save', 1, 'number of samples to save')
flags.DEFINE_boolean('check_nans', False, 'add ops to check for nans.')
flags.DEFINE_string('data_path',
                    '/readahead/256M/cns/in-d/home/jaana/binarized_mnist_new',
                    'Where to read the data from.')

FLAGS = flags.FLAGS


def run_training(hparams, train_dir, max_steps, container='',
                 trainer='supervisor', reporter_fn=None):
  """Trains a Gaussian Recurrent DEF.

  Args:
    hparams: A tf.HParams object with hyperparameters for training.
    train_dir: Where to store events files and checkpoints.
    max_steps: Integer number of steps to train.
    container: String specifying container for resource sharing.
    trainer: Train locally by loading an hdf5 file or with Supervisor.
    reporter_fn: Optional reporter function

  Returns:
    sess: Optionally, the session for training.
    vi: Optionally, VariationalInference object that has been trained.
  """
  hps = hparams
  tf.set_random_seed(4235)
  np.random.seed(4234)
  g = tf.Graph()
  if FLAGS.ps_tasks > 0:
    device_fn = tf.train.replica_device_setter(FLAGS.ps_tasks)
  else:
    device_fn = None
  with g.as_default(), g.device(device_fn), tf.container(container):
    if trainer == 'local':
      x_indexes = tf.placeholder(tf.int32, [None])
      x = tf.placeholder(tf.float32,
                         [None, hps.n_timesteps, hps.timestep_dim, 1])
      data_iterator = util.provide_hdf5_data(
          FLAGS.data_path,
          'train',
          hps.n_examples,
          hps.batch_size,
          hps.n_timesteps,
          hps.timestep_dim,
          hps.dataset)
    else:
      x_indexes, x = util.provide_tfrecords_data(
          FLAGS.data_path,
          'train_labeled',
          hps.batch_size,
          hps.n_timesteps,
          hps.timestep_dim)

    data = {'x': x, 'x_indexes': x_indexes}

    model = model_factory.NormalNormalRDEF(
        n_timesteps=hps.n_timesteps,
        batch_size=hps.batch_size,
        p_z_sigma=hps.p_z_sigma,
        p_w_mu_sigma=hps.p_w_mu_sigma,
        fixed_p_z_sigma=hps.fixed_p_z_sigma,
        p_w_sigma_sigma=hps.p_w_sigma_sigma,
        z_dim=hps.z_dim,
        dtype=hps.dtype)

    variational = model_factory.NormalNormalRDEFVariational(
        x_indexes=x_indexes,
        n_examples=hps.n_examples,
        n_timesteps=hps.n_timesteps,
        z_dim=hps.z_dim,
        timestep_dim=hps.timestep_dim,
        init_sigma_q_z=hps.p_z_sigma * hps.init_q_sigma_scale,
        init_sigma_q_w_mu=hps.p_w_mu_sigma * hps.init_q_sigma_scale,
        init_sigma_q_w_sigma=hps.p_w_sigma_sigma * hps.init_q_sigma_scale,
        init_sigma_q_w_0_sigma=hps.p_w_sigma_sigma * hps.init_q_sigma_scale,
        fixed_p_z_sigma=hps.fixed_p_z_sigma,
        fixed_q_z_sigma=hps.fixed_q_z_sigma,
        fixed_q_w_mu_sigma=hps.fixed_q_w_mu_sigma,
        fixed_q_w_sigma_sigma=hps.fixed_q_w_sigma_sigma,
        fixed_q_w_0_sigma=hps.fixed_q_w_0_sigma,
        dtype=hps.dtype)

    vi = inference.VariationalInference(model, variational, data)

    # Build graph for variational inference.
    vi.build_graph()

    # Build prior and posterior predictive samples
    z_1_prior_sample = model.recurrent_layer_sample(
        variational.sample['w_1_mu'], variational.sample['w_1_sigma'],
        hps.batch_size)
    prior_predictive = model.likelihood_sample(
        variational.sample, z_1_prior_sample, hps.batch_size)
    posterior_predictive = model.likelihood_sample(
        variational.sample, variational.sample['z_1'], hps.batch_size)

    # Build summaries.
    tf.image_summary('prior_predictive', prior_predictive, max_images=10)
    tf.image_summary('posterior_predictive', posterior_predictive,
                     max_images=10)
    tf.scalar_summary('ELBO', vi.scalar_elbo / hps.batch_size)
    tf.scalar_summary('log_p', tf.reduce_mean(vi.log_p))
    tf.scalar_summary('log_q', tf.reduce_mean(vi.log_q))

    # Total loss is the negative ELBO (we maximize the evidence lower bound).
    total_loss = -vi.elbo

    if FLAGS.check_nans:
      checks = tf.add_check_numerics_ops()
      total_loss = tf.control_flow_ops.with_dependencies([checks], total_loss)

    # Specify optimization scheme.
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)

    # Run training.
    if trainer == 'slim':
      train_op = slim.learning.create_train_op(total_loss, optimizer)
      slim.learning.train(
          train_op=train_op,
          logdir=train_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=max_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)
    elif trainer == 'supervisor':
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = optimizer.minimize(total_loss, global_step=global_step)
      summary_op = tf.merge_all_summaries()
      saver = tf.train.Saver()
      sv = tf.train.Supervisor(
          logdir=train_dir,
          is_chief=(FLAGS.task == 0),
          saver=saver,
          summary_op=summary_op,
          global_step=global_step,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_model_secs=FLAGS.save_summaries_secs,
          recovery_wait_secs=5)
      sess = sv.PrepareSession(FLAGS.master)
      sv.StartQueueRunners(sess)
      local_step = 0
      while not sv.ShouldStop():
        _, np_elbo, np_global_step = sess.run(
            [train_op, vi.elbo, global_step])
        np_elbo = np.mean(np_elbo)
        if reporter_fn:
          should_stop = reporter_fn(np_elbo, np_global_step)
          if should_stop:
            sv.RequestStop()
        if np_global_step >= max_steps:
          break
        if local_step % FLAGS.print_stats_every == 0:
          print 'step %d: %g' % (np_global_step - 1, np_elbo / hps.batch_size)
        local_step += 1
      sv.Stop()
      sess.close()
    elif trainer == 'local':
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.contrib.layers.optimize_loss(
          total_loss,
          global_step,
          hps.learning_rate,
          'Adam')
      sess = tf.InteractiveSession()
      sess.run(tf.initialize_all_variables())
      t0 = time.time()
      if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)
      else:
        tf.gfile.MakeDirs(train_dir)
      for i in range(max_steps):
        indexes, images = data_iterator.next()
        feed_dict = {x_indexes: indexes, x: images}
        if i % FLAGS.print_stats_every == 0:
          np_elbo, _, np_prior_predictive, np_posterior_predictive = sess.run(
              [vi.elbo, train_op, prior_predictive, posterior_predictive],
              feed_dict)
          examples_per_s = (hps.batch_size * FLAGS.print_stats_every /
                            (time.time() - t0))
          t0 = time.time()
          print 'iter %d\telbo: %.3e\texamples/s: %.3f' % (
              i, (np.mean(np_elbo) / hps.batch_size), examples_per_s)
          for k in range(hps.samples_to_save):
            im_name = 'i_%d_k_%d_' % (i, k)
            prior_name = im_name + 'prior_predictive.jpg'
            posterior_name = im_name + 'posterior_predictive.jpg'
            imsave(os.path.join(train_dir, prior_name),
                   np_prior_predictive[k, :, :, 0])
            imsave(os.path.join(train_dir, posterior_name),
                   np_posterior_predictive[k, :, :, 0])
        else:
          _ = sess.run(train_op, feed_dict)
      return vi, sess
