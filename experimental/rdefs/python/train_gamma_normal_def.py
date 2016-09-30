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
"""Trains a recurrent DEF with gamma latent variables and gaussian weights.
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
from ops import rmsprop
from ops import tf_lib
from ops import util

flags = tf.flags
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('logdir', '/tmp/write_logs',
                    'Directory where to write event logs.')
flags.DEFINE_integer('seed', 41312, 'Random seed for TensorFlow and Numpy')
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

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


def run_training(hparams, train_dir, max_steps, tuner, container='',
                 trainer='supervisor'):
  """Trains a Gaussian Recurrent DEF.

  Args:
    hparams: A tf.HParams object with hyperparameters for training.
    train_dir: Where to store events files and checkpoints.
    max_steps: Integer number of steps to train.
    tuner: An instance of a vizier tuner.
    container: String specifying container for resource sharing.
    trainer: Train locally by loading an hdf5 file or with Supervisor.

  Returns:
    sess: Optionally, the session for training.
    vi: Optionally, VariationalInference object that has been trained.

  Raises:
    ValueError: if ELBO is nan.
  """
  hps = hparams
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  g = tf.Graph()
  if FLAGS.ps_tasks > 0:
    device_fn = tf.ReplicaDeviceSetter(FLAGS.ps_tasks)
  else:
    device_fn = None
  with g.as_default(), g.device(device_fn), tf.container(container):
    if trainer == 'local':
      x_indexes = tf.placeholder(tf.int32, [hps.batch_size])
      x = tf.placeholder(tf.float32,
                         [hps.batch_size, hps.n_timesteps, hps.timestep_dim, 1])
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

    model = model_factory.GammaNormalRDEF(
        n_timesteps=hps.n_timesteps,
        batch_size=hps.batch_size,
        p_z_shape=hps.p_z_shape,
        p_z_mean=hps.p_z_mean,
        p_w_mean_sigma=hps.p_w_mean_sigma,
        fixed_p_z_mean=hps.fixed_p_z_mean,
        p_w_shape_sigma=hps.p_w_shape_sigma,
        z_dim=hps.z_dim,
        use_bias_observations=hps.use_bias_observations,
        n_samples_latents=hps.n_samples_latents,
        dtype=hps.dtype)

    variational = model_factory.GammaNormalRDEFVariational(
        x_indexes=x_indexes,
        n_examples=hps.n_examples,
        n_timesteps=hps.n_timesteps,
        z_dim=hps.z_dim,
        timestep_dim=hps.timestep_dim,
        init_shape_q_z=hps.init_shape_q_z,
        init_mean_q_z=hps.init_mean_q_z,
        init_sigma_q_w_mean=hps.p_w_mean_sigma * hps.init_q_sigma_scale,
        init_sigma_q_w_shape=hps.p_w_shape_sigma * hps.init_q_sigma_scale,
        init_sigma_q_w_0_sigma=hps.p_w_shape_sigma * hps.init_q_sigma_scale,
        fixed_p_z_mean=hps.fixed_p_z_mean,
        fixed_q_z_mean=hps.fixed_q_z_mean,
        fixed_q_w_mean_sigma=hps.fixed_q_w_mean_sigma,
        fixed_q_w_shape_sigma=hps.fixed_q_w_shape_sigma,
        fixed_q_w_0_sigma=hps.fixed_q_w_0_sigma,
        n_samples_latents=hps.n_samples_latents,
        use_bias_observations=hps.use_bias_observations,
        dtype=hps.dtype)

    vi = inference.VariationalInference(model, variational, data)
    vi.build_graph()

    # Build prior and posterior predictive samples
    z_1_prior_sample = model.recurrent_layer_sample(
        variational.sample['w_1_shape'], variational.sample['w_1_mean'],
        hps.batch_size)
    prior_predictive = model.likelihood_sample(
        variational.sample, z_1_prior_sample, hps.batch_size)
    posterior_predictive = model.likelihood_sample(
        variational.sample, variational.sample['z_1'], hps.batch_size)

    # Build summaries.
    float32 = lambda x: tf.cast(x, tf.float32)
    tf.image_summary('prior_predictive',
                     float32(prior_predictive),
                     max_images=10)
    tf.image_summary('posterior_predictive',
                     float32(posterior_predictive),
                     max_images=10)
    tf.scalar_summary('ELBO', vi.scalar_elbo / hps.batch_size)
    tf.scalar_summary('log_p', tf.reduce_mean(vi.log_p))
    tf.scalar_summary('log_q', tf.reduce_mean(vi.log_q))

    global_step = tf.contrib.framework.get_or_create_global_step()

    # Specify optimization scheme.
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
    if hps.control_variate == 'none':
      train_op = optimizer.minimize(-vi.surrogate_elbo, global_step=global_step)
    elif hps.control_variate == 'covariance':
      train_non_reparam = rmsprop.maximize_with_control_variate(
          learning_rate=hps.learning_rate,
          learning_signal=vi.elbo,
          log_prob=vi.log_q,
          variable_list=tf.get_collection('non_reparam_variables'),
          global_step=global_step)
      grad_tensors = [v.values if 'embedding_lookup' in v.name else v
                      for v in tf.get_collection('non_reparam_variable_grads')]

      train_reparam = optimizer.minimize(
          -tf.reduce_mean(vi.elbo, 0),  # optimize the mean across samples
          var_list=tf.get_collection('reparam_variables'))
      train_op = tf.group(train_reparam, train_non_reparam)

    if trainer == 'supervisor':
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = optimizer.minimize(-vi.elbo, global_step=global_step)
      summary_op = tf.merge_all_summaries()
      saver = tf.train.Saver()
      sv = tf.Supervisor(
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
        if tuner is not None:
          if np.isnan(np_elbo):
            tuner.report_done(infeasible=True, infeasible_reason='ELBO is nan')
            should_stop = True
          else:
            should_stop = tuner.report_measure(float(np_elbo),
                                               global_step=np_global_step)
            if should_stop:
              tuner.report_done()
              sv.RequestStop()
        if np_global_step >= max_steps:
          break
        if local_step % FLAGS.print_stats_every == 0:
          print 'step %d: %g' % (np_global_step - 1, np_elbo / hps.batch_size)
        local_step += 1
      sv.Stop()
      sess.close()
    elif trainer == 'local':
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
          _, np_prior_predictive, np_posterior_predictive = sess.run(
              [train_op, prior_predictive, posterior_predictive],
              feed_dict)
          print 'prior_predictive', np_prior_predictive.flatten()
          print 'posterior_predictive', np_posterior_predictive.flatten()
          print 'data', images.flatten()
          examples_per_s = (hps.batch_size * FLAGS.print_stats_every /
                            (time.time() - t0))
          q_z = variational.params['z_1'].distribution
          alpha = q_z.alpha
          beta = q_z.beta
          mean = alpha / beta
          grad_list = []
          elbo_list = []
          for k in range(100):
            elbo_list.append(vi.elbo.eval(feed_dict))
            grads = sess.run(grad_tensors, feed_dict)
            grad_list.append(grads)
          np_elbo = np.mean(np.vstack([np.sum(v, axis=1) for v in elbo_list]))
          if np.isnan(np_elbo):
            raise ValueError('ELBO is NaN. Please keep trying!')
          grads_per_var = [np.stack(
              [g_sample[var_idx] for g_sample in grad_list])
                           for var_idx in range(
                               len(tf.get_collection(
                                   'non_reparam_variable_grads')))]
          grads_per_timestep = [np.split(g, hps.n_timesteps, axis=2)
                                for g in grads_per_var]
          grads_per_timestep_per_dim = [[np.split(g, hps.z_dim, axis=3) for g in
                                         g_list] for g_list
                                        in grads_per_timestep]
          grads_per_timestep_per_dim = [sum(g_list, []) for g_list in
                                        grads_per_timestep_per_dim]
          print 'variance of gradients for each variable: '
          for var_idx, var in enumerate(
              tf.get_collection('non_reparam_variable_grads')):
            print 'variable: %s' % var.name
            var = [np.var(g, axis=0) for g in
                   grads_per_timestep_per_dim[var_idx]]
            print 'variance is: ', np.stack(var).flatten()
          print 'alpha ', alpha.eval(feed_dict).flatten()
          print 'mean ', mean.eval(feed_dict).flatten()
          print 'bernoulli p ', np.mean(
              vi.model.p_x_zw_bernoulli_p.eval(feed_dict), axis=0).flatten()
          t0 = time.time()
          print 'iter %d\telbo: %.3e\texamples/s: %.3f' % (
              i, np_elbo, examples_per_s)
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


def main(unused_argv):
  """Trains a gaussian def locally."""
  if tf.gfile.Exists(FLAGS.logdir) and FLAGS.delete_logdir:
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  # The HParams are commented in training_params.py
  try:
    hparams = tf.HParams
  except AttributeError:
    hparams = tf_lib.HParams
  hparams = hparams(
      dataset='alternating',
      z_dim=1,
      timestep_dim=1,
      n_timesteps=2,
      batch_size=1,
      n_examples=1,
      samples_to_save=1,
      learning_rate=0.01,
      momentum=0.0,
      n_samples_latents=100,
      p_z_shape=0.1,
      p_z_mean=1.,
      p_w_mean_sigma=5.,
      p_w_shape_sigma=5.,
      init_q_sigma_scale=0.1,
      use_bias_observations=True,
      init_q_z_scale=1.,
      init_shape_q_z=util.softplus(0.1),
      init_mean_q_z=util.softplus(0.01),
      fixed_p_z_mean=False,
      fixed_q_z_mean=False,
      fixed_q_z_shape=False,
      fixed_q_w_mean_sigma=False,
      fixed_q_w_shape_sigma=False,
      fixed_q_w_0_sigma=False,
      dtype='float64',
      control_variate='covariance')
  run_training(hparams, FLAGS.logdir, FLAGS.max_steps, None, trainer='local')

if __name__ == '__main__':
  tf.app.run()
