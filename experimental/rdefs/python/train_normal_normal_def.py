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
"""Wraps train_normal_normal_def."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import train_normal_normal_def_lib
from ops import tf_lib


FLAGS = tf.flags.FLAGS


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
      dataset='MNIST',
      z_dim=200,
      timestep_dim=28,
      n_timesteps=28,
      batch_size=50,
      samples_to_save=10,
      learning_rate=0.1,
      n_examples=50,
      momentum=0.0,
      p_z_sigma=1.,
      p_w_mu_sigma=1.,
      p_w_sigma_sigma=1.,
      init_q_sigma_scale=0.1,
      fixed_p_z_sigma=True,
      fixed_q_z_sigma=False,
      fixed_q_w_mu_sigma=False,
      fixed_q_w_sigma_sigma=False,
      fixed_q_w_0_sigma=False,
      dtype='float32')
  tf.logging.info('Starting experiment in %s with params %s', FLAGS.logdir,
                  hparams)
  train_normal_normal_def_lib.run_training(
      hparams, FLAGS.logdir, FLAGS.max_steps, None,
      trainer=FLAGS.trainer)


if __name__ == '__main__':
  tf.app.run()
