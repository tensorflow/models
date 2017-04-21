# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

r""" Script to train and test the grid navigation agent.
Usage:
  1. Testing a model.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+bench_test \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r

  2. Training a model (locally).
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_

  3. Training a model (distributed).
  # See https://www.tensorflow.org/deploy/distributed on how to setup distributed
  # training.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_ \
    --ps_tasks $num_ps --master $master_name --task $worker_id
"""

import sys, os, numpy as np
import copy
import argparse, pprint
import time
import cProfile
import platform


import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables

import logging
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cfgs import config_cmp
from cfgs import config_vision_baseline
import datasets.nav_env as nav_env
import src.file_utils as fu 
import src.utils as utils
import tfcode.cmp as cmp 
from tfcode import tf_utils
from tfcode import vision_baseline_lstm

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '',
                    'The address of the tensorflow master')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers. If the '
                     'value is 0, then the parameters are handled locally by '
                     'the worker.')
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training '
                     'with multiple workers to identify each worker.')

flags.DEFINE_integer('num_workers', 1, '')

flags.DEFINE_string('config_name', '', '')

flags.DEFINE_string('logdir', '', '')

flags.DEFINE_integer('solver_seed', 0, '')

flags.DEFINE_integer('delay_start_iters', 20, '')

logging.basicConfig(level=logging.INFO)

def main(_):
  _launcher(FLAGS.config_name, FLAGS.logdir)

def _launcher(config_name, logdir):
  args = _setup_args(config_name, logdir)

  fu.makedirs(args.logdir)

  if args.control.train:
    _train(args)

  if args.control.test:
    _test(args)

def get_args_for_config(config_name):
  configs = config_name.split('.')
  type = configs[0]
  config_name = '.'.join(configs[1:])
  if type == 'cmp':
    args = config_cmp.get_args_for_config(config_name)
    args.setup_to_run = cmp.setup_to_run
    args.setup_train_step_kwargs = cmp.setup_train_step_kwargs

  elif type == 'bl':
    args = config_vision_baseline.get_args_for_config(config_name)
    args.setup_to_run = vision_baseline_lstm.setup_to_run
    args.setup_train_step_kwargs = vision_baseline_lstm.setup_train_step_kwargs

  else:
    logging.fatal('Unknown type: {:s}'.format(type))
  return args

def _setup_args(config_name, logdir):
  args = get_args_for_config(config_name)
  args.solver.num_workers = FLAGS.num_workers
  args.solver.task = FLAGS.task
  args.solver.ps_tasks = FLAGS.ps_tasks
  args.solver.master = FLAGS.master
  args.solver.seed = FLAGS.solver_seed
  args.logdir = logdir
  args.navtask.logdir = None
  return args

def _train(args):
  container_name = ""

  R = lambda: nav_env.get_multiplexer_class(args.navtask, args.solver.task)
  m = utils.Foo()
  m.tf_graph = tf.Graph()

  config = tf.ConfigProto()
  config.device_count['GPU'] = 1

  with m.tf_graph.as_default():
    with tf.device(tf.train.replica_device_setter(args.solver.ps_tasks,
                                          merge_devices=True)):
      with tf.container(container_name):
        m = args.setup_to_run(m, args, is_training=True,
                             batch_norm_is_training=True, summary_mode='train')

        train_step_kwargs = args.setup_train_step_kwargs(
            m, R(), os.path.join(args.logdir, 'train'), rng_seed=args.solver.task,
            is_chief=args.solver.task==0,
            num_steps=args.navtask.task_params.num_steps*args.navtask.task_params.num_goals, iters=1,
            train_display_interval=args.summary.display_interval,
            dagger_sample_bn_false=args.arch.dagger_sample_bn_false)

        delay_start = (args.solver.task*(args.solver.task+1))/2 * FLAGS.delay_start_iters
        logging.error('delaying start for task %d by %d steps.',
                      args.solver.task, delay_start)

        additional_args = {}
        final_loss = slim.learning.train(
            train_op=m.train_op,
            logdir=args.logdir,
            master=args.solver.master,
            is_chief=args.solver.task == 0,
            number_of_steps=args.solver.max_steps,
            train_step_fn=tf_utils.train_step_custom_online_sampling,
            train_step_kwargs=train_step_kwargs,
            global_step=m.global_step_op,
            init_op=m.init_op,
            init_fn=m.init_fn,
            sync_optimizer=m.sync_optimizer,
            saver=m.saver_op,
            startup_delay_steps=delay_start,
            summary_op=None, session_config=config, **additional_args)

def _test(args):
  args.solver.master = ''
  container_name = ""
  checkpoint_dir = os.path.join(format(args.logdir))
  logging.error('Checkpoint_dir: %s', args.logdir)

  config = tf.ConfigProto();
  config.device_count['GPU'] = 1;

  m = utils.Foo()
  m.tf_graph = tf.Graph()

  rng_data_seed = 0; rng_action_seed = 0;
  R = lambda: nav_env.get_multiplexer_class(args.navtask, rng_data_seed)
  with m.tf_graph.as_default():
    with tf.container(container_name):
      m = args.setup_to_run(
        m, args, is_training=False,
        batch_norm_is_training=args.control.force_batchnorm_is_training_at_test,
        summary_mode=args.control.test_mode)
      train_step_kwargs = args.setup_train_step_kwargs(
        m, R(), os.path.join(args.logdir, args.control.test_name),
        rng_seed=rng_data_seed, is_chief=True,
        num_steps=args.navtask.task_params.num_steps*args.navtask.task_params.num_goals,
        iters=args.summary.test_iters, train_display_interval=None,
        dagger_sample_bn_false=args.arch.dagger_sample_bn_false)

      saver = slim.learning.tf_saver.Saver(variables.get_variables_to_restore())

      sv = slim.learning.supervisor.Supervisor(
          graph=ops.get_default_graph(), logdir=None, init_op=m.init_op,
          summary_op=None, summary_writer=None, global_step=None, saver=m.saver_op)

      last_checkpoint = None
      reported = False
      while True:
        last_checkpoint_ = None
        while last_checkpoint_ is None:
          last_checkpoint_ = slim.evaluation.wait_for_new_checkpoint(
            checkpoint_dir, last_checkpoint, seconds_to_sleep=10, timeout=60)
        if last_checkpoint_ is None: break

        last_checkpoint = last_checkpoint_
        checkpoint_iter = int(os.path.basename(last_checkpoint).split('-')[1])

        logging.info('Starting evaluation at %s using checkpoint %s.',
                     time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()),
                     last_checkpoint)

        if (args.control.only_eval_when_done == False or 
            checkpoint_iter >= args.solver.max_steps):
          start = time.time()
          logging.info('Starting evaluation at %s using checkpoint %s.', 
                       time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()),
                       last_checkpoint)

          with sv.managed_session(args.solver.master, config=config,
                                  start_standard_services=False) as sess:
            sess.run(m.init_op)
            sv.saver.restore(sess, last_checkpoint)
            sv.start_queue_runners(sess)
            if args.control.reset_rng_seed:
              train_step_kwargs['rng_data'] = [np.random.RandomState(rng_data_seed),
                                               np.random.RandomState(rng_data_seed)]
              train_step_kwargs['rng_action'] = np.random.RandomState(rng_action_seed)
            vals, _ = tf_utils.train_step_custom_online_sampling(
                sess, None, m.global_step_op, train_step_kwargs,
                mode=args.control.test_mode)
            should_stop = False

            if checkpoint_iter >= args.solver.max_steps: 
              should_stop = True

            if should_stop:
              break

if __name__ == '__main__':
  app.run()
