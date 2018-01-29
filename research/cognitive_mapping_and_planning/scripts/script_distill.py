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

r""" Script to setup the grid moving agent.

blaze build --define=ION_GFX_OGLES20=1 -c opt --copt=-mavx --config=cuda_clang \
    learning/brain/public/tensorflow_std_server{,_gpu} \
    experimental/users/saurabhgupta/navigation/cmp/scripts/script_distill.par \
    experimental/users/saurabhgupta/navigation/cmp/scripts/script_distill


./blaze-bin/experimental/users/saurabhgupta/navigation/cmp/scripts/script_distill \
  --logdir=/cns/iq-d/home/saurabhgupta/output/stanford-distill/local/v0/ \
  --config_name 'v0+train' --gfs_user robot-intelligence-gpu

"""
import sys, os, numpy as np
import copy
import argparse, pprint
import time
import cProfile


import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables 

import logging
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cfgs import config_distill
from tfcode import tf_utils
import src.utils as utils
import src.file_utils as fu
import tfcode.distillation as distill 
import datasets.nav_env as nav_env

FLAGS = flags.FLAGS

flags.DEFINE_string('master', 'local',
                    'The name of the TensorFlow master to use.')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers. If the '
                     'value is 0, then the parameters are handled locally by '
                     'the worker.')
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training '
                     'with multiple workers to identify each worker.')

flags.DEFINE_integer('num_workers', 1, '')

flags.DEFINE_string('config_name', '', '')

flags.DEFINE_string('logdir', '', '')

def main(_):
  args = config_distill.get_args_for_config(FLAGS.config_name)
  args.logdir = FLAGS.logdir
  args.solver.num_workers = FLAGS.num_workers
  args.solver.task = FLAGS.task
  args.solver.ps_tasks = FLAGS.ps_tasks
  args.solver.master = FLAGS.master
  
  args.buildinger.env_class = nav_env.MeshMapper
  fu.makedirs(args.logdir)
  args.buildinger.logdir = args.logdir
  R = nav_env.get_multiplexor_class(args.buildinger, args.solver.task)
  
  if False:
    pr = cProfile.Profile()
    pr.enable()
    rng = np.random.RandomState(0)
    for i in range(1):
      b, instances_perturbs = R.sample_building(rng)
      inputs = b.worker(*(instances_perturbs))
      for j in range(inputs['imgs'].shape[0]):
        p = os.path.join('tmp', '{:d}.png'.format(j))
        img = inputs['imgs'][j,0,:,:,:3]*1
        img = (img).astype(np.uint8)
        fu.write_image(p, img)
      print(inputs['imgs'].shape)
      inputs = R.pre(inputs)
    pr.disable()
    pr.print_stats(2)

  if args.control.train:
    if not gfile.Exists(args.logdir):
      gfile.MakeDirs(args.logdir)
   
    m = utils.Foo()
    m.tf_graph = tf.Graph()
    
    config = tf.ConfigProto()
    config.device_count['GPU'] = 1
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with m.tf_graph.as_default():
      with tf.device(tf.train.replica_device_setter(args.solver.ps_tasks)):
        m = distill.setup_to_run(m, args, is_training=True,
                                batch_norm_is_training=True)

        train_step_kwargs = distill.setup_train_step_kwargs_mesh(
            m, R, os.path.join(args.logdir, 'train'),
            rng_seed=args.solver.task, is_chief=args.solver.task==0, iters=1,
            train_display_interval=args.summary.display_interval)

        final_loss = slim.learning.train(
            train_op=m.train_op,
            logdir=args.logdir,
            master=args.solver.master,
            is_chief=args.solver.task == 0,
            number_of_steps=args.solver.max_steps,
            train_step_fn=tf_utils.train_step_custom,
            train_step_kwargs=train_step_kwargs,
            global_step=m.global_step_op,
            init_op=m.init_op,
            init_fn=m.init_fn,
            sync_optimizer=m.sync_optimizer,
            saver=m.saver_op,
            summary_op=None, session_config=config)
 
  if args.control.test:
    m = utils.Foo()
    m.tf_graph = tf.Graph()
    checkpoint_dir = os.path.join(format(args.logdir))
    with m.tf_graph.as_default():
      m = distill.setup_to_run(m, args, is_training=False,
                              batch_norm_is_training=args.control.force_batchnorm_is_training_at_test)
      
      train_step_kwargs = distill.setup_train_step_kwargs_mesh(
          m, R, os.path.join(args.logdir, args.control.test_name),
          rng_seed=args.solver.task+1, is_chief=args.solver.task==0,
          iters=args.summary.test_iters, train_display_interval=None)
      
      sv = slim.learning.supervisor.Supervisor(
          graph=ops.get_default_graph(), logdir=None, init_op=m.init_op,
          summary_op=None, summary_writer=None, global_step=None, saver=m.saver_op)

      last_checkpoint = None
      while True:
        last_checkpoint = slim.evaluation.wait_for_new_checkpoint(checkpoint_dir, last_checkpoint)
        checkpoint_iter = int(os.path.basename(last_checkpoint).split('-')[1])
        start = time.time()
        logging.info('Starting evaluation at %s using checkpoint %s.', 
                     time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()),
                     last_checkpoint)
        
        config = tf.ConfigProto()
        config.device_count['GPU'] = 1
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        
        with sv.managed_session(args.solver.master,config=config,
                                start_standard_services=False) as sess:
          sess.run(m.init_op)
          sv.saver.restore(sess, last_checkpoint)
          sv.start_queue_runners(sess)
          vals, _ = tf_utils.train_step_custom(
              sess, None, m.global_step_op, train_step_kwargs, mode='val')
          if checkpoint_iter >= args.solver.max_steps:
            break

if __name__ == '__main__':
  app.run()
