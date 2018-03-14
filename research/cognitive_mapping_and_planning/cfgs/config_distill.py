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

import pprint
import copy
import os
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import logging
import src.utils as utils
import cfgs.config_common as cc


import tensorflow as tf

rgb_resnet_v2_50_path = 'cache/resnet_v2_50_inception_preprocessed/model.ckpt-5136169'

def get_default_args():
  robot = utils.Foo(radius=15, base=10, height=140, sensor_height=120,
                    camera_elevation_degree=-15)

  camera_param = utils.Foo(width=225, height=225, z_near=0.05, z_far=20.0,
                           fov=60., modalities=['rgb', 'depth'])

  env = utils.Foo(padding=10, resolution=5, num_point_threshold=2,
                  valid_min=-10, valid_max=200, n_samples_per_face=200)

  data_augment = utils.Foo(lr_flip=0, delta_angle=1, delta_xy=4, relight=False,
                           relight_fast=False, structured=False)

  task_params = utils.Foo(num_actions=4, step_size=4, num_steps=0,
                          batch_size=32, room_seed=0, base_class='Building',
                          task='mapping', n_ori=6, data_augment=data_augment,
                          output_transform_to_global_map=False,
                          output_canonical_map=False,
                          output_incremental_transform=False,
                          output_free_space=False, move_type='shortest_path',
                          toy_problem=0)

  buildinger_args = utils.Foo(building_names=['area1_gates_wingA_floor1_westpart'],
                              env_class=None, robot=robot, 
                              task_params=task_params, env=env,
                              camera_param=camera_param)

  solver_args = utils.Foo(seed=0, learning_rate_decay=0.1,
                          clip_gradient_norm=0, max_steps=120000,
                          initial_learning_rate=0.001, momentum=0.99,
                          steps_per_decay=40000, logdir=None, sync=False,
                          adjust_lr_sync=True, wt_decay=0.0001,
                          data_loss_wt=1.0, reg_loss_wt=1.0,
                          num_workers=1, task=0, ps_tasks=0, master='local')

  summary_args = utils.Foo(display_interval=1, test_iters=100)

  control_args = utils.Foo(train=False, test=False,
                           force_batchnorm_is_training_at_test=False)
  
  arch_args = utils.Foo(rgb_encoder='resnet_v2_50', d_encoder='resnet_v2_50')

  return utils.Foo(solver=solver_args,
                   summary=summary_args, control=control_args, arch=arch_args,
                   buildinger=buildinger_args)

def get_vars(config_name):
  vars = config_name.split('_')
  if len(vars) == 1: # All data or not.
    vars.append('noall')
  if len(vars) == 2: # n_ori
    vars.append('4')
  logging.error('vars: %s', vars)
  return vars

def get_args_for_config(config_name):
  args = get_default_args()
  config_name, mode = config_name.split('+')
  vars = get_vars(config_name)
  
  logging.info('config_name: %s, mode: %s', config_name, mode)
  
  args.buildinger.task_params.n_ori = int(vars[2])
  args.solver.freeze_conv = True
  args.solver.pretrained_path = rgb_resnet_v2_50_path
  args.buildinger.task_params.img_channels = 5
  args.solver.data_loss_wt = 0.00001
 
  if vars[0] == 'v0':
    None
  else:
    logging.error('config_name: %s undefined', config_name)

  args.buildinger.task_params.height = args.buildinger.camera_param.height
  args.buildinger.task_params.width = args.buildinger.camera_param.width
  args.buildinger.task_params.modalities = args.buildinger.camera_param.modalities
  
  if vars[1] == 'all':
    args = cc.get_args_for_mode_building_all(args, mode)
  elif vars[1] == 'noall':
    args = cc.get_args_for_mode_building(args, mode)
  
  # Log the arguments
  logging.error('%s', args)
  return args
