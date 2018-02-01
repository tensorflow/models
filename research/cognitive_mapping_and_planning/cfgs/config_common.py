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

import os
import numpy as np
import logging
import src.utils as utils
import datasets.nav_env_config as nec
from datasets import factory

def adjust_args_for_mode(args, mode):
  if mode == 'train':
    args.control.train = True
  
  elif mode == 'val1':
    # Same settings as for training, to make sure nothing wonky is happening
    # there.
    args.control.test = True
    args.control.test_mode = 'val'
    args.navtask.task_params.batch_size = 32

  elif mode == 'val2':
    # No data augmentation, not sampling but taking the argmax action, not
    # sampling from the ground truth at all.
    args.control.test = True
    args.arch.action_sample_type = 'argmax'
    args.arch.sample_gt_prob_type = 'zero'
    args.navtask.task_params.data_augment = \
      utils.Foo(lr_flip=0, delta_angle=0, delta_xy=0, relight=False,
                relight_fast=False, structured=False)
    args.control.test_mode = 'val'
    args.navtask.task_params.batch_size = 32

  elif mode == 'bench':
    # Actually testing the agent in settings that are kept same between
    # different runs.
    args.navtask.task_params.batch_size = 16
    args.control.test = True
    args.arch.action_sample_type = 'argmax'
    args.arch.sample_gt_prob_type = 'zero'
    args.navtask.task_params.data_augment = \
      utils.Foo(lr_flip=0, delta_angle=0, delta_xy=0, relight=False,
                relight_fast=False, structured=False)
    args.summary.test_iters = 250
    args.control.only_eval_when_done = True
    args.control.reset_rng_seed = True
    args.control.test_mode = 'test'
  else:
    logging.fatal('Unknown mode: %s.', mode)
    assert(False)
  return args

def get_solver_vars(solver_str):
  if solver_str == '': vals = []; 
  else: vals = solver_str.split('_')
  ks = ['clip', 'dlw', 'long', 'typ', 'isdk', 'adam_eps', 'init_lr'];
  ks = ks[:len(vals)]

  # Gradient clipping or not.
  if len(vals) == 0: ks.append('clip'); vals.append('noclip');
  # data loss weight.
  if len(vals) == 1: ks.append('dlw');  vals.append('dlw20')
  # how long to train for.
  if len(vals) == 2: ks.append('long');  vals.append('nolong')
  # Adam
  if len(vals) == 3: ks.append('typ');  vals.append('adam2')
  # reg loss wt
  if len(vals) == 4: ks.append('rlw');  vals.append('rlw1')
  # isd_k
  if len(vals) == 5: ks.append('isdk');  vals.append('isdk415') # 415, inflexion at 2.5k.
  # adam eps
  if len(vals) == 6: ks.append('adam_eps');  vals.append('aeps1en8')
  # init lr
  if len(vals) == 7: ks.append('init_lr');  vals.append('lr1en3')

  assert(len(vals) == 8)
  
  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)
  logging.error('solver_vars: %s', vars)
  return vars

def process_solver_str(solver_str):
  solver = utils.Foo(
      seed=0, learning_rate_decay=None, clip_gradient_norm=None, max_steps=None,
      initial_learning_rate=None, momentum=None, steps_per_decay=None,
      logdir=None, sync=False, adjust_lr_sync=True, wt_decay=0.0001,
      data_loss_wt=None, reg_loss_wt=None, freeze_conv=True, num_workers=1,
      task=0, ps_tasks=0, master='local', typ=None, momentum2=None,
      adam_eps=None)

  # Clobber with overrides from solver str.
  solver_vars = get_solver_vars(solver_str)

  solver.data_loss_wt          = float(solver_vars.dlw[3:].replace('x', '.'))
  solver.adam_eps              = float(solver_vars.adam_eps[4:].replace('x', '.').replace('n', '-'))
  solver.initial_learning_rate = float(solver_vars.init_lr[2:].replace('x', '.').replace('n', '-'))
  solver.reg_loss_wt           = float(solver_vars.rlw[3:].replace('x', '.'))
  solver.isd_k                 = float(solver_vars.isdk[4:].replace('x', '.'))

  long = solver_vars.long
  if long == 'long':
    solver.steps_per_decay = 40000
    solver.max_steps = 120000
  elif long == 'long2':
    solver.steps_per_decay = 80000
    solver.max_steps = 120000
  elif long == 'nolong' or long == 'nol':
    solver.steps_per_decay = 20000
    solver.max_steps = 60000
  else:
    logging.fatal('solver_vars.long should be long, long2, nolong or nol.')
    assert(False)

  clip = solver_vars.clip
  if clip == 'noclip' or clip == 'nocl':
    solver.clip_gradient_norm = 0
  elif clip[:4] == 'clip':
    solver.clip_gradient_norm = float(clip[4:].replace('x', '.'))
  else:
    logging.fatal('Unknown solver_vars.clip: %s', clip)
    assert(False)

  typ = solver_vars.typ
  if typ == 'adam':
    solver.typ = 'adam'
    solver.momentum = 0.9
    solver.momentum2 = 0.999
    solver.learning_rate_decay = 1.0
  elif typ == 'adam2':
    solver.typ = 'adam'
    solver.momentum = 0.9
    solver.momentum2 = 0.999
    solver.learning_rate_decay = 0.1
  elif typ == 'sgd':
    solver.typ = 'sgd'
    solver.momentum = 0.99
    solver.momentum2 = None
    solver.learning_rate_decay = 0.1
  else:
    logging.fatal('Unknown solver_vars.typ: %s', typ)
    assert(False)

  logging.error('solver: %s', solver)
  return solver

def get_navtask_vars(navtask_str):
  if navtask_str == '': vals = []
  else: vals = navtask_str.split('_')

  ks_all = ['dataset_name', 'modality', 'task', 'history', 'max_dist',
            'num_steps', 'step_size', 'n_ori', 'aux_views', 'data_aug']
  ks = ks_all[:len(vals)]

  # All data or not.
  if len(vals) == 0: ks.append('dataset_name'); vals.append('sbpd')
  # modality
  if len(vals) == 1: ks.append('modality'); vals.append('rgb')
  # semantic task?
  if len(vals) == 2: ks.append('task'); vals.append('r2r')
  # number of history frames.
  if len(vals) == 3: ks.append('history'); vals.append('h0')
  # max steps
  if len(vals) == 4: ks.append('max_dist'); vals.append('32')
  # num steps
  if len(vals) == 5: ks.append('num_steps'); vals.append('40')
  # step size
  if len(vals) == 6: ks.append('step_size'); vals.append('8')
  # n_ori
  if len(vals) == 7: ks.append('n_ori'); vals.append('4')
  # Auxiliary views.
  if len(vals) == 8: ks.append('aux_views'); vals.append('nv0')
  # Normal data augmentation as opposed to structured data augmentation (if set
  # to straug.
  if len(vals) == 9: ks.append('data_aug'); vals.append('straug')

  assert(len(vals) == 10)
  for i in range(len(ks)):
    assert(ks[i] == ks_all[i])

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)
  logging.error('navtask_vars: %s', vals)
  return vars

def process_navtask_str(navtask_str):
  navtask = nec.nav_env_base_config()
  
  # Clobber with overrides from strings.
  navtask_vars = get_navtask_vars(navtask_str)

  navtask.task_params.n_ori = int(navtask_vars.n_ori)
  navtask.task_params.max_dist = int(navtask_vars.max_dist)
  navtask.task_params.num_steps = int(navtask_vars.num_steps)
  navtask.task_params.step_size = int(navtask_vars.step_size)
  navtask.task_params.data_augment.delta_xy = int(navtask_vars.step_size)/2.
  n_aux_views_each = int(navtask_vars.aux_views[2])
  aux_delta_thetas = np.concatenate((np.arange(n_aux_views_each) + 1,
                                     -1 -np.arange(n_aux_views_each)))
  aux_delta_thetas = aux_delta_thetas*np.deg2rad(navtask.camera_param.fov)
  navtask.task_params.aux_delta_thetas = aux_delta_thetas
  
  if navtask_vars.data_aug == 'aug':
    navtask.task_params.data_augment.structured = False
  elif navtask_vars.data_aug == 'straug':
    navtask.task_params.data_augment.structured = True
  else:
    logging.fatal('Unknown navtask_vars.data_aug %s.', navtask_vars.data_aug)
    assert(False)

  navtask.task_params.num_history_frames = int(navtask_vars.history[1:])
  navtask.task_params.n_views = 1+navtask.task_params.num_history_frames
  
  navtask.task_params.goal_channels = int(navtask_vars.n_ori)
  
  if navtask_vars.task == 'hard': 
    navtask.task_params.type = 'rng_rejection_sampling_many'
    navtask.task_params.rejection_sampling_M = 2000
    navtask.task_params.min_dist = 10
  elif navtask_vars.task == 'r2r':
    navtask.task_params.type = 'room_to_room_many'
  elif navtask_vars.task == 'ST':
    # Semantic task at hand.
    navtask.task_params.goal_channels = \
        len(navtask.task_params.semantic_task.class_map_names)
    navtask.task_params.rel_goal_loc_dim = \
        len(navtask.task_params.semantic_task.class_map_names)
    navtask.task_params.type = 'to_nearest_obj_acc'
  else:
    logging.fatal('navtask_vars.task: should be hard or r2r, ST')
    assert(False)
  
  if navtask_vars.modality == 'rgb':
    navtask.camera_param.modalities = ['rgb']
    navtask.camera_param.img_channels = 3
  elif navtask_vars.modality == 'd':
    navtask.camera_param.modalities = ['depth']
    navtask.camera_param.img_channels = 2
  
  navtask.task_params.img_height   = navtask.camera_param.height
  navtask.task_params.img_width    = navtask.camera_param.width
  navtask.task_params.modalities   = navtask.camera_param.modalities
  navtask.task_params.img_channels = navtask.camera_param.img_channels
  navtask.task_params.img_fov      = navtask.camera_param.fov
  
  navtask.dataset = factory.get_dataset(navtask_vars.dataset_name)
  return navtask
