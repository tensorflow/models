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
import os
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import logging
import src.utils as utils
import cfgs.config_common as cc
import datasets.nav_env_config as nec


import tensorflow as tf

FLAGS = flags.FLAGS

get_solver_vars = cc.get_solver_vars
get_navtask_vars = cc.get_navtask_vars


rgb_resnet_v2_50_path = 'data/init_models/resnet_v2_50/model.ckpt-5136169'
d_resnet_v2_50_path = 'data/init_models/distill_rgb_to_d_resnet_v2_50/model.ckpt-120002'

def get_default_args():
  summary_args = utils.Foo(display_interval=1, test_iters=26,
                           arop_full_summary_iters=14)

  control_args = utils.Foo(train=False, test=False,
                           force_batchnorm_is_training_at_test=False,
                           reset_rng_seed=False, only_eval_when_done=False,
                           test_mode=None)
  return summary_args, control_args

def get_default_baseline_args():
  batch_norm_param = {'center': True, 'scale': True,
                      'activation_fn':tf.nn.relu}
  arch_args = utils.Foo(
      pred_neurons=[], goal_embed_neurons=[], img_embed_neurons=[],
      batch_norm_param=batch_norm_param, dim_reduce_neurons=64, combine_type='',
      encoder='resnet_v2_50', action_sample_type='sample',
      action_sample_combine_type='one_or_other',
      sample_gt_prob_type='inverse_sigmoid_decay', dagger_sample_bn_false=True,
      isd_k=750., use_visit_count=False, lstm_output=False, lstm_ego=False,
      lstm_img=False, fc_dropout=0.0, embed_goal_for_state=False,
      lstm_output_init_state_from_goal=False)
  return arch_args

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  
  ks = ['ver', 'lstm_dim', 'dropout']
  
  # Exp Ver
  if len(vals) == 0: vals.append('v0')
  # LSTM dimentsions
  if len(vals) == 1: vals.append('lstm2048')
  # Dropout
  if len(vals) == 2: vals.append('noDO')
  
  assert(len(vals) == 3)
  
  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)
  
  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch = get_default_baseline_args()
  arch_vars = get_arch_vars(arch_str)

  args.navtask.task_params.outputs.rel_goal_loc = True
  args.navtask.task_params.input_type = 'vision'
  args.navtask.task_params.outputs.images = True
  
  if args.navtask.camera_param.modalities[0] == 'rgb':
    args.solver.pretrained_path = rgb_resnet_v2_50_path
  elif args.navtask.camera_param.modalities[0] == 'depth':
    args.solver.pretrained_path = d_resnet_v2_50_path
  else:
    logging.fatal('Neither of rgb or d')

  if arch_vars.dropout == 'DO': 
    args.arch.fc_dropout = 0.5

  args.tfcode = 'B'
  
  exp_ver = arch_vars.ver
  if exp_ver == 'v0':
    # Multiplicative interaction between goal loc and image features.
    args.arch.combine_type = 'multiply'
    args.arch.pred_neurons = [256, 256]
    args.arch.goal_embed_neurons = [64, 8]
    args.arch.img_embed_neurons = [1024, 512, 256*8]
  
  elif exp_ver == 'v1':
    # Additive interaction between goal and image features.
    args.arch.combine_type = 'add'
    args.arch.pred_neurons = [256, 256]
    args.arch.goal_embed_neurons = [64, 256]
    args.arch.img_embed_neurons = [1024, 512, 256]
  
  elif exp_ver == 'v2':
    # LSTM at the output on top of multiple interactions.
    args.arch.combine_type = 'multiply'
    args.arch.goal_embed_neurons = [64, 8]
    args.arch.img_embed_neurons = [1024, 512, 256*8]
    args.arch.lstm_output = True
    args.arch.lstm_output_dim = int(arch_vars.lstm_dim[4:])
    args.arch.pred_neurons = [256] # The other is inside the LSTM.
  
  elif exp_ver == 'v0blind':
    # LSTM only on the goal location.
    args.arch.combine_type = 'goalonly'
    args.arch.goal_embed_neurons = [64, 256]
    args.arch.img_embed_neurons = [2] # I dont know what it will do otherwise.
    args.arch.lstm_output = True
    args.arch.lstm_output_dim = 256
    args.arch.pred_neurons = [256] # The other is inside the LSTM.
  
  else:
    logging.fatal('exp_ver: %s undefined', exp_ver)
    assert(False)

  # Log the arguments
  logging.error('%s', args)
  return args

def get_args_for_config(config_name):
  args = utils.Foo()

  args.summary, args.control = get_default_args()

  exp_name, mode_str = config_name.split('+')
  arch_str, solver_str, navtask_str = exp_name.split('.')
  logging.error('config_name: %s', config_name)
  logging.error('arch_str: %s', arch_str)
  logging.error('navtask_str: %s', navtask_str)
  logging.error('solver_str: %s', solver_str)
  logging.error('mode_str: %s', mode_str)

  args.solver = cc.process_solver_str(solver_str)
  args.navtask = cc.process_navtask_str(navtask_str)

  args = process_arch_str(args, arch_str)
  args.arch.isd_k = args.solver.isd_k

  # Train, test, etc.
  mode, imset = mode_str.split('_')
  args = cc.adjust_args_for_mode(args, mode)
  args.navtask.building_names = args.navtask.dataset.get_split(imset)
  args.control.test_name = '{:s}_on_{:s}'.format(mode, imset)

  # Log the arguments
  logging.error('%s', args)
  return args
