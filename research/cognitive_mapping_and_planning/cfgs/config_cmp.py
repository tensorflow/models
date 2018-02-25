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

import os, sys
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import logging
import src.utils as utils
import cfgs.config_common as cc


import tensorflow as tf


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

def get_default_cmp_args():
  batch_norm_param = {'center': True, 'scale': True,
                      'activation_fn':tf.nn.relu}

  mapper_arch_args = utils.Foo(
      dim_reduce_neurons=64,
      fc_neurons=[1024, 1024],
      fc_out_size=8,
      fc_out_neurons=64,
      encoder='resnet_v2_50',
      deconv_neurons=[64, 32, 16, 8, 4, 2],
      deconv_strides=[2, 2, 2, 2, 2, 2],
      deconv_layers_per_block=2,
      deconv_kernel_size=4,
      fc_dropout=0.5,
      combine_type='wt_avg_logits',
      batch_norm_param=batch_norm_param)

  readout_maps_arch_args = utils.Foo(
      num_neurons=[],
      strides=[],
      kernel_size=None,
      layers_per_block=None)

  arch_args = utils.Foo(
      vin_val_neurons=8, vin_action_neurons=8, vin_ks=3, vin_share_wts=False,
      pred_neurons=[64, 64], pred_batch_norm_param=batch_norm_param,
      conv_on_value_map=0, fr_neurons=16, fr_ver='v2', fr_inside_neurons=64,
      fr_stride=1, crop_remove_each=30, value_crop_size=4,
      action_sample_type='sample', action_sample_combine_type='one_or_other',
      sample_gt_prob_type='inverse_sigmoid_decay', dagger_sample_bn_false=True,
      vin_num_iters=36, isd_k=750., use_agent_loc=False, multi_scale=True,
      readout_maps=False, rom_arch=readout_maps_arch_args)

  return arch_args, mapper_arch_args

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['var1', 'var2', 'var3']
  ks = ks[:len(vals)]
  
  # Exp Ver.
  if len(vals) == 0: ks.append('var1'); vals.append('v0')
  # custom arch.
  if len(vals) == 1: ks.append('var2'); vals.append('')
  # map scape for projection baseline.
  if len(vals) == 2: ks.append('var3'); vals.append('fr2')

  assert(len(vals) == 3)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch, args.mapper_arch = get_default_cmp_args()

  arch_vars = get_arch_vars(arch_str)

  args.navtask.task_params.outputs.ego_maps = True
  args.navtask.task_params.outputs.ego_goal_imgs = True
  args.navtask.task_params.outputs.egomotion = True
  args.navtask.task_params.toy_problem = False

  if arch_vars.var1 == 'lmap':
    args = process_arch_learned_map(args, arch_vars)

  elif arch_vars.var1 == 'pmap':
    args = process_arch_projected_map(args, arch_vars)

  else:
    logging.fatal('arch_vars.var1 should be lmap or pmap, but is %s', arch_vars.var1)
    assert(False)

  return args

def process_arch_learned_map(args, arch_vars):
  # Multiscale vision based system.
  args.navtask.task_params.input_type = 'vision'
  args.navtask.task_params.outputs.images = True
  
  if args.navtask.camera_param.modalities[0] == 'rgb':
    args.solver.pretrained_path = rgb_resnet_v2_50_path
  elif args.navtask.camera_param.modalities[0] == 'depth':
    args.solver.pretrained_path = d_resnet_v2_50_path

  if arch_vars.var2 == 'Ssc':
    sc = 1./args.navtask.task_params.step_size
    args.arch.vin_num_iters = 40
    args.navtask.task_params.map_scales = [sc]
    max_dist = args.navtask.task_params.max_dist * \
        args.navtask.task_params.num_goals
    args.navtask.task_params.map_crop_sizes = [2*max_dist]

    args.arch.fr_stride = 1
    args.arch.vin_action_neurons = 8
    args.arch.vin_val_neurons = 3
    args.arch.fr_inside_neurons = 32

    args.mapper_arch.pad_map_with_zeros_each = [24]
    args.mapper_arch.deconv_neurons = [64, 32, 16]
    args.mapper_arch.deconv_strides = [1, 2, 1]

  elif (arch_vars.var2 == 'Msc' or arch_vars.var2 == 'MscROMms' or
        arch_vars.var2 == 'MscROMss' or arch_vars.var2 == 'MscNoVin'):
    # Code for multi-scale planner.
    args.arch.vin_num_iters = 8
    args.arch.crop_remove_each = 4
    args.arch.value_crop_size = 8

    sc = 1./args.navtask.task_params.step_size
    max_dist = args.navtask.task_params.max_dist * \
        args.navtask.task_params.num_goals
    n_scales = np.log2(float(max_dist) / float(args.arch.vin_num_iters))
    n_scales = int(np.ceil(n_scales)+1)

    args.navtask.task_params.map_scales = \
        list(sc*(0.5**(np.arange(n_scales))[::-1]))
    args.navtask.task_params.map_crop_sizes = [16 for x in range(n_scales)]

    args.arch.fr_stride = 1
    args.arch.vin_action_neurons = 8
    args.arch.vin_val_neurons = 3
    args.arch.fr_inside_neurons = 32

    args.mapper_arch.pad_map_with_zeros_each = [0 for _ in range(n_scales)]
    args.mapper_arch.deconv_neurons = [64*n_scales, 32*n_scales, 16*n_scales]
    args.mapper_arch.deconv_strides = [1, 2, 1]

    if arch_vars.var2 == 'MscNoVin':
      # No planning version.
      args.arch.fr_stride = [1, 2, 1, 2]
      args.arch.vin_action_neurons = None
      args.arch.vin_val_neurons = 16
      args.arch.fr_inside_neurons = 32

      args.arch.crop_remove_each = 0
      args.arch.value_crop_size = 4
      args.arch.vin_num_iters = 0

    elif arch_vars.var2 == 'MscROMms' or arch_vars.var2 == 'MscROMss':
      # Code with read outs, MscROMms flattens and reads out,
      # MscROMss does not flatten and produces output at multiple scales.
      args.navtask.task_params.outputs.readout_maps = True
      args.navtask.task_params.map_resize_method = 'antialiasing'
      args.arch.readout_maps = True

      if arch_vars.var2 == 'MscROMms':
        args.arch.rom_arch.num_neurons = [64, 1]
        args.arch.rom_arch.kernel_size = 4
        args.arch.rom_arch.strides = [2,2]
        args.arch.rom_arch.layers_per_block = 2

        args.navtask.task_params.readout_maps_crop_sizes = [64]
        args.navtask.task_params.readout_maps_scales = [sc]

      elif arch_vars.var2 == 'MscROMss':
        args.arch.rom_arch.num_neurons = \
            [64, len(args.navtask.task_params.map_scales)]
        args.arch.rom_arch.kernel_size = 4
        args.arch.rom_arch.strides = [1,1]
        args.arch.rom_arch.layers_per_block = 1

        args.navtask.task_params.readout_maps_crop_sizes = \
            args.navtask.task_params.map_crop_sizes
        args.navtask.task_params.readout_maps_scales = \
            args.navtask.task_params.map_scales

  else:
    logging.fatal('arch_vars.var2 not one of Msc, MscROMms, MscROMss, MscNoVin.')
    assert(False)

  map_channels = args.mapper_arch.deconv_neurons[-1] / \
    (2*len(args.navtask.task_params.map_scales))
  args.navtask.task_params.map_channels = map_channels
  
  return args

def process_arch_projected_map(args, arch_vars):
  # Single scale vision based system which does not use a mapper but instead
  # uses an analytically estimated map.
  ds = int(arch_vars.var3[2])
  args.navtask.task_params.input_type = 'analytical_counts'
  args.navtask.task_params.outputs.analytical_counts = True

  assert(args.navtask.task_params.modalities[0] == 'depth')
  args.navtask.camera_param.img_channels = None

  analytical_counts = utils.Foo(map_sizes=[512/ds],
                                xy_resolution=[5.*ds],
                                z_bins=[[-10, 10, 150, 200]],
                                non_linearity=[arch_vars.var2])
  args.navtask.task_params.analytical_counts = analytical_counts

  sc = 1./ds
  args.arch.vin_num_iters = 36
  args.navtask.task_params.map_scales = [sc]
  args.navtask.task_params.map_crop_sizes = [512/ds]

  args.arch.fr_stride = [1,2]
  args.arch.vin_action_neurons = 8
  args.arch.vin_val_neurons = 3
  args.arch.fr_inside_neurons = 32

  map_channels = len(analytical_counts.z_bins[0]) + 1
  args.navtask.task_params.map_channels = map_channels
  args.solver.freeze_conv = False

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
