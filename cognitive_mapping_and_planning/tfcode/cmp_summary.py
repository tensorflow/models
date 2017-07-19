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

"""Code for setting up summaries for CMP.
"""

import sys, os, numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope

import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from src import utils 
import src.file_utils as fu
import tfcode.nav_utils as nu 

def _vis_readout_maps(outputs, global_step, output_dir, metric_summary, N):
  # outputs is [gt_map, pred_map]:
  if N >= 0:
    outputs = outputs[:N]
  N = len(outputs)

  plt.set_cmap('jet')
  fig, axes = utils.subplot(plt, (N, outputs[0][0].shape[4]*2), (5,5))
  axes = axes.ravel()[::-1].tolist()
  for i in range(N):
    gt_map, pred_map = outputs[i]
    for j in [0]:
      for k in range(gt_map.shape[4]):
        # Display something like the midpoint of the trajectory.
        id = np.int(gt_map.shape[1]/2)

        ax = axes.pop();
        ax.imshow(gt_map[j,id,:,:,k], origin='lower', interpolation='none',
                  vmin=0., vmax=1.)
        ax.set_axis_off();
        if i == 0: ax.set_title('gt_map')

        ax = axes.pop();
        ax.imshow(pred_map[j,id,:,:,k], origin='lower', interpolation='none',
                  vmin=0., vmax=1.)
        ax.set_axis_off();
        if i == 0: ax.set_title('pred_map')

  file_name = os.path.join(output_dir, 'readout_map_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close(fig)

def _vis(outputs, global_step, output_dir, metric_summary, N):
  # Plot the value map, goal for various maps to see what if the model is
  # learning anything useful.
  #
  # outputs is [values, goals, maps, occupancy, conf].
  #
  if N >= 0:
    outputs = outputs[:N]
  N = len(outputs)

  plt.set_cmap('jet')
  fig, axes = utils.subplot(plt, (N, outputs[0][0].shape[4]*5), (5,5))
  axes = axes.ravel()[::-1].tolist()
  for i in range(N):
    values, goals, maps, occupancy, conf = outputs[i]
    for j in [0]:
      for k in range(values.shape[4]):
        # Display something like the midpoint of the trajectory.
        id = np.int(values.shape[1]/2)

        ax = axes.pop();
        ax.imshow(goals[j,id,:,:,k], origin='lower', interpolation='none')
        ax.set_axis_off();
        if i == 0: ax.set_title('goal')

        ax = axes.pop();
        ax.imshow(occupancy[j,id,:,:,k], origin='lower', interpolation='none')
        ax.set_axis_off();
        if i == 0: ax.set_title('occupancy')

        ax = axes.pop();
        ax.imshow(conf[j,id,:,:,k], origin='lower', interpolation='none',
                  vmin=0., vmax=1.)
        ax.set_axis_off();
        if i == 0: ax.set_title('conf')

        ax = axes.pop();
        ax.imshow(values[j,id,:,:,k], origin='lower', interpolation='none')
        ax.set_axis_off();
        if i == 0: ax.set_title('value')

        ax = axes.pop();
        ax.imshow(maps[j,id,:,:,k], origin='lower', interpolation='none')
        ax.set_axis_off();
        if i == 0: ax.set_title('incr map')

  file_name = os.path.join(output_dir, 'value_vis_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close(fig)

def _summary_vis(m, batch_size, num_steps, arop_full_summary_iters):
  arop = []; arop_summary_iters = []; arop_eval_fns = [];
  vis_value_ops = []; vis_goal_ops = []; vis_map_ops = []; 
  vis_occupancy_ops = []; vis_conf_ops = [];
  for i, val_op in enumerate(m.value_ops):
    vis_value_op = tf.reduce_mean(tf.abs(val_op), axis=3, keep_dims=True)
    vis_value_ops.append(vis_value_op)
    
    vis_occupancy_op = tf.reduce_mean(tf.abs(m.occupancys[i]), 3, True)
    vis_occupancy_ops.append(vis_occupancy_op)
    
    vis_conf_op = tf.reduce_max(tf.abs(m.confs[i]), axis=3, keep_dims=True)
    vis_conf_ops.append(vis_conf_op)
    
    ego_goal_imgs_i_op = m.input_tensors['step']['ego_goal_imgs_{:d}'.format(i)]
    vis_goal_op = tf.reduce_max(ego_goal_imgs_i_op, 4, True)
    vis_goal_ops.append(vis_goal_op)
    
    vis_map_op = tf.reduce_mean(tf.abs(m.ego_map_ops[i]), 4, True)
    vis_map_ops.append(vis_map_op)

  vis_goal_ops = tf.concat(vis_goal_ops, 4)
  vis_map_ops = tf.concat(vis_map_ops, 4)
  vis_value_ops = tf.concat(vis_value_ops, 3)
  vis_occupancy_ops = tf.concat(vis_occupancy_ops, 3)
  vis_conf_ops = tf.concat(vis_conf_ops, 3)

  sh = tf.unstack(tf.shape(vis_value_ops))[1:]
  vis_value_ops = tf.reshape(vis_value_ops, shape=[batch_size, -1] + sh)

  sh = tf.unstack(tf.shape(vis_conf_ops))[1:]
  vis_conf_ops = tf.reshape(vis_conf_ops, shape=[batch_size, -1] + sh)

  sh = tf.unstack(tf.shape(vis_occupancy_ops))[1:]
  vis_occupancy_ops = tf.reshape(vis_occupancy_ops, shape=[batch_size,-1] + sh)

  # Save memory, only return time steps that need to be visualized, factor of
  # 32 CPU memory saving.
  id = np.int(num_steps/2)
  vis_goal_ops = tf.expand_dims(vis_goal_ops[:,id,:,:,:], axis=1)
  vis_map_ops = tf.expand_dims(vis_map_ops[:,id,:,:,:], axis=1)
  vis_value_ops = tf.expand_dims(vis_value_ops[:,id,:,:,:], axis=1)
  vis_conf_ops = tf.expand_dims(vis_conf_ops[:,id,:,:,:], axis=1)
  vis_occupancy_ops = tf.expand_dims(vis_occupancy_ops[:,id,:,:,:], axis=1)

  arop += [[vis_value_ops, vis_goal_ops, vis_map_ops, vis_occupancy_ops,
            vis_conf_ops]]
  arop_summary_iters += [arop_full_summary_iters]
  arop_eval_fns += [_vis]
  return arop, arop_summary_iters, arop_eval_fns

def _summary_readout_maps(m, num_steps, arop_full_summary_iters):
  arop = []; arop_summary_iters = []; arop_eval_fns = [];
  id = np.int(num_steps-1)
  vis_readout_maps_gt = m.readout_maps_gt
  vis_readout_maps_prob = tf.reshape(m.readout_maps_probs,
                                     shape=tf.shape(vis_readout_maps_gt))
  vis_readout_maps_gt = tf.expand_dims(vis_readout_maps_gt[:,id,:,:,:], 1)
  vis_readout_maps_prob = tf.expand_dims(vis_readout_maps_prob[:,id,:,:,:], 1)
  arop += [[vis_readout_maps_gt, vis_readout_maps_prob]]
  arop_summary_iters += [arop_full_summary_iters]
  arop_eval_fns += [_vis_readout_maps]
  return arop, arop_summary_iters, arop_eval_fns

def _add_summaries(m, args, summary_mode, arop_full_summary_iters):
  task_params = args.navtask.task_params
  
  summarize_ops = [m.lr_op, m.global_step_op, m.sample_gt_prob_op] + \
      m.loss_ops + m.acc_ops
  summarize_names = ['lr', 'global_step', 'sample_gt_prob_op'] + \
      m.loss_ops_names + ['acc_{:d}'.format(i) for i in range(len(m.acc_ops))]
  to_aggregate = [0, 0, 0] + [1]*len(m.loss_ops_names) + [1]*len(m.acc_ops)

  scope_name = 'summary'
  with tf.name_scope(scope_name):
    s_ops = nu.add_default_summaries(summary_mode, arop_full_summary_iters,
                                     summarize_ops, summarize_names,
                                     to_aggregate, m.action_prob_op,
                                     m.input_tensors, scope_name=scope_name)
    if summary_mode == 'val':
      arop, arop_summary_iters, arop_eval_fns = _summary_vis(
          m, task_params.batch_size, task_params.num_steps,
          arop_full_summary_iters)
      s_ops.additional_return_ops += arop
      s_ops.arop_summary_iters += arop_summary_iters
      s_ops.arop_eval_fns += arop_eval_fns
      
      if args.arch.readout_maps:
        arop, arop_summary_iters, arop_eval_fns = _summary_readout_maps(
            m, task_params.num_steps, arop_full_summary_iters)
        s_ops.additional_return_ops += arop
        s_ops.arop_summary_iters += arop_summary_iters
        s_ops.arop_eval_fns += arop_eval_fns
  
  return s_ops
