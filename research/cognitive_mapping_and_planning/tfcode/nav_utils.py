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

"""Various losses for training navigation agents.

Defines various loss functions for navigation agents, 
compute_losses_multi_or.
"""

import os, numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.training import moving_averages
import logging
from src import utils 
import src.file_utils as fu
from tfcode import tf_utils


def compute_losses_multi_or(logits, actions_one_hot, weights=None,
                            num_actions=-1, data_loss_wt=1., reg_loss_wt=1.,
                            ewma_decay=0.99, reg_loss_op=None):
  assert(num_actions > 0), 'num_actions must be specified and must be > 0.'
  
  with tf.name_scope('loss'):
    if weights is None:
      weight = tf.ones_like(actions_one_hot, dtype=tf.float32, name='weight')
    
    actions_one_hot = tf.cast(tf.reshape(actions_one_hot, [-1, num_actions],
                                         're_actions_one_hot'), tf.float32)
    weights = tf.reduce_sum(tf.reshape(weights, [-1, num_actions], 're_weight'),
                            reduction_indices=1)
    total = tf.reduce_sum(weights)

    action_prob = tf.nn.softmax(logits)
    action_prob = tf.reduce_sum(tf.multiply(action_prob, actions_one_hot),
                                reduction_indices=1)
    example_loss = -tf.log(tf.maximum(tf.constant(1e-4), action_prob))

    data_loss_op = tf.reduce_sum(example_loss * weights) / total
    if reg_loss_op is None:
      if reg_loss_wt > 0:
        reg_loss_op = tf.add_n(tf.losses.get_regularization_losses())
      else:
        reg_loss_op = tf.constant(0.)
    
    if reg_loss_wt > 0:
      total_loss_op = data_loss_wt*data_loss_op + reg_loss_wt*reg_loss_op 
    else:
      total_loss_op = data_loss_wt*data_loss_op

    is_correct = tf.cast(tf.greater(action_prob, 0.5, name='pred_class'), tf.float32)
    acc_op = tf.reduce_sum(is_correct*weights) / total

    ewma_acc_op = moving_averages.weighted_moving_average(
        acc_op, ewma_decay, weight=total, name='ewma_acc')

    acc_ops = [ewma_acc_op]

  return reg_loss_op, data_loss_op, total_loss_op, acc_ops


def get_repr_from_image(images_reshaped, modalities, data_augment, encoder,
                        freeze_conv, wt_decay, is_training):
  # Pass image through lots of convolutional layers, to obtain pool5
  if modalities == ['rgb']:
    with tf.name_scope('pre_rgb'):
      x = (images_reshaped + 128.) / 255. # Convert to brightness between 0 and 1.
      if data_augment.relight and is_training:
        x = tf_utils.distort_image(x, fast_mode=data_augment.relight_fast)
      x = (x-0.5)*2.0
    scope_name = encoder
  elif modalities == ['depth']:
    with tf.name_scope('pre_d'):
      d_image = images_reshaped
      x = 2*(d_image[...,0] - 80.0)/100.0
      y = d_image[...,1]
      d_image = tf.concat([tf.expand_dims(x, -1), tf.expand_dims(y, -1)], 3)
      x = d_image
    scope_name = 'd_'+encoder

  resnet_is_training = is_training and (not freeze_conv)
  with slim.arg_scope(resnet_v2.resnet_utils.resnet_arg_scope(resnet_is_training)):
    fn = getattr(tf_utils, encoder)
    x, end_points = fn(x, num_classes=None, global_pool=False,
                       output_stride=None, reuse=None,
                       scope=scope_name)
  vars_ = slim.get_variables_to_restore()

  conv_feat = x
  return conv_feat, vars_

def default_train_step_kwargs(m, obj, logdir, rng_seed, is_chief, num_steps,
                              iters, train_display_interval,
                              dagger_sample_bn_false):
  train_step_kwargs = {}
  train_step_kwargs['obj'] = obj 
  train_step_kwargs['m'] = m
  
  # rng_data has 2 independent rngs, one for sampling episodes and one for
  # sampling perturbs (so that we can make results reproducible.
  train_step_kwargs['rng_data'] = [np.random.RandomState(rng_seed), 
                                   np.random.RandomState(rng_seed)]
  train_step_kwargs['rng_action'] = np.random.RandomState(rng_seed)
  if is_chief: 
    train_step_kwargs['writer'] = tf.summary.FileWriter(logdir) #, m.tf_graph)
  else:
    train_step_kwargs['writer'] = None
  train_step_kwargs['iters'] = iters
  train_step_kwargs['train_display_interval'] = train_display_interval 
  train_step_kwargs['num_steps'] = num_steps
  train_step_kwargs['logdir'] = logdir
  train_step_kwargs['dagger_sample_bn_false'] = dagger_sample_bn_false 
  return train_step_kwargs

# Utilities for visualizing and analysing validation output.
def save_d_at_t(outputs, global_step, output_dir, metric_summary, N):
  """Save distance to goal at all time steps.
  
  Args:
    outputs        : [gt_dist_to_goal].
    global_step : number of iterations.
    output_dir     : output directory.
    metric_summary : to append scalars to summary.
    N              : number of outputs to process.

  """
  d_at_t = np.concatenate(map(lambda x: x[0][:,:,0]*1, outputs), axis=0)
  fig, axes = utils.subplot(plt, (1,1), (5,5))
  axes.plot(np.arange(d_at_t.shape[1]), np.mean(d_at_t, axis=0), 'r.')
  axes.set_xlabel('time step')
  axes.set_ylabel('dist to next goal')
  axes.grid('on')
  file_name = os.path.join(output_dir, 'dist_at_t_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  file_name = os.path.join(output_dir, 'dist_at_t_{:d}.pkl'.format(global_step))
  utils.save_variables(file_name, [d_at_t], ['d_at_t'], overwrite=True)
  plt.close(fig)
  return None

def save_all(outputs, global_step, output_dir, metric_summary, N):
  """Save numerous statistics.
  
  Args:
    outputs        : [locs, goal_loc, gt_dist_to_goal, node_ids, perturbs]
    global_step    : number of iterations.
    output_dir     : output directory.
    metric_summary : to append scalars to summary.
    N              : number of outputs to process.
  """
  all_locs = np.concatenate(map(lambda x: x[0], outputs), axis=0)
  all_goal_locs = np.concatenate(map(lambda x: x[1], outputs), axis=0)
  all_d_at_t = np.concatenate(map(lambda x: x[2][:,:,0]*1, outputs), axis=0)
  all_node_ids = np.concatenate(map(lambda x: x[3], outputs), axis=0)
  all_perturbs = np.concatenate(map(lambda x: x[4], outputs), axis=0)
  
  file_name = os.path.join(output_dir, 'all_locs_at_t_{:d}.pkl'.format(global_step))
  vars = [all_locs, all_goal_locs, all_d_at_t, all_node_ids, all_perturbs]
  var_names = ['all_locs', 'all_goal_locs', 'all_d_at_t', 'all_node_ids', 'all_perturbs']
  utils.save_variables(file_name, vars, var_names,  overwrite=True)
  return None

def eval_ap(outputs, global_step, output_dir, metric_summary, N, num_classes=4):
  """Processes the collected outputs to compute AP for action prediction.
  
  Args:
    outputs        : [logits, labels]
    global_step    : global_step.
    output_dir     : where to store results.
    metric_summary : summary object to add summaries to.
    N              : number of outputs to process.
    num_classes    : number of classes to compute AP over, and to reshape tensors.
  """
  if N >= 0:
    outputs = outputs[:N]
  logits = np.concatenate(map(lambda x: x[0], outputs), axis=0).reshape((-1, num_classes))
  labels = np.concatenate(map(lambda x: x[1], outputs), axis=0).reshape((-1, num_classes))
  aps = []
  for i in range(logits.shape[1]):
    ap, rec, prec = utils.calc_pr(labels[:,i], logits[:,i])
    ap = ap[0]
    tf_utils.add_value_to_summary(metric_summary, 'aps/ap_{:d}: '.format(i), ap)
    aps.append(ap)
  return aps

def eval_dist(outputs, global_step, output_dir, metric_summary, N):
  """Processes the collected outputs during validation to 
  1. Plot the distance over time curve.
  2. Compute mean and median distances.
  3. Plots histogram of end distances.
  
  Args:
    outputs        : [locs, goal_loc, gt_dist_to_goal].
    global_step    : global_step.
    output_dir     : where to store results.
    metric_summary : summary object to add summaries to.
    N              : number of outputs to process.
  """
  SUCCESS_THRESH = 3
  if N >= 0:
    outputs = outputs[:N]
  
  # Plot distance at time t.
  d_at_t = []
  for i in range(len(outputs)):
    locs, goal_loc, gt_dist_to_goal = outputs[i]
    d_at_t.append(gt_dist_to_goal[:,:,0]*1)

  # Plot the distance.
  fig, axes = utils.subplot(plt, (1,1), (5,5))
  d_at_t = np.concatenate(d_at_t, axis=0)
  axes.plot(np.arange(d_at_t.shape[1]), np.mean(d_at_t, axis=0), 'r.')
  axes.set_xlabel('time step')
  axes.set_ylabel('dist to next goal')
  axes.grid('on')
  file_name = os.path.join(output_dir, 'dist_at_t_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  file_name = os.path.join(output_dir, 'dist_at_t_{:d}.pkl'.format(global_step))
  utils.save_variables(file_name, [d_at_t], ['d_at_t'], overwrite=True)
  plt.close(fig)

  # Plot the trajectories and the init_distance and final distance.
  d_inits = []
  d_ends = []
  for i in range(len(outputs)):
    locs, goal_loc, gt_dist_to_goal = outputs[i]
    d_inits.append(gt_dist_to_goal[:,0,0]*1)
    d_ends.append(gt_dist_to_goal[:,-1,0]*1)

  # Plot the distance.
  fig, axes = utils.subplot(plt, (1,1), (5,5))
  d_inits = np.concatenate(d_inits, axis=0)
  d_ends = np.concatenate(d_ends, axis=0)
  axes.plot(d_inits+np.random.rand(*(d_inits.shape))-0.5,
            d_ends+np.random.rand(*(d_ends.shape))-0.5, '.', mec='red', mew=1.0)
  axes.set_xlabel('init dist'); axes.set_ylabel('final dist'); 
  axes.grid('on'); axes.axis('equal');
  title_str = 'mean: {:0.1f}, 50: {:0.1f}, 75: {:0.2f}, s: {:0.1f}'
  title_str = title_str.format(
      np.mean(d_ends), np.median(d_ends), np.percentile(d_ends, q=75),
      100*(np.mean(d_ends <= SUCCESS_THRESH)))
  axes.set_title(title_str)
  file_name = os.path.join(output_dir, 'dist_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)

  file_name = os.path.join(output_dir, 'dist_{:d}.pkl'.format(global_step))
  utils.save_variables(file_name, [d_inits, d_ends], ['d_inits', 'd_ends'],
                       overwrite=True)
  plt.close(fig)

  # Plot the histogram of the end_distance.
  with plt.style.context('seaborn-white'):
    d_ends_ = np.sort(d_ends)
    d_inits_ = np.sort(d_inits)
    leg = [];
    fig, ax = utils.subplot(plt, (1,1), (5,5))
    ax.grid('on')
    ax.set_xlabel('Distance from goal'); ax.xaxis.label.set_fontsize(16);
    ax.set_ylabel('Fraction of data'); ax.yaxis.label.set_fontsize(16);
    ax.plot(d_ends_, np.arange(d_ends_.size)*1./d_ends_.size, 'r')
    ax.plot(d_inits_, np.arange(d_inits_.size)*1./d_inits_.size, 'k')
    leg.append('Final'); leg.append('Init');
    ax.legend(leg, fontsize='x-large');
    ax.set_axis_on()
    title_str = 'mean: {:0.1f}, 50: {:0.1f}, 75: {:0.2f}, s: {:0.1f}'
    title_str = title_str.format(
        np.mean(d_ends), np.median(d_ends), np.percentile(d_ends, q=75),
        100*(np.mean(d_ends <= SUCCESS_THRESH)))
    ax.set_title(title_str)
    file_name = os.path.join(output_dir, 'dist_hist_{:d}.png'.format(global_step))
    with fu.fopen(file_name, 'w') as f:
      fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  
  # Log distance metrics.
  tf_utils.add_value_to_summary(metric_summary, 'dists/success_init: ',
                                100*(np.mean(d_inits <= SUCCESS_THRESH)))
  tf_utils.add_value_to_summary(metric_summary, 'dists/success_end: ',
                                100*(np.mean(d_ends <= SUCCESS_THRESH)))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_init (75): ',
                                np.percentile(d_inits, q=75))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_end (75): ',
                                np.percentile(d_ends, q=75))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_init (median): ',
                                np.median(d_inits))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_end (median): ',
                                np.median(d_ends))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_init (mean): ',
                                np.mean(d_inits))
  tf_utils.add_value_to_summary(metric_summary, 'dists/dist_end (mean): ',
                                np.mean(d_ends))
  return np.median(d_inits), np.median(d_ends), np.mean(d_inits), np.mean(d_ends), \
      np.percentile(d_inits, q=75), np.percentile(d_ends, q=75), \
      100*(np.mean(d_inits) <= SUCCESS_THRESH), 100*(np.mean(d_ends) <= SUCCESS_THRESH)

def plot_trajectories(outputs, global_step, output_dir, metric_summary, N):
  """Processes the collected outputs during validation to plot the trajectories
  in the top view.
  
  Args:
    outputs        : [locs, orig_maps, goal_loc].
    global_step    : global_step.
    output_dir     : where to store results.
    metric_summary : summary object to add summaries to.
    N              : number of outputs to process.
  """
  if N >= 0:
    outputs = outputs[:N]
  N = len(outputs)

  plt.set_cmap('gray')
  fig, axes = utils.subplot(plt, (N, outputs[0][1].shape[0]), (5,5))
  axes = axes.ravel()[::-1].tolist()
  for i in range(N):
    locs, orig_maps, goal_loc = outputs[i]
    is_semantic = np.isnan(goal_loc[0,0,1])
    for j in range(orig_maps.shape[0]):
      ax = axes.pop();
      ax.plot(locs[j,0,0], locs[j,0,1], 'ys')
      # Plot one by one, so that they come in different colors.
      for k in range(goal_loc.shape[1]):
        if not is_semantic:
          ax.plot(goal_loc[j,k,0], goal_loc[j,k,1], 's')
      if False:
        ax.plot(locs[j,:,0], locs[j,:,1], 'r.', ms=3)
        ax.imshow(orig_maps[j,0,:,:,0], origin='lower')
        ax.set_axis_off();
      else:
        ax.scatter(locs[j,:,0], locs[j,:,1], c=np.arange(locs.shape[1]),
                   cmap='jet', s=10, lw=0)
        ax.imshow(orig_maps[j,0,:,:,0], origin='lower', vmin=-1.0, vmax=2.0)
        if not is_semantic:
          xymin = np.minimum(np.min(goal_loc[j,:,:], axis=0), np.min(locs[j,:,:], axis=0))
          xymax = np.maximum(np.max(goal_loc[j,:,:], axis=0), np.max(locs[j,:,:], axis=0))
        else:
          xymin = np.min(locs[j,:,:], axis=0)
          xymax = np.max(locs[j,:,:], axis=0)
        xy1 = (xymax+xymin)/2. - np.maximum(np.max(xymax-xymin), 12)
        xy2 = (xymax+xymin)/2. + np.maximum(np.max(xymax-xymin), 12)
        ax.set_xlim([xy1[0], xy2[0]])
        ax.set_ylim([xy1[1], xy2[1]])
        ax.set_axis_off()
  file_name = os.path.join(output_dir, 'trajectory_{:d}.png'.format(global_step))
  with fu.fopen(file_name, 'w') as f:
    fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close(fig)
  return None

def add_default_summaries(mode, arop_full_summary_iters, summarize_ops,
                          summarize_names, to_aggregate, action_prob_op,
                          input_tensors, scope_name):
  assert(mode == 'train' or mode == 'val' or mode == 'test'), \
    'add_default_summaries mode is neither train or val or test.'
  
  s_ops = tf_utils.get_default_summary_ops()
  
  if mode == 'train':
    s_ops.summary_ops, s_ops.print_summary_ops, additional_return_ops, \
    arop_summary_iters, arop_eval_fns = tf_utils.simple_summaries(
            summarize_ops, summarize_names, mode, to_aggregate=False,
            scope_name=scope_name)
    s_ops.additional_return_ops += additional_return_ops
    s_ops.arop_summary_iters += arop_summary_iters
    s_ops.arop_eval_fns += arop_eval_fns
  elif mode == 'val':
    s_ops.summary_ops, s_ops.print_summary_ops, additional_return_ops, \
    arop_summary_iters, arop_eval_fns = tf_utils.simple_summaries(
            summarize_ops, summarize_names, mode, to_aggregate=to_aggregate,
            scope_name=scope_name)
    s_ops.additional_return_ops += additional_return_ops
    s_ops.arop_summary_iters += arop_summary_iters
    s_ops.arop_eval_fns += arop_eval_fns
  
  elif mode == 'test':
    s_ops.summary_ops, s_ops.print_summary_ops, additional_return_ops, \
    arop_summary_iters, arop_eval_fns = tf_utils.simple_summaries(
        [], [], mode, to_aggregate=[], scope_name=scope_name)
    s_ops.additional_return_ops += additional_return_ops
    s_ops.arop_summary_iters += arop_summary_iters
    s_ops.arop_eval_fns += arop_eval_fns

  
  if mode == 'val':
    arop = s_ops.additional_return_ops
    arop += [[action_prob_op, input_tensors['train']['action']]]
    arop += [[input_tensors['step']['loc_on_map'],
              input_tensors['common']['goal_loc'],
              input_tensors['step']['gt_dist_to_goal']]]
    arop += [[input_tensors['step']['loc_on_map'],
              input_tensors['common']['orig_maps'],
              input_tensors['common']['goal_loc']]]
    s_ops.arop_summary_iters += [-1, arop_full_summary_iters,
                                 arop_full_summary_iters]
    s_ops.arop_eval_fns += [eval_ap, eval_dist, plot_trajectories]
  
  elif mode == 'test':
    arop = s_ops.additional_return_ops
    arop += [[input_tensors['step']['loc_on_map'],
              input_tensors['common']['goal_loc'],
              input_tensors['step']['gt_dist_to_goal']]]
    arop += [[input_tensors['step']['gt_dist_to_goal']]]
    arop += [[input_tensors['step']['loc_on_map'],
              input_tensors['common']['goal_loc'],
              input_tensors['step']['gt_dist_to_goal'],
              input_tensors['step']['node_ids'],
              input_tensors['step']['perturbs']]]
    arop += [[input_tensors['step']['loc_on_map'],
              input_tensors['common']['orig_maps'],
              input_tensors['common']['goal_loc']]]
    s_ops.arop_summary_iters += [-1, -1, -1, arop_full_summary_iters]
    s_ops.arop_eval_fns += [eval_dist, save_d_at_t, save_all,
                            plot_trajectories]
  return s_ops


