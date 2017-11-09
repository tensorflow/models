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

import numpy as np


import tensorflow as tf

from tensorflow.contrib import slim

import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from src import utils 
import src.file_utils as fu
import tfcode.nav_utils as nu 
from tfcode import tf_utils

setup_train_step_kwargs = nu.default_train_step_kwargs
compute_losses_multi_or = nu.compute_losses_multi_or
get_repr_from_image = nu.get_repr_from_image

_save_d_at_t = nu.save_d_at_t
_save_all = nu.save_all
_eval_ap = nu.eval_ap
_eval_dist = nu.eval_dist
_plot_trajectories = nu.plot_trajectories

def lstm_online(cell_fn, num_steps, inputs, state, varscope):
  # inputs is B x num_steps x C, C channels.
  # state is 2 tuple with B x 1 x C1, B x 1 x C2 
  # Output state is always B x 1 x C
  inputs = tf.unstack(inputs, axis=1, num=num_steps)
  state = tf.unstack(state, axis=1, num=1)[0]
  outputs = [] 
  
  if num_steps > 1: 
    varscope.reuse_variables()
  
  for s in range(num_steps):
    output, state = cell_fn(inputs[s], state)
    outputs.append(output)
  outputs = tf.stack(outputs, axis=1)
  state = tf.stack([state], axis=1)
  return outputs, state

def _inputs(problem, lstm_states, lstm_state_dims):
  # Set up inputs.
  with tf.name_scope('inputs'):
    n_views = problem.n_views

    inputs = []
    inputs.append(('orig_maps', tf.float32,
                   (problem.batch_size, 1, None, None, 1)))
    inputs.append(('goal_loc', tf.float32,
                   (problem.batch_size, problem.num_goals, 2)))

    # For initing LSTM.
    inputs.append(('rel_goal_loc_at_start', tf.float32,
                   (problem.batch_size, problem.num_goals,
                    problem.rel_goal_loc_dim)))
    common_input_data, _ = tf_utils.setup_inputs(inputs)

    inputs = []
    inputs.append(('imgs', tf.float32, (problem.batch_size, None, n_views,
                                        problem.img_height, problem.img_width,
                                        problem.img_channels)))
    # Goal location as a tuple of delta location and delta theta.
    inputs.append(('rel_goal_loc', tf.float32, (problem.batch_size, None,
                                                problem.rel_goal_loc_dim)))
    if problem.outputs.visit_count:
      inputs.append(('visit_count', tf.int32, (problem.batch_size, None, 1)))
      inputs.append(('last_visit', tf.int32, (problem.batch_size, None, 1)))

    for i, (state, dim) in enumerate(zip(lstm_states, lstm_state_dims)):
      inputs.append((state, tf.float32, (problem.batch_size, 1, dim)))

    if problem.outputs.egomotion:
      inputs.append(('incremental_locs', tf.float32,
                     (problem.batch_size, None, 2)))
      inputs.append(('incremental_thetas', tf.float32,
                     (problem.batch_size, None, 1)))

    inputs.append(('step_number', tf.int32, (1, None, 1)))
    inputs.append(('node_ids', tf.int32, (problem.batch_size, None,
                                          problem.node_ids_dim)))
    inputs.append(('perturbs', tf.float32, (problem.batch_size, None,
                                            problem.perturbs_dim)))

    # For plotting result plots
    inputs.append(('loc_on_map', tf.float32, (problem.batch_size, None, 2)))
    inputs.append(('gt_dist_to_goal', tf.float32, (problem.batch_size, None, 1)))
    step_input_data, _ = tf_utils.setup_inputs(inputs)

    inputs = []
    inputs.append(('executed_actions', tf.int32, (problem.batch_size, None)))
    inputs.append(('rewards', tf.float32, (problem.batch_size, None)))
    inputs.append(('action_sample_wts', tf.float32, (problem.batch_size, None)))
    inputs.append(('action', tf.int32, (problem.batch_size, None,
                                        problem.num_actions)))
    train_data, _ = tf_utils.setup_inputs(inputs)
    train_data.update(step_input_data)
    train_data.update(common_input_data)
  return common_input_data, step_input_data, train_data


def _add_summaries(m, summary_mode, arop_full_summary_iters):
  summarize_ops = [m.lr_op, m.global_step_op, m.sample_gt_prob_op,
                   m.total_loss_op, m.data_loss_op, m.reg_loss_op] + m.acc_ops
  summarize_names = ['lr', 'global_step', 'sample_gt_prob_op', 'total_loss',
                     'data_loss', 'reg_loss'] + \
                    ['acc_{:d}'.format(i) for i in range(len(m.acc_ops))]
  to_aggregate = [0, 0, 0, 1, 1, 1] + [1]*len(m.acc_ops)

  scope_name = 'summary'
  with tf.name_scope(scope_name):
    s_ops = nu.add_default_summaries(summary_mode, arop_full_summary_iters,
                                     summarize_ops, summarize_names,
                                     to_aggregate, m.action_prob_op,
                                     m.input_tensors, scope_name=scope_name)
    m.summary_ops = {summary_mode: s_ops}

def visit_count_fc(visit_count, last_visit, embed_neurons, wt_decay, fc_dropout):
  with tf.variable_scope('embed_visit_count'):
    visit_count = tf.reshape(visit_count, shape=[-1])
    last_visit = tf.reshape(last_visit, shape=[-1])
    
    visit_count = tf.clip_by_value(visit_count, clip_value_min=-1,
                                   clip_value_max=15)
    last_visit = tf.clip_by_value(last_visit, clip_value_min=-1,
                                   clip_value_max=15)
    visit_count = tf.one_hot(visit_count, depth=16, axis=1, dtype=tf.float32,
                             on_value=10., off_value=0.)
    last_visit = tf.one_hot(last_visit, depth=16, axis=1, dtype=tf.float32,
                             on_value=10., off_value=0.)
    f = tf.concat([visit_count, last_visit], 1)
    x, _ = tf_utils.fc_network(
        f, neurons=embed_neurons, wt_decay=wt_decay, name='visit_count_embed',
        offset=0, batch_norm_param=None, dropout_ratio=fc_dropout,
        is_training=is_training)
  return x

def lstm_setup(name, x, batch_size, is_single_step, lstm_dim, lstm_out,
               num_steps, state_input_op):
  # returns state_name, state_init_op, updated_state_op, out_op 
  with tf.name_scope('reshape_'+name):
    sh = x.get_shape().as_list()
    x = tf.reshape(x, shape=[batch_size, -1, sh[-1]])

  with tf.variable_scope(name) as varscope:
    cell = tf.contrib.rnn.LSTMCell(
      num_units=lstm_dim, forget_bias=1.0, state_is_tuple=False,
      num_proj=lstm_out, use_peepholes=True,
      initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=0),
      cell_clip=None, proj_clip=None)

    sh = [batch_size, 1, lstm_dim+lstm_out]
    state_init_op = tf.constant(0., dtype=tf.float32, shape=sh)

    fn = lambda ns: lstm_online(cell, ns, x, state_input_op, varscope)
    out_op, updated_state_op = tf.cond(is_single_step, lambda: fn(1), lambda:
                                       fn(num_steps))

  return name, state_init_op, updated_state_op, out_op 

def combine_setup(name, combine_type, embed_img, embed_goal, num_img_neuorons=None,
                  num_goal_neurons=None):
  with tf.name_scope(name + '_' + combine_type):
    if combine_type == 'add':
      # Simple concat features from goal and image
      out = embed_img + embed_goal

    elif combine_type == 'multiply':
      # Multiply things together
      re_embed_img = tf.reshape(
          embed_img, shape=[-1, num_img_neuorons / num_goal_neurons,
                            num_goal_neurons])
      re_embed_goal = tf.reshape(embed_goal, shape=[-1, num_goal_neurons, 1])
      x = tf.matmul(re_embed_img, re_embed_goal, transpose_a=False, transpose_b=False)
      out = slim.flatten(x)
    elif combine_type == 'none' or combine_type == 'imgonly':
      out = embed_img
    elif combine_type == 'goalonly':
      out = embed_goal
    else:
      logging.fatal('Undefined combine_type: %s', combine_type)
  return out


def preprocess_egomotion(locs, thetas):
  with tf.name_scope('pre_ego'):
    pre_ego = tf.concat([locs, tf.sin(thetas), tf.cos(thetas)], 2)
    sh = pre_ego.get_shape().as_list()
    pre_ego = tf.reshape(pre_ego, [-1, sh[-1]])
  return pre_ego

def setup_to_run(m, args, is_training, batch_norm_is_training, summary_mode):
  # Set up the model.
  tf.set_random_seed(args.solver.seed)
  task_params = args.navtask.task_params
  num_steps = task_params.num_steps
  num_goals = task_params.num_goals
  num_actions = task_params.num_actions
  num_actions_ = num_actions

  n_views = task_params.n_views

  batch_norm_is_training_op = \
      tf.placeholder_with_default(batch_norm_is_training, shape=[],
                                  name='batch_norm_is_training_op') 
  # Setup the inputs
  m.input_tensors = {}
  lstm_states = []; lstm_state_dims = [];
  state_names = []; updated_state_ops = []; init_state_ops = [];
  if args.arch.lstm_output:
    lstm_states += ['lstm_output']
    lstm_state_dims += [args.arch.lstm_output_dim+task_params.num_actions]
  if args.arch.lstm_ego:
    lstm_states += ['lstm_ego']
    lstm_state_dims += [args.arch.lstm_ego_dim + args.arch.lstm_ego_out]
    lstm_states += ['lstm_img']
    lstm_state_dims += [args.arch.lstm_img_dim + args.arch.lstm_img_out]
  elif args.arch.lstm_img:
    # An LSTM only on the image
    lstm_states += ['lstm_img']
    lstm_state_dims += [args.arch.lstm_img_dim + args.arch.lstm_img_out]
  else:
    # No LSTMs involved here.
    None

  m.input_tensors['common'], m.input_tensors['step'], m.input_tensors['train'] = \
      _inputs(task_params, lstm_states, lstm_state_dims)

  with tf.name_scope('check_size'):
    is_single_step = tf.equal(tf.unstack(tf.shape(m.input_tensors['step']['imgs']), 
                                        num=6)[1], 1)

  images_reshaped = tf.reshape(m.input_tensors['step']['imgs'], 
      shape=[-1, task_params.img_height, task_params.img_width,
             task_params.img_channels], name='re_image')

  rel_goal_loc_reshaped = tf.reshape(m.input_tensors['step']['rel_goal_loc'], 
      shape=[-1, task_params.rel_goal_loc_dim], name='re_rel_goal_loc')

  x, vars_ = get_repr_from_image(
      images_reshaped, task_params.modalities, task_params.data_augment,
      args.arch.encoder, args.solver.freeze_conv, args.solver.wt_decay,
      is_training)

  # Reshape into nice things so that these can be accumulated over time steps
  # for faster backprop.
  sh_before = x.get_shape().as_list()
  m.encoder_output = tf.reshape(
      x, shape=[task_params.batch_size, -1, n_views] + sh_before[1:])
  x = tf.reshape(m.encoder_output, shape=[-1] + sh_before[1:])

  # Add a layer to reduce dimensions for a fc layer.
  if args.arch.dim_reduce_neurons > 0:
    ks = 1; neurons = args.arch.dim_reduce_neurons;
    init_var = np.sqrt(2.0/(ks**2)/neurons)
    batch_norm_param = args.arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    m.conv_feat = slim.conv2d(
        x, neurons, kernel_size=ks, stride=1, normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_param, padding='SAME', scope='dim_reduce',
        weights_regularizer=slim.l2_regularizer(args.solver.wt_decay),
        weights_initializer=tf.random_normal_initializer(stddev=init_var))
    reshape_conv_feat = slim.flatten(m.conv_feat)
    sh = reshape_conv_feat.get_shape().as_list()
    m.reshape_conv_feat = tf.reshape(reshape_conv_feat, 
                                     shape=[-1, sh[1]*n_views])

  # Restore these from a checkpoint.
  if args.solver.pretrained_path is not None:
    m.init_fn = slim.assign_from_checkpoint_fn(args.solver.pretrained_path,
                                               vars_)
  else:
    m.init_fn = None

  # Hit the goal_location with a bunch of fully connected layers, to embed it
  # into some space.
  with tf.variable_scope('embed_goal'):
    batch_norm_param = args.arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    m.embed_goal, _ = tf_utils.fc_network(
        rel_goal_loc_reshaped, neurons=args.arch.goal_embed_neurons,
        wt_decay=args.solver.wt_decay, name='goal_embed', offset=0,
        batch_norm_param=batch_norm_param, dropout_ratio=args.arch.fc_dropout,
        is_training=is_training)
  
  if args.arch.embed_goal_for_state:
    with tf.variable_scope('embed_goal_for_state'):
      batch_norm_param = args.arch.batch_norm_param
      batch_norm_param['is_training'] = batch_norm_is_training_op
      m.embed_goal_for_state, _ = tf_utils.fc_network(
          m.input_tensors['common']['rel_goal_loc_at_start'][:,0,:],
          neurons=args.arch.goal_embed_neurons, wt_decay=args.solver.wt_decay,
          name='goal_embed', offset=0, batch_norm_param=batch_norm_param,
          dropout_ratio=args.arch.fc_dropout, is_training=is_training)

  # Hit the goal_location with a bunch of fully connected layers, to embed it
  # into some space.
  with tf.variable_scope('embed_img'):
    batch_norm_param = args.arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    m.embed_img, _ = tf_utils.fc_network(
        m.reshape_conv_feat, neurons=args.arch.img_embed_neurons,
        wt_decay=args.solver.wt_decay, name='img_embed', offset=0,
        batch_norm_param=batch_norm_param, dropout_ratio=args.arch.fc_dropout,
        is_training=is_training)

  # For lstm_ego, and lstm_image, embed the ego motion, accumulate it into an
  # LSTM, combine with image features and accumulate those in an LSTM. Finally
  # combine what you get from the image LSTM with the goal to output an action.
  if args.arch.lstm_ego:
    ego_reshaped = preprocess_egomotion(m.input_tensors['step']['incremental_locs'], 
                                        m.input_tensors['step']['incremental_thetas'])
    with tf.variable_scope('embed_ego'):
      batch_norm_param = args.arch.batch_norm_param
      batch_norm_param['is_training'] = batch_norm_is_training_op
      m.embed_ego, _ = tf_utils.fc_network(
          ego_reshaped, neurons=args.arch.ego_embed_neurons,
          wt_decay=args.solver.wt_decay, name='ego_embed', offset=0,
          batch_norm_param=batch_norm_param, dropout_ratio=args.arch.fc_dropout,
          is_training=is_training)

    state_name, state_init_op, updated_state_op, out_op = lstm_setup(
        'lstm_ego', m.embed_ego, task_params.batch_size, is_single_step, 
        args.arch.lstm_ego_dim, args.arch.lstm_ego_out, num_steps*num_goals,
        m.input_tensors['step']['lstm_ego'])
    state_names += [state_name]
    init_state_ops += [state_init_op]
    updated_state_ops += [updated_state_op]

    # Combine the output with the vision features.
    m.img_ego_op = combine_setup('img_ego', args.arch.combine_type_ego,
                                 m.embed_img, out_op,
                                 args.arch.img_embed_neurons[-1],
                                 args.arch.lstm_ego_out)

    # LSTM on these vision features.
    state_name, state_init_op, updated_state_op, out_op = lstm_setup(
        'lstm_img', m.img_ego_op, task_params.batch_size, is_single_step, 
        args.arch.lstm_img_dim, args.arch.lstm_img_out, num_steps*num_goals,
        m.input_tensors['step']['lstm_img'])
    state_names += [state_name]
    init_state_ops += [state_init_op]
    updated_state_ops += [updated_state_op]

    m.img_for_goal = out_op
    num_img_for_goal_neurons = args.arch.lstm_img_out

  elif args.arch.lstm_img:
    # LSTM on just the image features.
    state_name, state_init_op, updated_state_op, out_op = lstm_setup(
        'lstm_img', m.embed_img, task_params.batch_size, is_single_step,
        args.arch.lstm_img_dim, args.arch.lstm_img_out, num_steps*num_goals,
        m.input_tensors['step']['lstm_img'])
    state_names += [state_name]
    init_state_ops += [state_init_op]
    updated_state_ops += [updated_state_op]
    m.img_for_goal = out_op
    num_img_for_goal_neurons = args.arch.lstm_img_out

  else:
    m.img_for_goal = m.embed_img
    num_img_for_goal_neurons = args.arch.img_embed_neurons[-1]


  if args.arch.use_visit_count:
    m.embed_visit_count = visit_count_fc(
        m.input_tensors['step']['visit_count'],
        m.input_tensors['step']['last_visit'], args.arch.goal_embed_neurons,
        args.solver.wt_decay, args.arch.fc_dropout, is_training=is_training)
    m.embed_goal = m.embed_goal + m.embed_visit_count
  
  m.combined_f = combine_setup('img_goal', args.arch.combine_type,
                               m.img_for_goal, m.embed_goal,
                               num_img_for_goal_neurons,
                               args.arch.goal_embed_neurons[-1])

  # LSTM on the combined representation.
  if args.arch.lstm_output:
    name = 'lstm_output'
    # A few fully connected layers here.
    with tf.variable_scope('action_pred'):
      batch_norm_param = args.arch.batch_norm_param
      batch_norm_param['is_training'] = batch_norm_is_training_op
      x, _ = tf_utils.fc_network(
          m.combined_f, neurons=args.arch.pred_neurons,
          wt_decay=args.solver.wt_decay, name='pred', offset=0,
          batch_norm_param=batch_norm_param, dropout_ratio=args.arch.fc_dropout)

    if args.arch.lstm_output_init_state_from_goal:
      # Use the goal embedding to initialize the LSTM state.
      # UGLY CLUGGY HACK: if this is doing computation for a single time step
      # then this will not involve back prop, so we can use the state input from
      # the feed dict, otherwise we compute the state representation from the
      # goal and feed that in. Necessary for using goal location to generate the
      # state representation.
      m.embed_goal_for_state = tf.expand_dims(m.embed_goal_for_state, dim=1)
      state_op = tf.cond(is_single_step, lambda: m.input_tensors['step'][name],
                         lambda: m.embed_goal_for_state)
      state_name, state_init_op, updated_state_op, out_op = lstm_setup(
          name, x, task_params.batch_size, is_single_step,
          args.arch.lstm_output_dim,
          num_actions_,
          num_steps*num_goals, state_op)
      init_state_ops += [m.embed_goal_for_state]
    else:
      state_op = m.input_tensors['step'][name]
      state_name, state_init_op, updated_state_op, out_op = lstm_setup(
          name, x, task_params.batch_size, is_single_step,
          args.arch.lstm_output_dim,
          num_actions_, num_steps*num_goals, state_op)
      init_state_ops += [state_init_op]

    state_names += [state_name]
    updated_state_ops += [updated_state_op]

    out_op = tf.reshape(out_op, shape=[-1, num_actions_])
    if num_actions_ > num_actions:
      m.action_logits_op = out_op[:,:num_actions]
      m.baseline_op = out_op[:,num_actions:]
    else:
      m.action_logits_op = out_op
      m.baseline_op = None
    m.action_prob_op = tf.nn.softmax(m.action_logits_op)

  else:
    # A few fully connected layers here.
    with tf.variable_scope('action_pred'):
      batch_norm_param = args.arch.batch_norm_param
      batch_norm_param['is_training'] = batch_norm_is_training_op
      out_op, _ = tf_utils.fc_network(
          m.combined_f, neurons=args.arch.pred_neurons,
          wt_decay=args.solver.wt_decay, name='pred', offset=0,
          num_pred=num_actions_,
          batch_norm_param=batch_norm_param,
          dropout_ratio=args.arch.fc_dropout, is_training=is_training)
      if num_actions_ > num_actions:
        m.action_logits_op = out_op[:,:num_actions]
        m.baseline_op = out_op[:,num_actions:]
      else:
        m.action_logits_op = out_op 
        m.baseline_op = None
      m.action_prob_op = tf.nn.softmax(m.action_logits_op)

  m.train_ops = {}
  m.train_ops['step'] = m.action_prob_op
  m.train_ops['common'] = [m.input_tensors['common']['orig_maps'],
                           m.input_tensors['common']['goal_loc'],
                           m.input_tensors['common']['rel_goal_loc_at_start']]
  m.train_ops['state_names'] = state_names
  m.train_ops['init_state'] = init_state_ops
  m.train_ops['updated_state'] = updated_state_ops
  m.train_ops['batch_norm_is_training_op'] = batch_norm_is_training_op

  # Flat list of ops which cache the step data.
  m.train_ops['step_data_cache'] = [tf.no_op()]

  if args.solver.freeze_conv:
    m.train_ops['step_data_cache'] = [m.encoder_output]
  else:
    m.train_ops['step_data_cache'] = []

  ewma_decay = 0.99 if is_training else 0.0
  weight = tf.ones_like(m.input_tensors['train']['action'], dtype=tf.float32,
                        name='weight')

  m.reg_loss_op, m.data_loss_op, m.total_loss_op, m.acc_ops = \
    compute_losses_multi_or(
        m.action_logits_op, m.input_tensors['train']['action'],
        weights=weight, num_actions=num_actions,
        data_loss_wt=args.solver.data_loss_wt,
        reg_loss_wt=args.solver.reg_loss_wt, ewma_decay=ewma_decay)


  if args.solver.freeze_conv:
    vars_to_optimize = list(set(tf.trainable_variables()) - set(vars_))
  else:
    vars_to_optimize = None

  m.lr_op, m.global_step_op, m.train_op, m.should_stop_op, m.optimizer, \
  m.sync_optimizer = tf_utils.setup_training(
      m.total_loss_op, 
      args.solver.initial_learning_rate, 
      args.solver.steps_per_decay,
      args.solver.learning_rate_decay, 
      args.solver.momentum,
      args.solver.max_steps, 
      args.solver.sync, 
      args.solver.adjust_lr_sync,
      args.solver.num_workers, 
      args.solver.task,
      vars_to_optimize=vars_to_optimize,
      clip_gradient_norm=args.solver.clip_gradient_norm,
      typ=args.solver.typ, momentum2=args.solver.momentum2,
      adam_eps=args.solver.adam_eps)
  
  
  if args.arch.sample_gt_prob_type == 'inverse_sigmoid_decay':
    m.sample_gt_prob_op = tf_utils.inverse_sigmoid_decay(args.arch.isd_k,
                                                         m.global_step_op)
  elif args.arch.sample_gt_prob_type == 'zero':
    m.sample_gt_prob_op = tf.constant(-1.0, dtype=tf.float32)
  elif args.arch.sample_gt_prob_type.split('_')[0] == 'step':
    step = int(args.arch.sample_gt_prob_type.split('_')[1])
    m.sample_gt_prob_op = tf_utils.step_gt_prob(
        step, m.input_tensors['step']['step_number'][0,0,0])
  
  m.sample_action_type = args.arch.action_sample_type
  m.sample_action_combine_type = args.arch.action_sample_combine_type
  _add_summaries(m, summary_mode, args.summary.arop_full_summary_iters)
  
  m.init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
  m.saver_op = tf.train.Saver(keep_checkpoint_every_n_hours=4,
                              write_version=tf.train.SaverDef.V2)
  
  return m
