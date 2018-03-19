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
import sys
import tensorflow as tf
import src.utils as utils
import logging
from tensorflow.contrib import slim
from tensorflow.contrib.metrics.python.ops import confusion_matrix_ops
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
sys.path.insert(0, '../slim')
from preprocessing import inception_preprocessing as ip

resnet_v2_50 = resnet_v2.resnet_v2_50


def custom_residual_block(x, neurons, kernel_size, stride, name, is_training,
                          wt_decay=0.0001, use_residual=True,
                          residual_stride_conv=True, conv_fn=slim.conv2d,
                          batch_norm_param=None):
  
  # batch norm x and relu
  init_var = np.sqrt(2.0/(kernel_size**2)/neurons)
  with arg_scope([conv_fn], 
                 weights_regularizer=slim.l2_regularizer(wt_decay),
                 weights_initializer=tf.random_normal_initializer(stddev=init_var),
                 biases_initializer=tf.zeros_initializer()): 
    
    if batch_norm_param is None:
      batch_norm_param = {'center': True, 'scale': False, 
                          'activation_fn':tf.nn.relu, 
                          'is_training': is_training}
    
    y = slim.batch_norm(x, scope=name+'_bn', **batch_norm_param)

    y = conv_fn(y, num_outputs=neurons, kernel_size=kernel_size, stride=stride,
                activation_fn=None, scope=name+'_1',
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_param)
    
    y = conv_fn(y, num_outputs=neurons, kernel_size=kernel_size,
                    stride=1, activation_fn=None, scope=name+'_2')

    if use_residual:
      if stride != 1 or x.get_shape().as_list()[-1] != neurons:
        batch_norm_param_ = dict(batch_norm_param)
        batch_norm_param_['activation_fn'] = None
        x = conv_fn(x, num_outputs=neurons, kernel_size=1,
                        stride=stride if residual_stride_conv else 1,
                        activation_fn=None, scope=name+'_0_1x1',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_param_)
        if not residual_stride_conv:
          x = slim.avg_pool2d(x, 1, stride=stride, scope=name+'_0_avg')
  
      y = tf.add(x, y, name=name+'_add')
    
    return y

def step_gt_prob(step, step_number_op):
  # Change samping probability from 1 to -1 at step steps.
  with tf.name_scope('step_gt_prob'):
    out = tf.cond(tf.less(step_number_op, step),
            lambda: tf.constant(1.), lambda: tf.constant(-1.))
    return out 

def inverse_sigmoid_decay(k, global_step_op):
  with tf.name_scope('inverse_sigmoid_decay'):
    k = tf.constant(k, dtype=tf.float32)
    tmp = k*tf.exp(-tf.cast(global_step_op, tf.float32)/k)
    tmp = tmp / (1. + tmp)
  return tmp

def dense_resample(im, flow_im, output_valid_mask, name='dense_resample'):
  """ Resample reward at particular locations.
  Args:
    im:      ...xHxWxC matrix to sample from.
    flow_im: ...xHxWx2 matrix, samples the image using absolute offsets as given
             by the flow_im.
  """
  with tf.name_scope(name):
    valid_mask = None
    
    x, y = tf.unstack(flow_im, axis=-1)
    x = tf.cast(tf.reshape(x, [-1]), tf.float32)
    y = tf.cast(tf.reshape(y, [-1]), tf.float32)

    # constants
    shape = tf.unstack(tf.shape(im))
    channels = shape[-1]
    width = shape[-2]
    height = shape[-3]
    num_batch = tf.cast(tf.reduce_prod(tf.stack(shape[:-3])), 'int32')
    zero = tf.constant(0, dtype=tf.int32)

    # Round up and down.
    x0 = tf.cast(tf.floor(x), 'int32'); x1 = x0 + 1;
    y0 = tf.cast(tf.floor(y), 'int32'); y1 = y0 + 1;
    
    if output_valid_mask:
      valid_mask = tf.logical_and(
          tf.logical_and(tf.less_equal(x, tf.cast(width, tf.float32)-1.), tf.greater_equal(x, 0.)),
          tf.logical_and(tf.less_equal(y, tf.cast(height, tf.float32)-1.), tf.greater_equal(y, 0.)))
      valid_mask = tf.reshape(valid_mask, shape=shape[:-1] + [1])
  
    x0 = tf.clip_by_value(x0, zero, width-1)
    x1 = tf.clip_by_value(x1, zero, width-1)
    y0 = tf.clip_by_value(y0, zero, height-1)
    y1 = tf.clip_by_value(y1, zero, height-1)

    dim2 = width; dim1 = width * height;

    # Create base index
    base = tf.reshape(tf.range(num_batch) * dim1, shape=[-1,1])
    base = tf.reshape(tf.tile(base, [1, height*width]), shape=[-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore channels dim
    sh = tf.stack([tf.constant(-1,dtype=tf.int32), channels])
    im_flat = tf.cast(tf.reshape(im, sh), dtype=tf.float32)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = tf.reshape(output, shape=tf.shape(im))
    return output, valid_mask
 
def get_flow(t, theta, map_size, name_scope='gen_flow'):
  """
  Rotates the map by theta and translates the rotated map by t.
  
  Assume that the robot rotates by an angle theta and then moves forward by
  translation t. This function returns the flow field field. For every pixel in
  the new image it tells us which pixel in the original image it came from:
  NewI(x, y) = OldI(flow_x(x,y), flow_y(x,y)).

  Assume there is a point p in the original image. Robot rotates by R and moves
  forward by t.  p1 = Rt*p; p2 = p1 - t; (the world moves in opposite direction.
  So, p2 = Rt*p - t, thus p2 came from R*(p2+t), which is what this function
  calculates.

    t:      ... x 2 (translation for B batches of N motions each).
    theta:  ... x 1 (rotation for B batches of N motions each).
    
    Output: ... x map_size x map_size x 2
  """

  with tf.name_scope(name_scope):
    tx, ty = tf.unstack(tf.reshape(t, shape=[-1, 1, 1, 1, 2]), axis=4)
    theta = tf.reshape(theta, shape=[-1, 1, 1, 1])
    c = tf.constant((map_size-1.)/2., dtype=tf.float32)

    x, y = np.meshgrid(np.arange(map_size), np.arange(map_size))
    x = tf.constant(x[np.newaxis, :, :, np.newaxis], dtype=tf.float32, name='x', 
                    shape=[1, map_size, map_size, 1])
    y = tf.constant(y[np.newaxis, :, :, np.newaxis], dtype=tf.float32, name='y',
                    shape=[1,map_size, map_size, 1])

    x = x-(-tx+c)
    y = y-(-ty+c)

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    xr = cos_theta*x - sin_theta*y
    yr = sin_theta*x + cos_theta*y

    xr = xr + c
    yr = yr + c
    
    flow = tf.stack([xr, yr], axis=-1)
    sh = tf.unstack(tf.shape(t), axis=0)
    sh = tf.stack(sh[:-1]+[tf.constant(_, dtype=tf.int32) for _ in [map_size, map_size, 2]])
    flow = tf.reshape(flow, shape=sh)
    return flow

def distort_image(im, fast_mode=False):
  # All images in the same batch are transformed the same way, but over
  # iterations you see different distortions.
  # im should be float with values between 0 and 1.
  im_ = tf.reshape(im, shape=(-1,1,3))
  im_ = ip.apply_with_random_selector(
      im_, lambda x, ordering: ip.distort_color(x, ordering, fast_mode),
      num_cases=4)
  im_ = tf.reshape(im_, tf.shape(im))
  return im_

def fc_network(x, neurons, wt_decay, name, num_pred=None, offset=0,
               batch_norm_param=None, dropout_ratio=0.0, is_training=None): 
  if dropout_ratio > 0:
    assert(is_training is not None), \
      'is_training needs to be defined when trainnig with dropout.'
  
  repr = []
  for i, neuron in enumerate(neurons):
    init_var = np.sqrt(2.0/neuron)
    if batch_norm_param is not None:
      x = slim.fully_connected(x, neuron, activation_fn=None,
                               weights_initializer=tf.random_normal_initializer(stddev=init_var),
                               weights_regularizer=slim.l2_regularizer(wt_decay),
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_param,
                               biases_initializer=tf.zeros_initializer(),
                               scope='{:s}_{:d}'.format(name, offset+i))
    else:
      x = slim.fully_connected(x, neuron, activation_fn=tf.nn.relu,
                               weights_initializer=tf.random_normal_initializer(stddev=init_var),
                               weights_regularizer=slim.l2_regularizer(wt_decay),
                               biases_initializer=tf.zeros_initializer(),
                               scope='{:s}_{:d}'.format(name, offset+i))
    if dropout_ratio > 0:
       x = slim.dropout(x, keep_prob=1-dropout_ratio, is_training=is_training,
                        scope='{:s}_{:d}'.format('dropout_'+name, offset+i))
    repr.append(x)
  
  if num_pred is not None:
    init_var = np.sqrt(2.0/num_pred)
    x = slim.fully_connected(x, num_pred,
                             weights_regularizer=slim.l2_regularizer(wt_decay),
                             weights_initializer=tf.random_normal_initializer(stddev=init_var),
                             biases_initializer=tf.zeros_initializer(),
                             activation_fn=None,
                             scope='{:s}_pred'.format(name))
  return x, repr

def concat_state_x_list(f, names):
  af = {}
  for i, k in enumerate(names):
    af[k] = np.concatenate([x[i] for x in f], axis=1)
  return af

def concat_state_x(f, names):
  af = {}
  for k in names:
    af[k] = np.concatenate([x[k] for x in f], axis=1)
    # af[k] = np.swapaxes(af[k], 0, 1)
  return af

def sample_action(rng, action_probs, optimal_action, sample_gt_prob,
                  type='sample', combine_type='one_or_other'):
  optimal_action_ = optimal_action/np.sum(optimal_action+0., 1, keepdims=True)
  action_probs_ = action_probs/np.sum(action_probs+0.001, 1, keepdims=True)
  batch_size = action_probs_.shape[0]

  action = np.zeros((batch_size), dtype=np.int32)
  action_sample_wt = np.zeros((batch_size), dtype=np.float32)
  if combine_type == 'add':
    sample_gt_prob_ = np.minimum(np.maximum(sample_gt_prob, 0.), 1.)

  for i in range(batch_size):
    if combine_type == 'one_or_other':
      sample_gt = rng.rand() < sample_gt_prob
      if sample_gt: distr_ = optimal_action_[i,:]*1.
      else: distr_ = action_probs_[i,:]*1.
    elif combine_type == 'add':
      distr_ = optimal_action_[i,:]*sample_gt_prob_ + \
          (1.-sample_gt_prob_)*action_probs_[i,:]
      distr_ = distr_ / np.sum(distr_)

    if type == 'sample':
      action[i] = np.argmax(rng.multinomial(1, distr_, size=1))
    elif type == 'argmax':
      action[i] = np.argmax(distr_)
    action_sample_wt[i] = action_probs_[i, action[i]] / distr_[action[i]]
  return action, action_sample_wt

def train_step_custom_online_sampling(sess, train_op, global_step,
                                      train_step_kwargs, mode='train'):
  m          = train_step_kwargs['m']
  obj        = train_step_kwargs['obj']
  rng_data   = train_step_kwargs['rng_data']
  rng_action = train_step_kwargs['rng_action']
  writer     = train_step_kwargs['writer']
  iters      = train_step_kwargs['iters']
  num_steps  = train_step_kwargs['num_steps']
  logdir     = train_step_kwargs['logdir']
  dagger_sample_bn_false = train_step_kwargs['dagger_sample_bn_false']
  train_display_interval = train_step_kwargs['train_display_interval']
  if 'outputs' not in m.train_ops:
    m.train_ops['outputs'] = []

  s_ops = m.summary_ops[mode]
  val_additional_ops = []

  # Print all variables here.
  if False:
    v = tf.get_collection(tf.GraphKeys.VARIABLES)
    v_op = [_.value() for _ in v]
    v_op_value = sess.run(v_op)

    filter = lambda x, y: 'Adam' in x.name
    # filter = lambda x, y: np.is_any_nan(y)
    ind = [i for i, (_, __) in enumerate(zip(v, v_op_value)) if filter(_, __)]
    v = [v[i] for i in ind]
    v_op_value = [v_op_value[i] for i in ind]

    for i in range(len(v)): 
      logging.info('XXXX: variable: %30s, is_any_nan: %5s, norm: %f.',
                   v[i].name, np.any(np.isnan(v_op_value[i])),
                   np.linalg.norm(v_op_value[i]))

  tt = utils.Timer()
  for i in range(iters):
    tt.tic()
    # Sample a room.
    e = obj.sample_env(rng_data)

    # Initialize the agent.
    init_env_state = e.reset(rng_data)

    # Get and process the common data.
    input = e.get_common_data()
    input = e.pre_common_data(input)
    feed_dict  = prepare_feed_dict(m.input_tensors['common'], input)
    if dagger_sample_bn_false:
      feed_dict[m.train_ops['batch_norm_is_training_op']] = False
    common_data = sess.run(m.train_ops['common'], feed_dict=feed_dict)

    states = []
    state_features = []
    state_targets = []
    net_state_to_input = []
    step_data_cache = []
    executed_actions = []
    rewards = []
    action_sample_wts = []
    states.append(init_env_state)

    net_state = sess.run(m.train_ops['init_state'], feed_dict=feed_dict)
    net_state = dict(zip(m.train_ops['state_names'], net_state))
    net_state_to_input.append(net_state)
    for j in range(num_steps):
      f = e.get_features(states[j], j)
      f = e.pre_features(f)
      f.update(net_state)
      f['step_number'] = np.ones((1,1,1), dtype=np.int32)*j
      state_features.append(f)

      feed_dict = prepare_feed_dict(m.input_tensors['step'], state_features[-1])
      optimal_action = e.get_optimal_action(states[j], j)
      for x, v in zip(m.train_ops['common'], common_data):
        feed_dict[x] = v
      if dagger_sample_bn_false:
        feed_dict[m.train_ops['batch_norm_is_training_op']] = False
      outs = sess.run([m.train_ops['step'], m.sample_gt_prob_op,
                       m.train_ops['step_data_cache'],
                       m.train_ops['updated_state'],
                       m.train_ops['outputs']], feed_dict=feed_dict)
      action_probs = outs[0]
      sample_gt_prob = outs[1]
      step_data_cache.append(dict(zip(m.train_ops['step_data_cache'], outs[2])))
      net_state = outs[3]
      if hasattr(e, 'update_state'):
        outputs = outs[4]
        outputs = dict(zip(m.train_ops['output_names'], outputs))
        e.update_state(outputs, j)
      state_targets.append(e.get_targets(states[j], j))

      if j < num_steps-1:
        # Sample from action_probs and optimal action.
        action, action_sample_wt = sample_action(
            rng_action, action_probs, optimal_action, sample_gt_prob,
            m.sample_action_type, m.sample_action_combine_type)
        next_state, reward = e.take_action(states[j], action, j)
        executed_actions.append(action)
        states.append(next_state)
        rewards.append(reward)
        action_sample_wts.append(action_sample_wt)
        net_state = dict(zip(m.train_ops['state_names'], net_state))
        net_state_to_input.append(net_state)
    
    # Concatenate things together for training.
    rewards = np.array(rewards).T
    action_sample_wts = np.array(action_sample_wts).T
    executed_actions = np.array(executed_actions).T
    all_state_targets = concat_state_x(state_targets, e.get_targets_name())
    all_state_features = concat_state_x(state_features,
                                        e.get_features_name()+['step_number'])
    # all_state_net = concat_state_x(net_state_to_input,
    # m.train_ops['state_names'])
    all_step_data_cache = concat_state_x(step_data_cache,
                                         m.train_ops['step_data_cache'])

    dict_train = dict(input)
    dict_train.update(all_state_features)
    dict_train.update(all_state_targets)
    # dict_train.update(all_state_net)
    dict_train.update(net_state_to_input[0])
    dict_train.update(all_step_data_cache)
    dict_train.update({'rewards': rewards, 
                       'action_sample_wts': action_sample_wts,
                       'executed_actions': executed_actions})
    feed_dict = prepare_feed_dict(m.input_tensors['train'], dict_train)
    for x in m.train_ops['step_data_cache']:
      feed_dict[x] = all_step_data_cache[x]
    if mode == 'train':
      n_step = sess.run(global_step)

      if np.mod(n_step, train_display_interval) == 0:
        total_loss, np_global_step, summary, print_summary = sess.run(
            [train_op, global_step, s_ops.summary_ops, s_ops.print_summary_ops],
            feed_dict=feed_dict)
        logging.error("")
      else:
        total_loss, np_global_step, summary = sess.run(
            [train_op, global_step, s_ops.summary_ops], feed_dict=feed_dict)

      if writer is not None and summary is not None:
        writer.add_summary(summary, np_global_step)

      should_stop = sess.run(m.should_stop_op)

    if mode != 'train':
      arop = [[] for j in range(len(s_ops.additional_return_ops))]
      for j in range(len(s_ops.additional_return_ops)):
        if s_ops.arop_summary_iters[j] < 0 or i < s_ops.arop_summary_iters[j]:
          arop[j] = s_ops.additional_return_ops[j]
      val = sess.run(arop, feed_dict=feed_dict)
      val_additional_ops.append(val)
      tt.toc(log_at=60, log_str='val timer {:d} / {:d}: '.format(i, iters), 
             type='time')

  if mode != 'train':
    # Write the default val summaries.
    summary, print_summary, np_global_step = sess.run(
        [s_ops.summary_ops, s_ops.print_summary_ops, global_step]) 
    if writer is not None and summary is not None:
      writer.add_summary(summary, np_global_step)

    # write custom validation ops
    val_summarys = []
    val_additional_ops = zip(*val_additional_ops)
    if len(s_ops.arop_eval_fns) > 0:
      val_metric_summary = tf.summary.Summary()
      for i in range(len(s_ops.arop_eval_fns)):
        val_summary = None
        if s_ops.arop_eval_fns[i] is not None:
          val_summary = s_ops.arop_eval_fns[i](val_additional_ops[i],
                                               np_global_step, logdir,
                                               val_metric_summary,
                                               s_ops.arop_summary_iters[i])
        val_summarys.append(val_summary)
      if writer is not None:
        writer.add_summary(val_metric_summary, np_global_step)

    # Return the additional val_ops
    total_loss = (val_additional_ops, val_summarys)
    should_stop = None
  
  return total_loss, should_stop

def train_step_custom_v2(sess, train_op, global_step, train_step_kwargs,
                         mode='train'):
  m      = train_step_kwargs['m']
  obj    = train_step_kwargs['obj']
  rng    = train_step_kwargs['rng']
  writer = train_step_kwargs['writer']
  iters  = train_step_kwargs['iters']
  logdir = train_step_kwargs['logdir']
  train_display_interval = train_step_kwargs['train_display_interval']

  s_ops = m.summary_ops[mode]
  val_additional_ops = [] 

  # Print all variables here.
  if False:
    v = tf.get_collection(tf.GraphKeys.VARIABLES)
    v_op = [_.value() for _ in v]
    v_op_value = sess.run(v_op)

    filter = lambda x, y: 'Adam' in x.name
    # filter = lambda x, y: np.is_any_nan(y)
    ind = [i for i, (_, __) in enumerate(zip(v, v_op_value)) if filter(_, __)]
    v = [v[i] for i in ind]
    v_op_value = [v_op_value[i] for i in ind]

    for i in range(len(v)): 
      logging.info('XXXX: variable: %30s, is_any_nan: %5s, norm: %f.',
                   v[i].name, np.any(np.isnan(v_op_value[i])),
                   np.linalg.norm(v_op_value[i]))

  tt = utils.Timer()
  for i in range(iters):
    tt.tic()
    e          = obj.sample_env(rng)
    rngs       = e.gen_rng(rng)
    input_data = e.gen_data(*rngs)
    input_data = e.pre_data(input_data)
    feed_dict  = prepare_feed_dict(m.input_tensors, input_data)

    if mode == 'train':
      n_step = sess.run(global_step)

      if np.mod(n_step, train_display_interval) == 0:
        total_loss, np_global_step, summary, print_summary = sess.run(
            [train_op, global_step, s_ops.summary_ops, s_ops.print_summary_ops], 
            feed_dict=feed_dict)
      else:
        total_loss, np_global_step, summary = sess.run(
            [train_op, global_step, s_ops.summary_ops],
            feed_dict=feed_dict)

      if writer is not None and summary is not None:
        writer.add_summary(summary, np_global_step)

      should_stop = sess.run(m.should_stop_op)

    if mode != 'train':
      arop = [[] for j in range(len(s_ops.additional_return_ops))]
      for j in range(len(s_ops.additional_return_ops)):
        if s_ops.arop_summary_iters[j] < 0 or i < s_ops.arop_summary_iters[j]:
          arop[j] = s_ops.additional_return_ops[j]
      val = sess.run(arop, feed_dict=feed_dict)
      val_additional_ops.append(val)
      tt.toc(log_at=60, log_str='val timer {:d} / {:d}: '.format(i, iters), 
             type='time')

  if mode != 'train':
    # Write the default val summaries.
    summary, print_summary, np_global_step = sess.run(
        [s_ops.summary_ops, s_ops.print_summary_ops, global_step]) 
    if writer is not None and summary is not None:
      writer.add_summary(summary, np_global_step)

    # write custom validation ops
    val_summarys = []
    val_additional_ops = zip(*val_additional_ops)
    if len(s_ops.arop_eval_fns) > 0:
      val_metric_summary = tf.summary.Summary()
      for i in range(len(s_ops.arop_eval_fns)):
        val_summary = None
        if s_ops.arop_eval_fns[i] is not None:
          val_summary = s_ops.arop_eval_fns[i](val_additional_ops[i],
                                               np_global_step, logdir,
                                               val_metric_summary,
                                               s_ops.arop_summary_iters[i])
        val_summarys.append(val_summary)
      if writer is not None:
        writer.add_summary(val_metric_summary, np_global_step)

    # Return the additional val_ops
    total_loss = (val_additional_ops, val_summarys)
    should_stop = None

  return total_loss, should_stop

def train_step_custom(sess, train_op, global_step, train_step_kwargs, 
                      mode='train'):
  m        = train_step_kwargs['m']
  params   = train_step_kwargs['params']
  rng      = train_step_kwargs['rng']
  writer   = train_step_kwargs['writer']
  iters    = train_step_kwargs['iters']
  gen_rng  = train_step_kwargs['gen_rng']
  logdir   = train_step_kwargs['logdir']
  gen_data = train_step_kwargs['gen_data']
  pre_data = train_step_kwargs['pre_data']
  train_display_interval = train_step_kwargs['train_display_interval']
  
  val_additional_ops = [] 
  # Print all variables here.
  if False:
    v = tf.get_collection(tf.GraphKeys.VARIABLES)
    for _ in v: 
      val = sess.run(_.value())
      logging.info('variable: %30s, is_any_nan: %5s, norm: %f.', _.name,
                   np.any(np.isnan(val)), np.linalg.norm(val))

  for i in range(iters):
    rngs       = gen_rng(params, rng)
    input_data = gen_data(params, *rngs)
    input_data = pre_data(params, input_data)
    feed_dict  = prepare_feed_dict(m.input_tensors, input_data)
    
    if mode == 'train':
      n_step = sess.run(global_step)
      
      if np.mod(n_step, train_display_interval) == 0:
        total_loss, np_global_step, summary, print_summary = sess.run(
            [train_op, global_step, m.summary_op[mode], m.print_summary_op[mode]], 
            feed_dict=feed_dict)
      else:
        total_loss, np_global_step, summary = sess.run(
            [train_op, global_step, m.summary_op[mode]],
            feed_dict=feed_dict)

      if writer is not None:
        writer.add_summary(summary, np_global_step)
        
      should_stop = sess.run(m.should_stop_op)
    
    if mode == 'val':
      val = sess.run(m.agg_update_op[mode] + m.additional_return_op[mode], 
                     feed_dict=feed_dict)
      val_additional_ops.append(val[len(m.agg_update_op[mode]):])
  
  if mode == 'val':
    summary, print_summary, np_global_step = sess.run(
        [m.summary_op[mode], m.print_summary_op[mode], global_step]) 
    if writer is not None:
      writer.add_summary(summary, np_global_step)
    sess.run([m.agg_reset_op[mode]])
    
    # write custom validation ops
    if m.eval_metrics_fn[mode] is not None:
      val_metric_summary = m.eval_metrics_fn[mode](val_additional_ops,
                                                   np_global_step, logdir)
      if writer is not None:
        writer.add_summary(val_metric_summary, np_global_step)
    
    total_loss = val_additional_ops
    should_stop = None
    
  return total_loss, should_stop

def setup_training(loss_op, initial_learning_rate, steps_per_decay,
                   learning_rate_decay, momentum, max_steps,
                   sync=False, adjust_lr_sync=True,
                   num_workers=1, replica_id=0, vars_to_optimize=None, 
                   clip_gradient_norm=0, typ=None, momentum2=0.999,
                   adam_eps=1e-8):
  if sync and adjust_lr_sync:
    initial_learning_rate = initial_learning_rate * num_workers
    max_steps = np.int(max_steps / num_workers)
    steps_per_decay = np.int(steps_per_decay / num_workers)

  global_step_op = slim.get_or_create_global_step()
  lr_op          = tf.train.exponential_decay(initial_learning_rate,
    global_step_op, steps_per_decay, learning_rate_decay, staircase=True)
  if typ == 'sgd':
    optimizer      = tf.train.MomentumOptimizer(lr_op, momentum)
  elif typ == 'adam':
    optimizer      = tf.train.AdamOptimizer(learning_rate=lr_op, beta1=momentum,
                                            beta2=momentum2, epsilon=adam_eps)
  
  if sync:
    
    sync_optimizer = tf.train.SyncReplicasOptimizer(optimizer, 
                                               replicas_to_aggregate=num_workers, 
                                               replica_id=replica_id, 
                                               total_num_replicas=num_workers)
    train_op       = slim.learning.create_train_op(loss_op, sync_optimizer,
                                                   variables_to_train=vars_to_optimize,
                                                   clip_gradient_norm=clip_gradient_norm)
  else:
    sync_optimizer = None
    train_op       = slim.learning.create_train_op(loss_op, optimizer,
                                                   variables_to_train=vars_to_optimize,
                                                   clip_gradient_norm=clip_gradient_norm)
    should_stop_op = tf.greater_equal(global_step_op, max_steps)
  return lr_op, global_step_op, train_op, should_stop_op, optimizer, sync_optimizer

def add_value_to_summary(metric_summary, tag, val, log=True, tag_str=None):
  """Adds a scalar summary to the summary object. Optionally also logs to
  logging."""
  new_value = metric_summary.value.add();
  new_value.tag = tag
  new_value.simple_value = val
  if log:
    if tag_str is None:
      tag_str = tag + '%f'
    logging.info(tag_str, val)

def add_scalar_summary_op(tensor, name=None, 
    summary_key='summaries', print_summary_key='print_summaries', prefix=''):
  collections = []
  op = tf.summary.scalar(name, tensor, collections=collections)
  if summary_key != print_summary_key:
    tf.add_to_collection(summary_key, op)
  
  op = tf.Print(op, [tensor], '    {:-<25s}: '.format(name) + prefix)
  tf.add_to_collection(print_summary_key, op)
  return op

def setup_inputs(inputs):
  input_tensors = {}
  input_shapes  = {}
  for (name, typ, sz) in inputs:
    _ = tf.placeholder(typ, shape=sz, name=name)
    input_tensors[name] = _
    input_shapes[name]  = sz
  return input_tensors, input_shapes

def prepare_feed_dict(input_tensors, inputs):
  feed_dict = {}
  for n in input_tensors.keys():
    feed_dict[input_tensors[n]] = inputs[n].astype(input_tensors[n].dtype.as_numpy_dtype)
  return feed_dict

def simple_add_summaries(summarize_ops, summarize_names,
                         summary_key='summaries',
                         print_summary_key='print_summaries', prefix=''):
  for op, name, in zip(summarize_ops, summarize_names):
    add_scalar_summary_op(op, name, summary_key, print_summary_key, prefix)

  summary_op       = tf.summary.merge_all(summary_key)
  print_summary_op = tf.summary.merge_all(print_summary_key)
  return summary_op, print_summary_op

def add_summary_ops(m, summarize_ops, summarize_names, to_aggregate=None,
                    summary_key='summaries',
                    print_summary_key='print_summaries', prefix=''):
  if type(to_aggregate) != list:
    to_aggregate = [to_aggregate for _ in summarize_ops]
  
  # set up aggregating metrics
  if np.any(to_aggregate):
    agg_ops = []
    for op, name, to_agg in zip(summarize_ops, summarize_names, to_aggregate):
      if to_agg:
        # agg_ops.append(slim.metrics.streaming_mean(op, return_reset_op=True))
        agg_ops.append(tf.contrib.metrics.streaming_mean(op))
        # agg_ops.append(tf.contrib.metrics.streaming_mean(op, return_reset_op=True))
      else:
        agg_ops.append([None, None, None])

    # agg_values_op, agg_update_op, agg_reset_op = zip(*agg_ops)
    # agg_update_op = [x for x in agg_update_op if x is not None]
    # agg_reset_op = [x for x in agg_reset_op if x is not None]
    agg_values_op, agg_update_op = zip(*agg_ops)
    agg_update_op = [x for x in agg_update_op if x is not None]
    agg_reset_op  = [tf.no_op()]
  else:
    agg_values_op = [None for _ in to_aggregate]
    agg_update_op = [tf.no_op()]
    agg_reset_op  = [tf.no_op()]

  for op, name, to_agg, agg_op in zip(summarize_ops, summarize_names, to_aggregate, agg_values_op):
    if to_agg:
      add_scalar_summary_op(agg_op, name, summary_key, print_summary_key, prefix)
    else:
      add_scalar_summary_op(op, name, summary_key, print_summary_key, prefix)

  summary_op       = tf.summary.merge_all(summary_key)
  print_summary_op = tf.summary.merge_all(print_summary_key)
  return summary_op, print_summary_op, agg_update_op, agg_reset_op



def accum_val_ops(outputs, names, global_step, output_dir, metric_summary, N):
  """Processes the collected outputs to compute AP for action prediction.
  
  Args:
    outputs        : List of scalar ops to summarize.
    names          : Name of the scalar ops.
    global_step    : global_step.
    output_dir     : where to store results.
    metric_summary : summary object to add summaries to.
    N              : number of outputs to process.
  """
  outs = []
  if N >= 0:
    outputs = outputs[:N]
  for i in range(len(outputs[0])):
    scalar = np.array(map(lambda x: x[i], outputs))
    assert(scalar.ndim == 1)
    add_value_to_summary(metric_summary, names[i], np.mean(scalar),
                         tag_str='{:>27s}:  [{:s}]: %f'.format(names[i], ''))
    outs.append(np.mean(scalar))
  return outs

def get_default_summary_ops():
  return utils.Foo(summary_ops=None, print_summary_ops=None, 
                   additional_return_ops=[], arop_summary_iters=[],
                   arop_eval_fns=[])


def simple_summaries(summarize_ops, summarize_names, mode, to_aggregate=False,
                     scope_name='summary'):

  if type(to_aggregate) != list:
    to_aggregate = [to_aggregate for _ in summarize_ops]
  
  summary_key = '{:s}_summaries'.format(mode)
  print_summary_key = '{:s}_print_summaries'.format(mode)
  prefix=' [{:s}]: '.format(mode)
  
  # Default ops for things that dont need to be aggregated.
  if not np.all(to_aggregate):
    for op, name, to_agg in zip(summarize_ops, summarize_names, to_aggregate):
      if not to_agg:
        add_scalar_summary_op(op, name, summary_key, print_summary_key, prefix)
    summary_ops = tf.summary.merge_all(summary_key)
    print_summary_ops = tf.summary.merge_all(print_summary_key)
  else:
    summary_ops = tf.no_op()
    print_summary_ops = tf.no_op()
 
  # Default ops for things that dont need to be aggregated.
  if np.any(to_aggregate):
    additional_return_ops = [[summarize_ops[i] 
                              for i, x in enumerate(to_aggregate )if x]]
    arop_summary_iters = [-1]
    s_names = ['{:s}/{:s}'.format(scope_name, summarize_names[i]) 
               for i, x in enumerate(to_aggregate) if x]
    fn = lambda outputs, global_step, output_dir, metric_summary, N: \
      accum_val_ops(outputs, s_names, global_step, output_dir, metric_summary,
                    N)
    arop_eval_fns = [fn]
  else:
    additional_return_ops = []
    arop_summary_iters = []
    arop_eval_fns = []
  return summary_ops, print_summary_ops, additional_return_ops, \
    arop_summary_iters, arop_eval_fns
