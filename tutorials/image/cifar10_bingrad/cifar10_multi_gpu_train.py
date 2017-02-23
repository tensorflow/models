# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_common

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', cifar10_common.WORKSPACE_PATH+'/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

def tower_loss(scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  images, labels = cifar10.distorted_inputs()

  # Build inference Graph.
  logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    #loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    if re.compile(".*(cross_entropy)|(total_loss).*").match(l.op.name):
      tf.contrib.deprecated.scalar_summary(l.op.name, l)

  return total_loss


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False,dtype=tf.int64)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    #lr = tf.train.exponential_decay(FLAGS.base_lr,
    #                                global_step,
    #                                decay_steps,
    #                                FLAGS.decay_factor,
    #                                staircase=True)

    # Decay the learning rate exponentially based on the number of steps.
    if (('momentum' == FLAGS.optimizer) or ('gd' == FLAGS.optimizer)):
      lr = tf.train.piecewise_constant(global_step,
                                       [tf.to_int64(decay_steps), tf.to_int64(decay_steps + decay_steps / 2)],
                                       # [tf.to_int64(60000), tf.to_int64(180000)],
                                       [FLAGS.base_lr,
                                        FLAGS.base_lr * FLAGS.decay_factor,
                                        FLAGS.base_lr * FLAGS.decay_factor * FLAGS.decay_factor])
    elif ('adam' == FLAGS.optimizer):
      lr = FLAGS.base_lr
    else:
      raise ValueError('Unsupported optimizer type.')

    # Create an optimizer that performs gradient descent.
    #opt = tf.train.GradientDescentOptimizer(lr)
    if ('gd' == FLAGS.optimizer):
      opt = tf.train.GradientDescentOptimizer(lr)
    elif ('momentum' == FLAGS.optimizer):
      opt = tf.train.MomentumOptimizer(lr, 0.9)
    elif ('adam' == FLAGS.optimizer):
      opt = tf.train.AdamOptimizer(lr)
    else:
      opt = tf.train.AdamOptimizer(FLAGS.base_lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_scalers = []
    device_list = ['/gpu:0','/gpu:1','/gpu:0','/gpu:1','/gpu:0','/gpu:1','/gpu:0','/gpu:1']
    summaries = [] # tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
        #with tf.device(device_list[i]):
          #with tf.variable_scope('%s_%d' % (cifar10.TOWER_NAME, i)):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope)

            # Reuse variables for the next tower.
            #tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

            # Calculate the gradients for the batch of data on this CIFAR tower.
            #grads = opt.compute_gradients(loss)
            grads = opt.compute_gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

            for grad, var in grads:
              if grad is not None:
                summaries.append(
                  tf.contrib.deprecated.histogram_summary(var.op.name +  '/%s_%d_orig_gradients' % (cifar10.TOWER_NAME, i),grad))

            ## Clip gradients
            #if FLAGS.clip_factor > 1.0e-5:
            #  grads = cifar10_common.clip_gradients_by_stddev(grads, clip_factor=FLAGS.clip_factor)
            ## Binarize gradients
            #if 1 == FLAGS.grad_bits:
            #  grads = cifar10_common.stochastical_binarize_gradients(grads)

            # Keep track of the gradients across all towers.
            '''
            !!!!!!!!!!!!! WARNING, THE WEIGHTS IN grads MAY NOT BE THE SAME IF EACH GPU USES A LOCAL WEIGHTS!!!!!!!!!!!!!!!!!
            '''
            tower_grads.append(grads)


            if 1 == FLAGS.grad_bits:
              # Always calculate scalers whatever clip_factor is. Returns max value when clip_factor==0.0
              scalers = cifar10_common.gradient_binarizing_scalers(grads, FLAGS.clip_factor)
              tower_scalers.append(scalers)
              for scaler, var in scalers:
               if scaler is not None:
                 summaries.append(
                   tf.contrib.deprecated.scalar_summary(var.op.name +  '/%s_%d_scaler' % (cifar10.TOWER_NAME, i),scaler))

            #for grad, var in grads:
            #  if grad is not None:
            #    summaries.append(
            #      tf.contrib.deprecated.histogram_summary(var.op.name +  '/%s_%d_bin_gradients' % (cifar10.TOWER_NAME, i),grad))

    # We must calculate the mean of each scaler. Note that this is the
    # synchronization point across all towers.
    if 1 == FLAGS.grad_bits:
      mean_scalers = cifar10_common.average_scalers(tower_scalers)
      for mscaler in mean_scalers:
        if mscaler is not None:
          summaries.append(
            tf.contrib.deprecated.scalar_summary(mscaler.op.name + '/mean_scaler', mscaler))

    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
      #with tf.device(device_list[i]):
        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
          if 1 == FLAGS.grad_bits:
            # Clip gradients. Always clip since the max value in towers may be different even when clip_factor==0.0
            grads = cifar10_common.clip_gradients_by_thresholds(tower_grads[i], mean_scalers)
            # Binarize gradients
            grads = cifar10_common.stochastical_binarize_gradients(grads)
            # Keep track of the gradients across all towers.
            tower_grads[i]=grads

            for grad, var in grads:
             if grad is not None:
               summaries.append(
                 tf.contrib.deprecated.histogram_summary(var.op.name +  '/%s_%d_bin_gradients' % (cifar10.TOWER_NAME, i),grad))

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    #grads = cifar10_common.average_gradients(tower_grads)
    tower_grads = cifar10_common.average_gradients2(tower_grads)

    ## Add histograms for gradients.
    #for grad, var in grads:
    #  if grad is not None:
    #    summaries.append(
    #      tf.contrib.deprecated.histogram_summary(var.op.name + '/mean_bin_gradients',grad))

    ## Clip gradients
    #if FLAGS.clip_factor > 1.0e-5:
    #  grads = cifar10_common.clip_gradients_by_stddev(grads, clip_factor=FLAGS.clip_factor)
    ## Binarize gradients
    #if 1 == FLAGS.grad_bits:
    #  grads = cifar10_common.stochastical_binarize_gradients(grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.contrib.deprecated.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in tower_grads[0]:
      if grad is not None:
        summaries.append(
            tf.contrib.deprecated.histogram_summary(var.op.name + '/final_gradients',grad))

    # Apply the gradients to adjust the shared variables.
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    apply_gradient_op = []
    for tower_grad in tower_grads:
      apply_gradient_op.append( opt.apply_gradients(tower_grad, global_step=global_step) )

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(
          tf.contrib.deprecated.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(variables_averages_op,*apply_gradient_op)

    # Only save the variables in the first tower
    save_pattern = ('(%s_%d)' % (cifar10.TOWER_NAME, 0))+".*ExponentialMovingAverage"
    var_dic = {}
    _vars = tf.global_variables() #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=first_tower_scope)
    for _var in _vars:
      if re.compile(save_pattern).match(_var.op.name):
        _var_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', _var.op.name)
        var_dic[_var_name] = _var
    # Create a saver.
    #saver = tf.train.Saver(tf.global_variables())
    saver = tf.train.Saver(var_dic)


    # Build the summary operation from the last tower summaries.
    summary_op = tf.contrib.deprecated.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = FLAGS.log_device_placement
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration #/ FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.6f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
