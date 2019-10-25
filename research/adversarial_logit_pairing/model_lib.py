# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Library with common functions for training and eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2


def default_hparams():
  """Returns default hyperparameters."""
  return tf.contrib.training.HParams(
      # Batch size for training and evaluation.
      batch_size=32,
      eval_batch_size=50,

      # General training parameters.
      weight_decay=0.0001,
      label_smoothing=0.1,

      # Parameters of the adversarial training.
      train_adv_method='clean',  # adversarial training method
      train_lp_weight=0.0,  # Weight of adversarial logit pairing loss

      # Parameters of the optimizer.
      optimizer='rms',  # possible values are: 'rms', 'momentum', 'adam'
      momentum=0.9,  # momentum
      rmsprop_decay=0.9,  # Decay term for RMSProp
      rmsprop_epsilon=1.0,  # Epsilon term for RMSProp

      # Parameters of learning rate schedule.
      lr_schedule='exp_decay',  # Possible values: 'exp_decay', 'step', 'fixed'
      learning_rate=0.045,
      lr_decay_factor=0.94,  # Learning exponential decay
      lr_num_epochs_per_decay=2.0,  # Number of epochs per lr decay
      lr_list=[1.0 / 6, 2.0 / 6, 3.0 / 6,
               4.0 / 6, 5.0 / 6, 1.0, 0.1, 0.01,
               0.001, 0.0001],
      lr_decay_epochs=[1, 2, 3, 4, 5, 30, 60, 80,
                       90])


def get_lr_schedule(hparams, examples_per_epoch, replicas_to_aggregate=1):
  """Returns TensorFlow op which compute learning rate.

  Args:
    hparams: hyper parameters.
    examples_per_epoch: number of training examples per epoch.
    replicas_to_aggregate: number of training replicas running in parallel.

  Raises:
    ValueError: if learning rate schedule specified in hparams is incorrect.

  Returns:
    learning_rate: tensor with learning rate.
    steps_per_epoch: number of training steps per epoch.
  """
  global_step = tf.train.get_or_create_global_step()
  steps_per_epoch = float(examples_per_epoch) / float(hparams.batch_size)
  if replicas_to_aggregate > 0:
    steps_per_epoch /= replicas_to_aggregate

  if hparams.lr_schedule == 'exp_decay':
    decay_steps = long(steps_per_epoch * hparams.lr_num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,
        global_step,
        decay_steps,
        hparams.lr_decay_factor,
        staircase=True)
  elif hparams.lr_schedule == 'step':
    lr_decay_steps = [long(epoch * steps_per_epoch)
                      for epoch in hparams.lr_decay_epochs]
    learning_rate = tf.train.piecewise_constant(
        global_step, lr_decay_steps, hparams.lr_list)
  elif hparams.lr_schedule == 'fixed':
    learning_rate = hparams.learning_rate
  else:
    raise ValueError('Invalid value of lr_schedule: %s' % hparams.lr_schedule)

  if replicas_to_aggregate > 0:
    learning_rate *= replicas_to_aggregate

  return learning_rate, steps_per_epoch


def get_optimizer(hparams, learning_rate):
  """Returns optimizer.

  Args:
    hparams: hyper parameters.
    learning_rate: learning rate tensor.

  Raises:
    ValueError: if type of optimizer specified in hparams is incorrect.

  Returns:
    Instance of optimizer class.
  """
  if hparams.optimizer == 'rms':
    optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                          hparams.rmsprop_decay,
                                          hparams.momentum,
                                          hparams.rmsprop_epsilon)
  elif hparams.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           hparams.momentum)
  elif hparams.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  else:
    raise ValueError('Invalid value of optimizer: %s' % hparams.optimizer)
  return optimizer


RESNET_MODELS = {'resnet_v2_50': resnet_v2.resnet_v2_50}


def get_model(model_name, num_classes):
  """Returns function which creates model.

  Args:
    model_name: Name of the model.
    num_classes: Number of classes.

  Raises:
    ValueError: If model_name is invalid.

  Returns:
    Function, which creates model when called.
  """
  if model_name.startswith('resnet'):
    def resnet_model(images, is_training, reuse=tf.AUTO_REUSE):
      with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_fn = RESNET_MODELS[model_name]
        logits, _ = resnet_fn(images, num_classes, is_training=is_training,
                              reuse=reuse)
        logits = tf.reshape(logits, [-1, num_classes])
      return logits
    return resnet_model
  else:
    raise ValueError('Invalid model: %s' % model_name)


def filter_trainable_variables(trainable_scopes):
  """Keep only trainable variables which are prefixed with given scopes.

  Args:
    trainable_scopes: either list of trainable scopes or string with comma
      separated list of trainable scopes.

  This function removes all variables which are not prefixed with given
  trainable_scopes from collection of trainable variables.
  Useful during network fine tuning, when you only need to train subset of
  variables.
  """
  if not trainable_scopes:
    return
  if isinstance(trainable_scopes, six.string_types):
    trainable_scopes = [scope.strip() for scope in trainable_scopes.split(',')]
  trainable_scopes = {scope for scope in trainable_scopes if scope}
  if not trainable_scopes:
    return
  trainable_collection = tf.get_collection_ref(
      tf.GraphKeys.TRAINABLE_VARIABLES)
  non_trainable_vars = [
      v for v in trainable_collection
      if not any([v.op.name.startswith(s) for s in trainable_scopes])
  ]
  for v in non_trainable_vars:
    trainable_collection.remove(v)
