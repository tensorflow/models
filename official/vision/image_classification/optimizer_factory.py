# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Optimizer factory for vision tasks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, Dict, Text
from official.vision.image_classification import learning_rate
from official.vision.image_classification.configs import base_configs


def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any]):
  """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples:
      sgd, momentum, rmsprop.
    base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params.
      This should contain optimizer specific parameters such as
      `base_learning_rate`, `decay`, etc.

  Returns:
    A tf.keras.Optimizer.

  Raises:
    ValueError if the provided optimizer_name is not supported.

  """
  optimizer_name = optimizer_name.lower()
  logging.info('Building %s optimizer with params %s', optimizer_name, params)

  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        nesterov=nesterov)
  elif optimizer_name == 'momentum':
    logging.info('Using momentum optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        momentum=params['momentum'],
                                        nesterov=nesterov)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp')
    rho = params.get('decay', None) or params.get('rho', 0.9)
    momentum = params.get('momentum', 0.9)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate,
                                            rho=rho,
                                            momentum=momentum,
                                            epsilon=epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam')
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon)
  elif optimizer_name == 'adamw':
    logging.info('Using AdamW')
    weight_decay = params.get('weight_decay', 0.01)
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
                                     learning_rate=base_learning_rate,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)
  else:
    raise ValueError('Unknown optimizer %s' % optimizer_name)

  moving_average_decay = params.get('moving_average_decay', 0.)
  if moving_average_decay is not None and moving_average_decay > 0.:
    logging.info('Including moving average decay.')
    optimizer = tfa.optimizers.MovingAverage(
        optimizer,
        average_decay=params['moving_average_decay'],
        num_updates=None)
  if params.get('lookahead', None):
    logging.info('Using lookahead optimizer.')
    optimizer = tfa.optimizers.Lookahead(optimizer)
  return optimizer


def build_learning_rate(params: base_configs.LearningRateConfig,
                        batch_size: int = None,
                        train_steps: int = None):
  """Build the learning rate given the provided configuration."""
  decay_type = params.name
  base_lr = params.initial_lr
  decay_rate = params.decay_rate
  if params.decay_epochs is not None:
    decay_steps = params.decay_epochs * train_steps
  else:
    decay_steps = 0
  if params.warmup_epochs is not None:
    warmup_steps = params.warmup_epochs * train_steps
  else:
    warmup_steps = 0

  lr_multiplier = params.scale_by_batch_size

  if lr_multiplier and lr_multiplier > 0:
    # Scale the learning rate based on the batch size and a multiplier
    base_lr *= lr_multiplier * batch_size
    logging.info('Scaling the learning rate based on the batch size '
                 'multiplier. New base_lr: %f', base_lr)

  if decay_type == 'exponential':
    logging.info('Using exponential learning rate with: '
                 'initial_learning_rate: %f, decay_steps: %d, '
                 'decay_rate: %f', base_lr, decay_steps, decay_rate)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
  elif decay_type == 'piecewise_constant_with_warmup':
    logging.info('Using Piecewise constant decay with warmup. '
                 'Parameters: batch_size: %d, epoch_size: %d, '
                 'warmup_epochs: %d, boundaries: %s, multipliers: %s',
                 batch_size, params.examples_per_epoch,
                 params.warmup_epochs, params.boundaries,
                 params.multipliers)
    lr = learning_rate.PiecewiseConstantDecayWithWarmup(
        batch_size=batch_size,
        epoch_size=params.examples_per_epoch,
        warmup_epochs=params.warmup_epochs,
        boundaries=params.boundaries,
        multipliers=params.multipliers)
  if warmup_steps > 0:
    if decay_type != 'piecewise_constant_with_warmup':
      logging.info('Applying %d warmup steps to the learning rate',
                   warmup_steps)
      lr = learning_rate.WarmupDecaySchedule(lr, warmup_steps)
  return lr
