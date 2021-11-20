# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Optimizer factory for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa

from official.modeling import optimization
from official.vision.image_classification import learning_rate
from official.vision.image_classification.configs import base_configs

# pylint: disable=protected-access


def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any],
    model: Optional[tf.keras.Model] = None):
  """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples: sgd,
      momentum, rmsprop.
    base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params. This
      should contain optimizer specific parameters such as `base_learning_rate`,
      `decay`, etc.
    model: The `tf.keras.Model`. This is used for the shadow copy if using
      `ExponentialMovingAverage`.

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
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_learning_rate, nesterov=nesterov)
  elif optimizer_name == 'momentum':
    logging.info('Using momentum optimizer')
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_learning_rate,
        momentum=params['momentum'],
        nesterov=nesterov)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp')
    rho = params.get('decay', None) or params.get('rho', 0.9)
    momentum = params.get('momentum', 0.9)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=base_learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam')
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)
  elif optimizer_name == 'adamw':
    logging.info('Using AdamW')
    weight_decay = params.get('weight_decay', 0.01)
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay,
        learning_rate=base_learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)
  else:
    raise ValueError('Unknown optimizer %s' % optimizer_name)

  if params.get('lookahead', None):
    logging.info('Using lookahead optimizer.')
    optimizer = tfa.optimizers.Lookahead(optimizer)

  # Moving average should be applied last, as it's applied at test time
  moving_average_decay = params.get('moving_average_decay', 0.)
  if moving_average_decay is not None and moving_average_decay > 0.:
    if model is None:
      raise ValueError(
          '`model` must be provided if using `ExponentialMovingAverage`.')
    logging.info('Including moving average decay.')
    optimizer = optimization.ExponentialMovingAverage(
        optimizer=optimizer, average_decay=moving_average_decay)
    optimizer.shadow_copy(model)
  return optimizer


def build_learning_rate(params: base_configs.LearningRateConfig,
                        batch_size: Optional[int] = None,
                        train_epochs: Optional[int] = None,
                        train_steps: Optional[int] = None):
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
    logging.info(
        'Scaling the learning rate based on the batch size '
        'multiplier. New base_lr: %f', base_lr)

  if decay_type == 'exponential':
    logging.info(
        'Using exponential learning rate with: '
        'initial_learning_rate: %f, decay_steps: %d, '
        'decay_rate: %f', base_lr, decay_steps, decay_rate)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=params.staircase)
  elif decay_type == 'stepwise':
    steps_per_epoch = params.examples_per_epoch // batch_size
    boundaries = [boundary * steps_per_epoch for boundary in params.boundaries]
    multipliers = [batch_size * multiplier for multiplier in params.multipliers]
    logging.info(
        'Using stepwise learning rate. Parameters: '
        'boundaries: %s, values: %s', boundaries, multipliers)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=multipliers)
  elif decay_type == 'cosine_with_warmup':
    lr = learning_rate.CosineDecayWithWarmup(
        batch_size=batch_size,
        total_steps=train_epochs * train_steps,
        warmup_steps=warmup_steps)
  if warmup_steps > 0:
    if decay_type not in ['cosine_with_warmup']:
      logging.info('Applying %d warmup steps to the learning rate',
                   warmup_steps)
      lr = learning_rate.WarmupDecaySchedule(
          lr, warmup_steps, warmup_lr=base_lr)
  return lr
