# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Functions to build DetectionModel training optimizers."""

import tensorflow as tf


from object_detection.utils import learning_schedules


def build_optimizers_tf_v1(optimizer_config, global_step=None):
  """Create a TF v1 compatible optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.
    global_step: A variable representing the current step.
      If None, defaults to tf.train.get_or_create_global_step()

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')
  optimizer = None

  summary_vars = []
  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=config.decay,
        momentum=config.momentum_optimizer_value,
        epsilon=config.epsilon)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=config.momentum_optimizer_value)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)


  if optimizer is None:
    raise ValueError('Optimizer %s not supported.' % optimizer_type)

  if optimizer_config.use_moving_average:
    optimizer = tf.contrib.opt.MovingAverageOptimizer(
        optimizer, average_decay=optimizer_config.moving_average_decay)

  return optimizer, summary_vars


def build_optimizers_tf_v2(optimizer_config, global_step=None):
  """Create a TF v2 compatible optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.
    global_step: A variable representing the current step.
      If None, defaults to tf.train.get_or_create_global_step()

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')
  optimizer = None

  summary_vars = []
  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate,
        decay=config.decay,
        momentum=config.momentum_optimizer_value,
        epsilon=config.epsilon)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate,
        momentum=config.momentum_optimizer_value)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    learning_rate = _create_learning_rate(config.learning_rate,
                                          global_step=global_step)
    summary_vars.append(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

  if optimizer is None:
    raise ValueError('Optimizer %s not supported.' % optimizer_type)

  if optimizer_config.use_moving_average:
    raise ValueError('Moving average not supported in eager mode.')

  return optimizer, summary_vars


def build(config, global_step=None):

  if tf.executing_eagerly():
    return build_optimizers_tf_v2(config, global_step)
  else:
    return build_optimizers_tf_v1(config, global_step)


def _create_learning_rate(learning_rate_config, global_step=None):
  """Create optimizer learning rate based on config.

  Args:
    learning_rate_config: A LearningRate proto message.
    global_step: A variable representing the current step.
      If None, defaults to tf.train.get_or_create_global_step()

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  if global_step is None:
    global_step = tf.train.get_or_create_global_step()
  learning_rate = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    learning_rate = tf.constant(config.learning_rate, dtype=tf.float32,
                                name='learning_rate')

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    learning_rate = learning_schedules.exponential_decay_with_burnin(
        global_step,
        config.initial_learning_rate,
        config.decay_steps,
        config.decay_factor,
        burnin_learning_rate=config.burnin_learning_rate,
        burnin_steps=config.burnin_steps,
        min_learning_rate=config.min_learning_rate,
        staircase=config.staircase)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    learning_rate = learning_schedules.manual_stepping(
        global_step, learning_rate_step_boundaries,
        learning_rate_sequence, config.warmup)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    learning_rate = learning_schedules.cosine_decay_with_warmup(
        global_step,
        config.learning_rate_base,
        config.total_steps,
        config.warmup_learning_rate,
        config.warmup_steps,
        config.hold_base_rate_steps)

  if learning_rate is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return learning_rate
