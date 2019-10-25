# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Utility functions for training."""

import six

import tensorflow as tf
from deeplab.core import preprocess_utils


def _div_maybe_zero(total_loss, num_present):
  """Normalizes the total loss with the number of present pixels."""
  return tf.to_float(num_present > 0) * tf.div(total_loss,
                                               tf.maximum(1e-5, num_present))


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    hard_example_mining_step: An integer, the training step in which the hard
      exampling mining kicks off. Note that we gradually reduce the mining
      percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
      < 1.0, only compute the loss for the top k percent pixels (e.g., the top
      20% pixels). This is useful for hard pixel mining.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                               ignore_label)) * loss_weight
    one_hot_labels = tf.one_hot(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)

    if top_k_percent_pixels == 1.0:
      # Compute the loss for all pixels.
      tf.losses.softmax_cross_entropy(
          one_hot_labels,
          tf.reshape(logits, shape=[-1, num_classes]),
          weights=not_ignore_mask,
          scope=loss_scope)
    else:
      logits = tf.reshape(logits, shape=[-1, num_classes])
      weights = not_ignore_mask
      with tf.name_scope(loss_scope, 'softmax_hard_example_mining',
                         [logits, one_hot_labels, weights]):
        one_hot_labels = tf.stop_gradient(
            one_hot_labels, name='labels_stop_gradient')
        pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=logits,
            name='pixel_losses')
        weighted_pixel_losses = tf.multiply(pixel_losses, weights)
        num_pixels = tf.to_float(tf.shape(logits)[0])
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
          # Directly focus on the top_k pixels.
          top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
        else:
          # Gradually reduce the mining percent to top_k_percent_pixels.
          global_step = tf.to_float(tf.train.get_or_create_global_step())
          ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
          top_k_pixels = tf.to_int32(
              (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
        top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                      k=top_k_pixels,
                                      sorted=True,
                                      name='top_k_percent_pixels')
        total_loss = tf.reduce_sum(top_k_losses)
        num_present = tf.reduce_sum(
            tf.to_float(tf.not_equal(top_k_losses, 0.0)))
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = tf.contrib.framework.get_variables_to_restore(
      exclude=exclude_list)

  if variables_to_restore:
    init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
    global_step = tf.train.get_or_create_global_step()

    def restore_fn(unused_scaffold, sess):
      sess.run(init_op, init_feed_dict)
      sess.run([global_step])

    return restore_fn

  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in tf.model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none'):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  adjusted_global_step = global_step

  if slow_start_burnin_type != 'none':
    adjusted_global_step -= slow_start_step

  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        adjusted_global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        adjusted_global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  adjusted_slow_start_learning_rate = slow_start_learning_rate
  if slow_start_burnin_type == 'linear':
    # Do linear burnin. Increase linearly from slow_start_learning_rate and
    # reach base_learning_rate after (global_step >= slow_start_steps).
    adjusted_slow_start_learning_rate = (
        slow_start_learning_rate +
        (base_learning_rate - slow_start_learning_rate) *
        tf.to_float(global_step) / slow_start_step)
  elif slow_start_burnin_type != 'none':
    raise ValueError('Unknown burnin type.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step,
                  adjusted_slow_start_learning_rate, learning_rate)
