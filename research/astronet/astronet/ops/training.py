# Copyright 2018 The TensorFlow Authors.
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

"""Functions for training an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_learning_rate(hparams, global_step):
  """Creates a learning rate Tensor.

  Args:
    hparams: ConfigDict containing the learning rate configuration.
    global_step: The global step Tensor.

  Returns:
    A learning rate Tensor.
  """
  if hparams.get("learning_rate_decay_factor"):
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(hparams.learning_rate),
        global_step=global_step,
        decay_steps=hparams.learning_rate_decay_steps,
        decay_rate=hparams.learning_rate_decay_factor,
        staircase=hparams.learning_rate_decay_staircase)
  else:
    learning_rate = tf.constant(hparams.learning_rate)

  return learning_rate


def create_optimizer(hparams, learning_rate, use_tpu=False):
  """Creates a TensorFlow Optimizer.

  Args:
    hparams: ConfigDict containing the optimizer configuration.
    learning_rate: A Python float or a scalar Tensor.
    use_tpu: If True, the returned optimizer is wrapped in a
      CrossShardOptimizer.

  Returns:
    A TensorFlow optimizer.

  Raises:
    ValueError: If hparams.optimizer is unrecognized.
  """
  optimizer_name = hparams.optimizer.lower()
  if optimizer_name == "momentum":
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=hparams.get("momentum", 0.9),
        use_nesterov=hparams.get("use_nesterov", False))
  elif optimizer_name == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer_name == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif optimizer_name == "rmsprop":
    optimizer = tf.RMSPropOptimizer(learning_rate)
  else:
    raise ValueError("Unknown optimizer: {}".format(hparams.optimizer))

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  return optimizer


def create_train_op(model, optimizer):
  """Creates a Tensor to train the model.

  Args:
    model: Instance of AstroModel.
    optimizer: Instance of tf.train.Optimizer.

  Returns:
    A Tensor that runs a single training step and returns model.total_loss.
  """
  # Maybe clip gradient norms.
  transform_grads_fn = None
  if model.hparams.get("clip_grad_norm"):
    transform_grads_fn = tf.contrib.training.clip_gradient_norms_fn(
        model.hparams.clip_gradient_norm)

  # Create train op.
  return tf.contrib.training.create_train_op(
      total_loss=model.total_loss,
      optimizer=optimizer,
      global_step=model.global_step,
      transform_grads_fn=transform_grads_fn)
