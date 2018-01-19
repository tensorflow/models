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
"""Functions for running Resnet that are shared across datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf


def resnet_model_fn(features, labels, mode, params, model_class):
  """Shared functionality for different resnet model_fns."""

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  model = model_class(params['resnet_size'], params['data_format'])
  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / params['batch_denom']
    batches_per_epoch = params['train_images'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in params['epochs']]
    vals = [initial_learning_rate * decay for decay in params['learning_rates']]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, vals)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def resnet_main(flags, model_function, input_function):
    # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, config=run_config,
      params={
          'resnet_size': flags.resnet_size,
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
      })

  for _ in range(flags.train_epochs // flags.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    classifier.train(
        input_fn=lambda: input_function(
            True, flags.data_dir, flags.batch_size, flags.epochs_per_eval),
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = classifier.evaluate(input_fn=lambda: input_function(
        False, flags.data_dir, flags.batch_size))
    print(eval_results)


class ResnetArgParser(argparse.ArgumentParser):
  """
  Arguments for configuring and running a Resnet Model.
  """

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__()
    self._set_args(resnet_size_choices)

  def _set_args(self, resnet_size_choices=None):
    self.add_argument(
        '--data_dir', type=str, default='/tmp/resnet_data',
        help='The directory where the input data is stored.')

    self.add_argument(
        '--model_dir', type=str, default='/tmp/resnet_model',
        help='The directory where the model will be stored.')

    self.add_argument(
        '--resnet_size', type=int, default=50,
        choices=resnet_size_choices,
        help='The size of the ResNet model to use.')

    self.add_argument(
        '--train_epochs', type=int, default=100,
        help='The number of epochs to use for training.')

    self.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')

    self.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training and evaluation.')

    self.add_argument(
        '--data_format', type=str, default=None,
        choices=['channels_first', 'channels_last'],
        help='A flag to override the data format used in the model. '
             'channels_first provides a performance boost on GPU but '
             'is not always compatible with CPU. If left unspecified, '
             'the data format will be chosen automatically based on '
             'whether TensorFlow was built for CPU or GPU.')


