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

"""Helper functions used for training AutoAugment models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def setup_loss(logits, labels):
  """Returns the cross entropy for the given `logits` and `labels`."""
  predictions = tf.nn.softmax(logits)
  cost = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                         logits=logits)
  return predictions, cost


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost


def eval_child_model(session, model, data_loader, mode):
  """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.
    mode: Will `model` either evaluate validation or test data.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
  if mode == 'val':
    images = data_loader.val_images
    labels = data_loader.val_labels
  elif mode == 'test':
    images = data_loader.test_images
    labels = data_loader.test_labels
  else:
    raise ValueError('Not valid eval mode')
  assert len(images) == len(labels)
  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  assert len(images) % model.batch_size == 0
  eval_batches = int(len(images) / model.batch_size)
  for i in range(eval_batches):
    eval_images = images[i * model.batch_size:(i + 1) * model.batch_size]
    eval_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]
    _ = session.run(
        model.eval_op,
        feed_dict={
            model.images: eval_images,
            model.labels: eval_labels,
        })
  return session.run(model.accuracy)


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
  """Cosine Learning rate.

  Args:
    learning_rate: Initial learning rate.
    epoch: Current epoch we are one. This is one based.
    iteration: Current batch in this epoch.
    batches_per_epoch: Batches per epoch.
    total_epochs: Total epochs you are training for.

  Returns:
    The learning rate to be used for this current batch.
  """
  t_total = total_epochs * batches_per_epoch
  t_cur = float(epoch * batches_per_epoch + iteration)
  return 0.5 * learning_rate * (1 + np.cos(np.pi * t_cur / t_total))


def get_lr(curr_epoch, hparams, iteration=None):
  """Returns the learning rate during training based on the current epoch."""
  assert iteration is not None
  batches_per_epoch = int(hparams.train_size / hparams.batch_size)
  lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,
                 hparams.num_epochs)
  return lr


def run_epoch_training(session, model, data_loader, curr_epoch):
  """Runs one epoch of training for the model passed in.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.
    curr_epoch: How many of epochs of training have been done so far.

  Returns:
    The accuracy of 'model' on the training set
  """
  steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
  tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
  curr_step = session.run(model.global_step)
  assert curr_step % steps_per_epoch == 0

  # Get the current learning rate for the model based on the current epoch
  curr_lr = get_lr(curr_epoch, model.hparams, iteration=0)
  tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))

  for step in xrange(steps_per_epoch):
    curr_lr = get_lr(curr_epoch, model.hparams, iteration=(step + 1))
    # Update the lr rate variable to the current LR.
    model.lr_rate_ph.load(curr_lr, session=session)
    if step % 20 == 0:
      tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

    train_images, train_labels = data_loader.next_batch()
    _, step, _ = session.run(
        [model.train_op, model.global_step, model.eval_op],
        feed_dict={
            model.images: train_images,
            model.labels: train_labels,
        })

  train_accuracy = session.run(model.accuracy)
  tf.logging.info('Train accuracy: {}'.format(train_accuracy))
  return train_accuracy
