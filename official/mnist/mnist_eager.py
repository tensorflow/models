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
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import mnist
import dataset

FLAGS = None


def loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(model, optimizer, dataset, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  global_step = tf.train.get_or_create_global_step()

  start = time.time()
  for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
      # Record the operations used to compute the loss given the input,
      # so that the gradient of the loss with respect to the variables
      # can be computed.
      with tfe.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss(logits, labels)
        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=global_step)
      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
        start = time.time()


def test(model, dataset):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  avg_loss = tfe.metrics.Mean('loss')
  accuracy = tfe.metrics.Accuracy('accuracy')

  for (images, labels) in tfe.Iterator(dataset):
    logits = model(images, training=False)
    avg_loss(loss(logits, labels))
    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar('loss', avg_loss.result())
    tf.contrib.summary.scalar('accuracy', accuracy.result())


def main(_):
  tfe.enable_eager_execution()

  (device, data_format) = ('/gpu:0', 'channels_first')
  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
    (device, data_format) = ('/cpu:0', 'channels_last')
  print('Using device %s, and data format %s.' % (device, data_format))

  # Load the datasets
  train_ds = dataset.train(FLAGS.data_dir).shuffle(60000).batch(
      FLAGS.batch_size)
  test_ds = dataset.test(FLAGS.data_dir).batch(FLAGS.batch_size)

  # Create the model and optimizer
  model = mnist.Model(data_format)
  optimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)

  if FLAGS.output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(FLAGS.output_dir, 'train')
    test_dir = os.path.join(FLAGS.output_dir, 'eval')
    tf.gfile.MakeDirs(FLAGS.output_dir)
  else:
    train_dir = None
    test_dir = None
  summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')
  checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')

  # Train and evaluate for 11 epochs.
  with tf.device(device):
    for epoch in range(1, 11):
      with tfe.restore_variables_on_create(
          tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):
        global_step = tf.train.get_or_create_global_step()
        start = time.time()
        with summary_writer.as_default():
          train(model, optimizer, train_ds, FLAGS.log_interval)
        end = time.time()
        print('\nTrain time for epoch #%d (global step %d): %f' %
              (epoch, global_step.numpy(), end - start))
      with test_summary_writer.as_default():
        test(model, test_ds)
      all_variables = (model.variables + optimizer.variables() + [global_step])
      tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      metavar='N',
      help='input batch size for training (default: 100)')
  parser.add_argument(
      '--log_interval',
      type=int,
      default=10,
      metavar='N',
      help='how many batches to wait before logging training status')
  parser.add_argument(
      '--output_dir',
      type=str,
      default=None,
      metavar='N',
      help='Directory to write TensorBoard summaries')
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='/tmp/tensorflow/mnist/checkpoints/',
      metavar='N',
      help='Directory to save checkpoints in (once per epoch)')
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      metavar='LR',
      help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      metavar='M',
      help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--no_gpu',
      action='store_true',
      default=False,
      help='disables GPU usage even if a GPU is available')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
