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

import tensorflow as tf  # pylint: disable=g-bad-import-order
import tensorflow.contrib.eager as tfe  # pylint: disable=g-bad-import-order

from official.mnist import dataset as mnist_dataset
from official.mnist import mnist
from official.utils.arg_parsers import parsers


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


def train(model, optimizer, dataset, step_counter, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  start = time.time()
  for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    with tf.contrib.summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
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
          zip(grads, model.variables), global_step=step_counter)
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


def main(argv):
  parser = MNISTEagerArgParser()
  flags = parser.parse_args(args=argv[1:])

  tfe.enable_eager_execution()

  # Automatically determine device and data_format
  (device, data_format) = ('/gpu:0', 'channels_first')
  if flags.no_gpu or tfe.num_gpus() <= 0:
    (device, data_format) = ('/cpu:0', 'channels_last')
  # If data_format is defined in FLAGS, overwrite automatically set value.
  if flags.data_format is not None:
    data_format = flags.data_format
  print('Using device %s, and data format %s.' % (device, data_format))

  # Load the datasets
  train_ds = mnist_dataset.train(flags.data_dir).shuffle(60000).batch(
      flags.batch_size)
  test_ds = mnist_dataset.test(flags.data_dir).batch(flags.batch_size)

  # Create the model and optimizer
  model = mnist.create_model(data_format)
  optimizer = tf.train.MomentumOptimizer(flags.lr, flags.momentum)

  # Create file writers for writing TensorBoard summaries.
  if flags.output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(flags.output_dir, 'train')
    test_dir = os.path.join(flags.output_dir, 'eval')
    tf.gfile.MakeDirs(flags.output_dir)
  else:
    train_dir = None
    test_dir = None
  summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  checkpoint_prefix = os.path.join(flags.model_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tfe.Checkpoint(
      model=model, optimizer=optimizer, step_counter=step_counter)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(flags.model_dir))

  # Train and evaluate for a set number of epochs.
  with tf.device(device):
    for _ in range(flags.train_epochs):
      start = time.time()
      with summary_writer.as_default():
        train(model, optimizer, train_ds, step_counter, flags.log_interval)
      end = time.time()
      print('\nTrain time for epoch #%d (%d total steps): %f' %
            (checkpoint.save_counter.numpy() + 1,
             step_counter.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, test_ds)
      checkpoint.save(checkpoint_prefix)


class MNISTEagerArgParser(argparse.ArgumentParser):
  """Argument parser for running MNIST model with eager training loop."""

  def __init__(self):
    super(MNISTEagerArgParser, self).__init__(parents=[
        parsers.EagerParser(),
        parsers.ImageModelParser()])

    self.add_argument(
        '--log_interval', '-li',
        type=int,
        default=10,
        metavar='N',
        help='[default: %(default)s] batches between logging training status')
    self.add_argument(
        '--output_dir', '-od',
        type=str,
        default=None,
        metavar='<OD>',
        help='[default: %(default)s] Directory to write TensorBoard summaries')
    self.add_argument(
        '--lr', '-lr',
        type=float,
        default=0.01,
        metavar='<LR>',
        help='[default: %(default)s] learning rate')
    self.add_argument(
        '--momentum', '-m',
        type=float,
        default=0.5,
        metavar='<M>',
        help='[default: %(default)s] SGD momentum')
    self.add_argument(
        '--no_gpu', '-nogpu',
        action='store_true',
        default=False,
        help='disables GPU usage even if a GPU is available')

    self.set_defaults(
        data_dir='/tmp/tensorflow/mnist/input_data',
        model_dir='/tmp/tensorflow/mnist/checkpoints/',
        batch_size=100,
        train_epochs=10,
    )

if __name__ == '__main__':
  main(argv=sys.argv)
