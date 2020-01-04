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

import os
import time

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from tensorflow.python import eager as tfe
# pylint: enable=g-bad-import-order

from official.r1.mnist import dataset as mnist_dataset
from official.r1.mnist import mnist
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers


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
  from tensorflow.contrib import summary as contrib_summary  # pylint: disable=g-import-not-at-top

  start = time.time()
  for (batch, (images, labels)) in enumerate(dataset):
    with contrib_summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
      # Record the operations used to compute the loss given the input,
      # so that the gradient of the loss with respect to the variables
      # can be computed.
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss(logits, labels)
        contrib_summary.scalar('loss', loss_value)
        contrib_summary.scalar('accuracy',
                                    compute_accuracy(logits, labels))
      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=step_counter)
      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
        start = time.time()


def test(model, dataset):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  from tensorflow.contrib import summary as contrib_summary  # pylint: disable=g-import-not-at-top
  avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tf.keras.metrics.Accuracy('accuracy', dtype=tf.float32)

  for (images, labels) in dataset:
    logits = model(images, training=False)
    avg_loss.update_state(loss(logits, labels))
    accuracy.update_state(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))
  with contrib_summary.always_record_summaries():
    contrib_summary.scalar('loss', avg_loss.result())
    contrib_summary.scalar('accuracy', accuracy.result())


def run_mnist_eager(flags_obj):
  """Run MNIST training and eval loop in eager mode.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  tf.enable_eager_execution()
  model_helpers.apply_clean(flags.FLAGS)

  # Automatically determine device and data_format
  (device, data_format) = ('/gpu:0', 'channels_first')
  if flags_obj.no_gpu or not tf.test.is_gpu_available():
    (device, data_format) = ('/cpu:0', 'channels_last')
  # If data_format is defined in FLAGS, overwrite automatically set value.
  if flags_obj.data_format is not None:
    data_format = flags_obj.data_format
  print('Using device %s, and data format %s.' % (device, data_format))

  # Load the datasets
  train_ds = mnist_dataset.train(flags_obj.data_dir).shuffle(60000).batch(
      flags_obj.batch_size)
  test_ds = mnist_dataset.test(flags_obj.data_dir).batch(
      flags_obj.batch_size)

  # Create the model and optimizer
  model = mnist.create_model(data_format)
  optimizer = tf.train.MomentumOptimizer(flags_obj.lr, flags_obj.momentum)

  # Create file writers for writing TensorBoard summaries.
  if flags_obj.output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(flags_obj.output_dir, 'train')
    test_dir = os.path.join(flags_obj.output_dir, 'eval')
    tf.gfile.MakeDirs(flags_obj.output_dir)
  else:
    train_dir = None
    test_dir = None
  summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=10000)
  test_summary_writer = tf.compat.v2.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  checkpoint_prefix = os.path.join(flags_obj.model_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
      model=model, optimizer=optimizer, step_counter=step_counter)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(flags_obj.model_dir))

  # Train and evaluate for a set number of epochs.
  with tf.device(device):
    for _ in range(flags_obj.train_epochs):
      start = time.time()
      with summary_writer.as_default():
        train(model, optimizer, train_ds, step_counter,
              flags_obj.log_interval)
      end = time.time()
      print('\nTrain time for epoch #%d (%d total steps): %f' %
            (checkpoint.save_counter.numpy() + 1,
             step_counter.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, test_ds)
      checkpoint.save(checkpoint_prefix)


def define_mnist_eager_flags():
  """Defined flags and defaults for MNIST in eager mode."""
  flags_core.define_base(clean=True, train_epochs=True, export_dir=True,
                         distribution_strategy=True)
  flags_core.define_image()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_integer(
      name='log_interval', short_name='li', default=10,
      help=flags_core.help_wrap('batches between logging training status'))

  flags.DEFINE_string(
      name='output_dir', short_name='od', default=None,
      help=flags_core.help_wrap('Directory to write TensorBoard summaries'))

  flags.DEFINE_float(name='learning_rate', short_name='lr', default=0.01,
                     help=flags_core.help_wrap('Learning rate.'))

  flags.DEFINE_float(name='momentum', short_name='m', default=0.5,
                     help=flags_core.help_wrap('SGD momentum.'))

  flags.DEFINE_bool(name='no_gpu', short_name='nogpu', default=False,
                    help=flags_core.help_wrap(
                        'disables GPU usage even if a GPU is available'))

  flags_core.set_defaults(
      data_dir='/tmp/tensorflow/mnist/input_data',
      model_dir='/tmp/tensorflow/mnist/checkpoints/',
      batch_size=100,
      train_epochs=10,
  )


def main(_):
  run_mnist_eager(flags.FLAGS)


if __name__ == '__main__':
  define_mnist_eager_flags()
  absl_app.run(main=main)
