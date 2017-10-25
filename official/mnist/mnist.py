#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images to process in a batch')

parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data',
                    help='Path to the MNIST data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/mnist_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=40,
                    help='Number of epochs to train.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def input_fn(mode, batch_size=1):
  """A simple input_fn using the contrib.data input pipeline."""

  def example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([28 * 28])

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32) / 255 - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, tf.one_hot(label, 10)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
  else:
    assert mode == tf.estimator.ModeKeys.EVAL, 'invalid mode'
    tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')

  assert tf.gfile.Exists(tfrecords_file), (
      'Run convert_to_records.py first to convert the MNIST data to TFRecord '
      'file format.')

  dataset = tf.contrib.data.TFRecordDataset([tfrecords_file])

  # For training, repeat the dataset forever
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()

  # Map example_parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(
      example_parser, num_threads=1, output_buffer_size=batch_size)
  dataset = dataset.batch(batch_size)
  images, labels = dataset.make_one_shot_iterator().get_next()

  return images, labels


def mnist_model(inputs, mode):
  """Takes the MNIST inputs and mode and outputs a tensor of logits."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  inputs = tf.reshape(inputs, [-1, 28, 28, 1])
  data_format = FLAGS.data_format

  if data_format is None:
    # When running on GPU, transpose the data from channels_last (NHWC) to
    # channels_first (NCHW) to improve performance.
    # See https://www.tensorflow.org/performance/performance_guide#data_formats
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else
                   'channels_last')

  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
  return logits


def mnist_model_fn(features, labels, mode):
  """Model function for MNIST."""
  logits = mnist_model(features, mode)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  # Configure the training op
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
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


def main(unused_argv):
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=mnist_model_fn, model_dir=FLAGS.model_dir)

  # Train the model
  tensors_to_log = {
      'train_accuracy': 'train_accuracy'
  }

  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  batches_per_epoch = _NUM_IMAGES['train'] / FLAGS.batch_size

  mnist_classifier.train(
      input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN, FLAGS.batch_size),
      steps=FLAGS.train_epochs * batches_per_epoch,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL))
  print()
  print('Evaluation results:\n    %s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
