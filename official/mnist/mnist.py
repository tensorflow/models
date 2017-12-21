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
from tensorflow.examples.tutorials.mnist import input_data


def train_dataset(data_dir):
  """Returns a tf.data.Dataset yielding (image, label) pairs for training."""
  data = input_data.read_data_sets(data_dir, one_hot=True).train
  return tf.data.Dataset.from_tensor_slices((data.images, data.labels))


def eval_dataset(data_dir):
  """Returns a tf.data.Dataset yielding (image, label) pairs for evaluation."""
  data = input_data.read_data_sets(data_dir, one_hot=True).test
  return tf.data.Dataset.from_tensors((data.images, data.labels))


class Model(object):
  """Class that defines a graph to recognize digits in the MNIST dataset."""

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
        'channels_first' is typically faster on GPUs while 'channels_last' is
        typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
    """
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == 'channels_last'
      self._input_shape = [-1, 28, 28, 1]

    self.conv1 = tf.layers.Conv2D(
        32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
    self.conv2 = tf.layers.Conv2D(
        64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
    self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.fc2 = tf.layers.Dense(10)
    self.dropout = tf.layers.Dropout(0.4)
    self.max_pool2d = tf.layers.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, 10].
    """
    y = tf.reshape(inputs, self._input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = tf.layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training=training)
    return self.fc2(y)


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = Model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    logits = model(image, training=True)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
    # Name the accuracy tensor 'train_accuracy' to demonstrate the
    # LoggingTensorHook.
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=tf.argmax(labels, axis=1),
                    predictions=tf.argmax(logits, axis=1)),
        })


def main(unused_argv):
  data_format = FLAGS.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'data_format': data_format
      })

  # Train the model
  def train_input_fn():
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    dataset = train_dataset(FLAGS.data_dir)
    dataset = dataset.shuffle(buffer_size=50000).batch(FLAGS.batch_size).repeat(
        FLAGS.train_epochs)
    (images, labels) = dataset.make_one_shot_iterator().get_next()
    return (images, labels)

  # Set up training hook that logs the training accuracy every 100 steps.
  tensors_to_log = {'train_accuracy': 'train_accuracy'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

  # Evaluate the model and print results
  def eval_input_fn():
    return eval_dataset(FLAGS.data_dir).make_one_shot_iterator().get_next()

  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print()
  print('Evaluation results:\n\t%s' % eval_results)

  # Export the model
  if FLAGS.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Number of images to process in a batch')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/mnist_data',
      help='Path to directory containing the MNIST dataset')
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/mnist_model',
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--train_epochs', type=int, default=40, help='Number of epochs to train.')
  parser.add_argument(
      '--data_format',
      type=str,
      default=None,
      choices=['channels_first', 'channels_last'],
      help='A flag to override the data format used in the model. channels_first '
      'provides a performance boost on GPU but is not always compatible '
      'with CPU. If left unspecified, the data format will be chosen '
      'automatically based on whether TensorFlow was built for CPU or GPU.')
  parser.add_argument(
      '--export_dir',
      type=str,
      help='The directory where the exported SavedModel will be stored.')

  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
