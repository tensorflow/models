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
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.mnist import dataset
from official.utils.arg_parsers import parsers
from official.utils.logging import hooks_helper

LEARNING_RATE = 1e-4


class Model(tf.keras.Model):
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But written as a tf.keras.Model using the tf.layers API.
  """

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
        'channels_first' is typically faster on GPUs while 'channels_last' is
        typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
    """
    super(Model, self).__init__()
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
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels,
                    predictions=tf.argmax(logits, axis=1)),
        })


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: the number of examples processed in each training batch.

  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)


def main(argv):
  parser = MNISTArgParser()
  flags = parser.parse_args(args=argv[1:])

  model_function = model_fn

  if flags.multi_gpu:
    validate_batch_size_for_multi_gpu(flags.batch_size)

    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
        model_fn, loss_reduction=tf.losses.Reduction.MEAN)

  data_format = flags.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_function,
      model_dir=flags.model_dir,
      params={
          'data_format': data_format,
          'multi_gpu': flags.multi_gpu
      })

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(flags.data_dir)
    ds = ds.cache().shuffle(buffer_size=50000).batch(flags.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds.repeat(flags.epochs_between_evals)
    return ds

  def eval_input_fn():
    return dataset.test(flags.data_dir).batch(
        flags.batch_size).make_one_shot_iterator().get_next()

  # Set up hook that outputs training logs every 100 steps.
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size)

  # Train and evaluate model.
  for _ in range(flags.train_epochs // flags.epochs_between_evals):
    mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

  # Export the model
  if flags.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(flags.export_dir, input_fn)


class MNISTArgParser(argparse.ArgumentParser):
  """Argument parser for running MNIST model."""

  def __init__(self):
    super(MNISTArgParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.ImageModelParser()])

    self.add_argument(
        '--export_dir',
        type=str,
        help='[default: %(default)s] If set, a SavedModel serialization of the '
             'model will be exported to this directory at the end of training. '
             'See the README for more details and relevant links.')

    self.set_defaults(
        data_dir='/tmp/mnist_data',
        model_dir='/tmp/mnist_model',
        batch_size=100,
        train_epochs=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
