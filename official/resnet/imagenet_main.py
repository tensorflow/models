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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import resnet_model
import vgg_preprocessing

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='',
    help='The directory where the ImageNet input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default='/tmp/resnet_model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=50, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_steps', type=int, default=6400000,
    help='The number of steps to use for training.')

parser.add_argument(
    '--steps_per_eval', type=int, default=40000,
    help='The number of training steps to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--map_threads', type=int, default=5,
    help='The number of threads for dataset.map.')

parser.add_argument(
    '--first_cycle_steps', type=int, default=None,
    help='The number of steps to run before the first evaluation. Useful if '
    'you have stopped partway through a training cycle.')

FLAGS = parser.parse_args()

# Scale the learning rate linearly with the batch size. When the batch size is
# 256, the learning rate should be 0.1.
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 256

_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

image_preprocessing_fn = vgg_preprocessing.preprocess_image
network = resnet_model.resnet_v2(
    resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)

batches_per_epoch = _NUM_IMAGES['train'] / FLAGS.batch_size


def filenames(is_training):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(FLAGS.data_dir, 'train-%05d-of-01024' % i)
        for i in range(0, 1024)]
  else:
    return [
        os.path.join(FLAGS.data_dir, 'validation-%05d-of-00128' % i)
        for i in range(0, 128)]


def dataset_parser(value, is_training):
  """Parse an Imagenet record from value."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = image_preprocessing_fn(
      image=image,
      output_height=network.default_image_size,
      output_width=network.default_image_size,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training):
  """Input function which provides a single batch for train or eval."""
  dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames(is_training))
  if is_training:
    dataset = dataset.shuffle(buffer_size=1024)
  dataset = dataset.flat_map(tf.contrib.data.TFRecordDataset)

  if is_training:
    dataset = dataset.repeat()

  dataset = dataset.map(lambda value: dataset_parser(value, is_training),
                        num_threads=FLAGS.map_threads,
                        output_buffer_size=FLAGS.batch_size)

  if is_training:
    buffer_size = 1250 + 2 * FLAGS.batch_size
    dataset = dataset.shuffle(buffer_size=buffer_size)

  iterator = dataset.batch(FLAGS.batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def resnet_model_fn(features, labels, mode):
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

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

  # Add weight decay to the loss. We perform weight decay on all trainable
  # variables, which includes batch norm beta and gamma variables.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 120, and 150 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [30, 60, 120, 150]]
    values = [
        _INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  resnet_classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn, model_dir=FLAGS.model_dir)

  for _ in range(FLAGS.train_steps // FLAGS.steps_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(True),
        steps=FLAGS.first_cycle_steps or FLAGS.steps_per_eval,
        hooks=[logging_hook])
    FLAGS.first_cycle_steps = None

    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(input_fn=lambda: input_fn(False))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
