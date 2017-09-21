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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import imagenet
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
    '--train_batch_size', type=int, default=32, help='Batch size for training.')

parser.add_argument(
    '--eval_batch_size', type=int, default=100,
    help='Batch size for evaluation.')

parser.add_argument(
    '--first_cycle_steps', type=int, default=None,
    help='The number of steps to run before the first evaluation. Useful if '
    'you have stopped partway through a training cycle.')

FLAGS = parser.parse_args()
_EVAL_STEPS = 50000 // FLAGS.eval_batch_size

# Scale the learning rate linearly with the batch size. When the batch size is
# 256, the learning rate should be 0.1.
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.train_batch_size / 256

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

train_dataset = imagenet.get_split('train', FLAGS.data_dir)
eval_dataset = imagenet.get_split('validation', FLAGS.data_dir)

image_preprocessing_fn = vgg_preprocessing.preprocess_image
network = resnet_model.resnet_v2(
    resnet_size=FLAGS.resnet_size, num_classes=train_dataset.num_classes)

batches_per_epoch = train_dataset.num_samples / FLAGS.train_batch_size


def input_fn(is_training):
  """Input function which provides a single batch for train or eval."""
  batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size
  dataset = train_dataset if is_training else eval_dataset
  capacity_multiplier = 20 if is_training else 2
  min_multiplier = 10 if is_training else 1

  provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset=dataset,
      num_readers=4,
      common_queue_capacity=capacity_multiplier * batch_size,
      common_queue_min=min_multiplier * batch_size)

  image, label = provider.get(['image', 'label'])

  image = image_preprocessing_fn(image=image,
                                 output_height=network.default_image_size,
                                 output_width=network.default_image_size,
                                 is_training=is_training)

  images, labels = tf.train.batch(tensors=[image, label],
                                  batch_size=batch_size,
                                  num_threads=4,
                                  capacity=5 * batch_size)

  labels = tf.one_hot(labels, imagenet._NUM_CLASSES)
  return images, labels


def resnet_model_fn(features, labels, mode):
  """ Our model_fn for ResNet to be used with our Estimator."""
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

  for cycle in range(FLAGS.train_steps // FLAGS.steps_per_eval):
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
    eval_results = resnet_classifier.evaluate(
      input_fn=lambda: input_fn(False), steps=_EVAL_STEPS)
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
