# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

# pylint: disable=line-too-long
"""Evaluation for Domain Separation Networks (DSNs)."""
# pylint: enable=line-too-long
import math

import numpy as np
from six.moves import xrange
import tensorflow as tf

from domain_adaptation.datasets import dataset_factory
from domain_adaptation.domain_separation import losses
from domain_adaptation.domain_separation import models

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_string('master', '',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/da/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/da/',
    'Directory where we should write the tf summaries to.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('dataset', 'mnist_m',
                           'Which dataset to test on: "mnist", "mnist_m".')

tf.app.flags.DEFINE_string('split', 'valid',
                           'Which portion to test on: "valid", "test".')

tf.app.flags.DEFINE_integer('num_examples', 1000, 'Number of test examples.')

tf.app.flags.DEFINE_string('basic_tower', 'dann_mnist',
                           'The basic tower building block.')

tf.app.flags.DEFINE_bool('enable_precision_recall', False,
                         'If True, precision and recall for each class will '
                         'be added to the metrics.')

tf.app.flags.DEFINE_bool('use_logging', False, 'Debugging messages.')


def quaternion_metric(predictions, labels):
  params = {'batch_size': FLAGS.batch_size, 'use_logging': False}
  logcost = losses.log_quaternion_loss_batch(predictions, labels, params)
  return slim.metrics.streaming_mean(logcost)


def angle_diff(true_q, pred_q):
  angles = 2 * (
      180.0 /
      np.pi) * np.arccos(np.abs(np.sum(np.multiply(pred_q, true_q), axis=1)))
  return angles


def provide_batch_fn():
  """ The provide_batch function to use. """
  return dataset_factory.provide_batch


def main(_):
  g = tf.Graph()
  with g.as_default():
    # Load the data.
    images, labels = provide_batch_fn()(
        FLAGS.dataset, FLAGS.split, FLAGS.dataset_dir, 4, FLAGS.batch_size, 4)

    num_classes = labels['classes'].get_shape().as_list()[1]

    tf.summary.image('eval_images', images, max_outputs=3)

    # Define the model:
    with tf.variable_scope('towers'):
      basic_tower = getattr(models, FLAGS.basic_tower)
      predictions, endpoints = basic_tower(
          images,
          num_classes=num_classes,
          is_training=False,
          batch_norm_params=None)
    metric_names_to_values = {}

    # Define the metrics:
    if 'quaternions' in labels:  # Also have to evaluate pose estimation!
      quaternion_loss = quaternion_metric(labels['quaternions'],
                                          endpoints['quaternion_pred'])

      angle_errors, = tf.py_func(
          angle_diff, [labels['quaternions'], endpoints['quaternion_pred']],
          [tf.float32])

      metric_names_to_values[
          'Angular mean error'] = slim.metrics.streaming_mean(angle_errors)
      metric_names_to_values['Quaternion Loss'] = quaternion_loss

    accuracy = tf.contrib.metrics.streaming_accuracy(
        tf.argmax(predictions, 1), tf.argmax(labels['classes'], 1))

    predictions = tf.argmax(predictions, 1)
    labels = tf.argmax(labels['classes'], 1)
    metric_names_to_values['Accuracy'] = accuracy

    if FLAGS.enable_precision_recall:
      for i in xrange(num_classes):
        index_map = tf.one_hot(i, depth=num_classes)
        name = 'PR/Precision_{}'.format(i)
        metric_names_to_values[name] = slim.metrics.streaming_precision(
            tf.gather(index_map, predictions), tf.gather(index_map, labels))
        name = 'PR/Recall_{}'.format(i)
        metric_names_to_values[name] = slim.metrics.streaming_recall(
            tf.gather(index_map, predictions), tf.gather(index_map, labels))

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
        metric_names_to_values)

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
      op = tf.summary.scalar(metric_name, metric_value)
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))

    # Setup the global step.
    slim.get_or_create_global_step()
    slim.evaluation.evaluation_loop(
        FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        summary_op=tf.summary.merge(summary_ops))


if __name__ == '__main__':
  tf.app.run()
