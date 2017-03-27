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
r"""Evaluation for Domain Separation Networks (DSNs).

To build locally for CPU:
  blaze build -c opt --copt=-mavx \
    third_party/tensorflow_models/domain_adaptation/domain_separation:dsn_eval

To build locally for GPU:
  blaze build -c opt --copt=-mavx --config=cuda_clang \
    third_party/tensorflow_models/domain_adaptation/domain_separation:dsn_eval

To run locally:
$
./blaze-bin/third_party/tensorflow_models/domain_adaptation/domain_separation/dsn_eval
\
    --alsologtostderr
"""
# pylint: enable=line-too-long
import math

import google3

import numpy as np
import tensorflow as tf
from google3.robotics.cad_learning.domain_adaptation.fnist import data_provider
from google3.third_party.tensorflow_models.domain_adaptation.domain_separation import models

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_string('master', 'local',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/da/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/da/',
    'Directory where we should write the tf summaries to.')

tf.app.flags.DEFINE_string(
    'dataset', 'pose_real',
    'Which dataset to test on: "pose_real", "pose_synthetic".')
tf.app.flags.DEFINE_string('portion', 'valid',
                           'Which portion to test on: "valid", "test".')
tf.app.flags.DEFINE_integer('num_examples', 1000, 'Number of test examples.')

tf.app.flags.DEFINE_string('basic_tower', 'pose_mini',
                           'The basic tower building block.')
tf.app.flags.DEFINE_bool('use_logging', False, 'Debugging messages.')


def quaternion_metric(predictions, labels):
  product = tf.multiply(predictions, labels)
  internal_dot_products = tf.reduce_sum(product, [1])
  logcost = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
  return tf.contrib.metrics.streaming_mean(logcost)


def to_degrees(predictions, labels):
  """Converts a log quaternion distance to an angle.

  Args:
    log_quaternion_loss: The log quaternion distance between two
      unit quaternions (or a batch of pairs of quaternions).

  Returns:
    The angle in degrees of the implied angle-axis representation.
  """
  product = tf.multiply(predictions, labels)
  internal_dot_products = tf.reduce_sum(product, [1])
  log_quaternion_loss = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
  angle_loss = tf.acos(-(tf.exp(log_quaternion_loss) - 1)) * 2 * 180 / math.pi
  return tf.contrib.metrics.streaming_mean(angle_loss)


def main(_):
  g = tf.Graph()
  with g.as_default():
    images, labels = data_provider.provide(FLAGS.dataset, FLAGS.portion,
                                           FLAGS.batch_size)

    num_classes = labels['classes'].shape[1]

    # Define the model:
    with tf.variable_scope('towers'):
      basic_tower = models.provide(FLAGS.basic_tower)
      predictions, endpoints = basic_tower(
          images, is_training=False, num_classes=num_classes)
    names_to_values = {}
    names_to_updates = {}
    # Define the metrics:
    if 'quaternions' in labels:  # Also have to evaluate pose estimation!
      quaternion_loss = quaternion_metric(labels['quaternions'],
                                          endpoints['quaternion_pred'])

      metric_name = 'Angle Mean Error'
      names_to_values[metric_name], names_to_updates[metric_name] = to_degrees(
          labels['quaternions'], endpoints['quaternion_pred'])

      metric_name = 'Log Quaternion Error'
      names_to_values[metric_name], names_to_updates[
          metric_name] = quaternion_metric(labels['quaternions'],
                                           endpoints['quaternion_pred'])
      metric_name = 'Accuracy'
      names_to_values[metric_name], names_to_updates[
          metric_name] = tf.contrib.metrics.streaming_accuracy(
              tf.argmax(predictions, 1), tf.argmax(labels['classes'], 1))

    metric_name = 'Accuracy'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(predictions, 1), tf.argmax(labels['classes'], 1))

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
      op = tf.contrib.deprecated.scalar_summary(metric_name, metric_value)
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
        summary_op=tf.contrib.deprecated.merge_summary(summary_ops))


if __name__ == '__main__':
  tf.app.run()
