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
from google3.third_party.tensorflow_models.domain_adaptation.domain_separation import losses
from google3.third_party.tensorflow_models.domain_adaptation.domain_separation import models

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
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

tf.app.flags.DEFINE_string('basic_tower', 'dsn_cropped_linemod',
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


def main(_):
  g = tf.Graph()
  with g.as_default():
    images, labels = data_provider.provide(FLAGS.dataset, FLAGS.portion,
                                           FLAGS.batch_size)

    num_classes = labels['classes'].get_shape().as_list()[1]

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
