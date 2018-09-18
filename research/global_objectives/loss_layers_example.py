# Copyright 2018 The TensorFlow Global Objectives Authors. All Rights Reserved.
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
"""Example for using global objectives.

Illustrate, using synthetic data, how using the precision_at_recall loss
significanly improves the performace of a linear classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from sklearn.metrics import precision_score
import tensorflow as tf
from global_objectives import loss_layers

# When optimizing using global_objectives, if set to True then the saddle point
# optimization steps are performed internally by the Tensorflow optimizer,
# otherwise by dedicated saddle-point steps as part of the optimization loop.
USE_GO_SADDLE_POINT_OPT = False

TARGET_RECALL = 0.98
TRAIN_ITERATIONS = 150
LEARNING_RATE = 1.0
GO_DUAL_RATE_FACTOR = 15.0
NUM_CHECKPOINTS = 6

EXPERIMENT_DATA_CONFIG = {
    'positives_centers': [[0, 1.0], [1, -0.5]],
    'negatives_centers': [[0, -0.5], [1, 1.0]],
    'positives_variances': [0.15, 0.1],
    'negatives_variances': [0.15, 0.1],
    'positives_counts': [500, 50],
    'negatives_counts': [3000, 100]
}


def create_training_and_eval_data_for_experiment(**data_config):
  """Creates train and eval data sets.

  Note: The synthesized binary-labeled data is a mixture of four Gaussians - two
    positives and two negatives. The centers, variances, and sizes for each of
    the two positives and negatives mixtures are passed in the respective keys
    of data_config:

  Args:
      **data_config: Dictionary with Array entries as follows:
        positives_centers - float [2,2] two centers of positives data sets.
        negatives_centers - float [2,2] two centers of negatives data sets.
        positives_variances - float [2] Variances for the positives sets.
        negatives_variances - float [2] Variances for the negatives sets.
        positives_counts - int [2] Counts for each of the two positives sets.
        negatives_counts - int [2] Counts for each of the two negatives sets.

  Returns:
    A dictionary with two shuffled data sets created - one for training and one
    for eval. The dictionary keys are 'train_data', 'train_labels', 'eval_data',
    and 'eval_labels'. The data points are two-dimentional floats, and the
    labels are in {0,1}.
  """
  def data_points(is_positives, index):
    variance = data_config['positives_variances'
                           if is_positives else 'negatives_variances'][index]
    center = data_config['positives_centers'
                         if is_positives else 'negatives_centers'][index]
    count = data_config['positives_counts'
                        if is_positives else 'negatives_counts'][index]
    return variance*np.random.randn(count, 2) + np.array([center])

  def create_data():
    return np.concatenate([data_points(False, 0),
                           data_points(True, 0),
                           data_points(True, 1),
                           data_points(False, 1)], axis=0)

  def create_labels():
    """Creates an array of 0.0 or 1.0 labels for the data_config batches."""
    return np.array([0.0]*data_config['negatives_counts'][0] +
                    [1.0]*data_config['positives_counts'][0] +
                    [1.0]*data_config['positives_counts'][1] +
                    [0.0]*data_config['negatives_counts'][1])

  permutation = np.random.permutation(
      sum(data_config['positives_counts'] + data_config['negatives_counts']))

  train_data = create_data()[permutation, :]
  eval_data = create_data()[permutation, :]
  train_labels = create_labels()[permutation]
  eval_labels = create_labels()[permutation]

  return {
      'train_data': train_data,
      'train_labels': train_labels,
      'eval_data': eval_data,
      'eval_labels': eval_labels
  }


def train_model(data, use_global_objectives):
  """Trains a linear model for maximal accuracy or precision at given recall."""

  def precision_at_recall(scores, labels, target_recall):
    """Computes precision - at target recall - over data."""
    positive_scores = scores[labels == 1.0]
    threshold = np.percentile(positive_scores, 100 - target_recall*100)
    predicted = scores >= threshold
    return precision_score(labels, predicted)

  w = tf.Variable(tf.constant([-1.0, -1.0], shape=[2, 1]), trainable=True,
                  name='weights', dtype=tf.float32)
  b = tf.Variable(tf.zeros([1]), trainable=True, name='biases',
                  dtype=tf.float32)

  logits = tf.matmul(tf.cast(data['train_data'], tf.float32), w) + b

  labels = tf.constant(
      data['train_labels'],
      shape=[len(data['train_labels']), 1],
      dtype=tf.float32)

  if use_global_objectives:
    loss, other_outputs = loss_layers.precision_at_recall_loss(
        labels, logits,
        TARGET_RECALL,
        dual_rate_factor=GO_DUAL_RATE_FACTOR)
    loss = tf.reduce_mean(loss)
  else:
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

  global_step = tf.Variable(0, trainable=False)

  learning_rate = tf.train.polynomial_decay(
      LEARNING_RATE,
      global_step,
      TRAIN_ITERATIONS, (LEARNING_RATE / TRAIN_ITERATIONS),
      power=1.0,
      cycle=False,
      name='learning_rate')

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  if (not use_global_objectives) or USE_GO_SADDLE_POINT_OPT:
    training_op = optimizer.minimize(loss, global_step=global_step)
  else:
    lambdas = other_outputs['lambdas']
    primal_update_op = optimizer.minimize(loss, var_list=[w, b])
    dual_update_op = optimizer.minimize(
        loss, global_step=global_step, var_list=[lambdas])

  # Training loop:
  with tf.Session() as sess:
    checkpoint_step = TRAIN_ITERATIONS // NUM_CHECKPOINTS
    sess.run(tf.global_variables_initializer())
    step = sess.run(global_step)

    while step <= TRAIN_ITERATIONS:
      if (not use_global_objectives) or USE_GO_SADDLE_POINT_OPT:
        _, step, loss_value, w_value, b_value = sess.run(
            [training_op, global_step, loss, w, b])
      else:
        _, w_value, b_value = sess.run([primal_update_op, w, b])
        _, loss_value, step = sess.run([dual_update_op, loss, global_step])

      if use_global_objectives:
        go_outputs = sess.run(other_outputs.values())

      if step % checkpoint_step == 0:
        precision = precision_at_recall(
            np.dot(data['train_data'], w_value) + b_value,
            data['train_labels'], TARGET_RECALL)

        tf.logging.info('Loss = %f Precision = %f', loss_value, precision)
        if use_global_objectives:
          for i, output_name in enumerate(other_outputs.keys()):
            tf.logging.info('\t%s = %f', output_name, go_outputs[i])

    w_value, b_value = sess.run([w, b])
    return precision_at_recall(np.dot(data['eval_data'], w_value) + b_value,
                               data['eval_labels'],
                               TARGET_RECALL)


def main(unused_argv):
  del unused_argv
  experiment_data = create_training_and_eval_data_for_experiment(
      **EXPERIMENT_DATA_CONFIG)
  global_objectives_loss_precision = train_model(experiment_data, True)
  tf.logging.info('global_objectives precision at requested recall is %f',
                  global_objectives_loss_precision)
  cross_entropy_loss_precision = train_model(experiment_data, False)
  tf.logging.info('cross_entropy precision at requested recall is %f',
                  cross_entropy_loss_precision)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
