# Copyright 2018 Google, Inc. All Rights Reserved.
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



"""Closed form linear regression.

Can be differentiated through.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from learning_unsupervised_learning import utils
from learning_unsupervised_learning import variable_replace


def solve_ridge(x, y, ridge_factor):
  with tf.name_scope("solve_ridge"):
    # Added a column of ones to the end of the feature matrix for bias
    A = tf.concat([x, tf.ones((x.shape.as_list()[0], 1))], axis=1)

    # Analytic solution for the ridge regression loss
    inv_target = tf.matmul(A, A, transpose_a=True)
    np_diag_penalty = ridge_factor * np.ones(
        A.shape.as_list()[1], dtype="float32")
    # Remove penalty on bias component of weights
    np_diag_penalty[-1] = 0.
    diag_penalty = tf.constant(np_diag_penalty)
    inv_target += tf.diag(diag_penalty)

    inv = tf.matrix_inverse(inv_target)
    w = tf.matmul(inv, tf.matmul(A, y, transpose_a=True))
    return w


class LinearRegressionMetaObjective(snt.AbstractModule):
  """A meta objective based on training Ridge Regression with analytic solution.

  This is used to evaluate the performance of a given feature set trained in
  some other manner.
  """

  def __init__(self,
               local_device=None,
               remote_device=None,
               zero_one_labels=True,
               normalize_y_hat=True,
               normalize_act=False,
               averages=1,
               ridge_factor=0.1,
               center_y=True,
               hinge_loss=False,
               samples_per_class=10,
               test_train_scalar=1.0,
              ):
    self._local_device = local_device
    self._remote_device = remote_device
    self.zero_one_labels = zero_one_labels
    self.normalize_y_hat = normalize_y_hat
    self.normalize_act = normalize_act
    self.ridge_factor = ridge_factor
    self.averages = averages
    self.samples_per_class = samples_per_class
    self.center_y=center_y
    self.test_train_scalar=test_train_scalar
    self.hinge_loss = hinge_loss

    self.dataset_map = {}

    super(LinearRegressionMetaObjective,
          self).__init__(name="LinearRegressionMetaObjective")

  def _build(self, dataset, feature_transformer):
    if self.samples_per_class is not None:
      if dataset not in self.dataset_map:
        # datasets are outside of frames from while loops
        with tf.control_dependencies(None):
          self.dataset_map[dataset] = utils.sample_n_per_class(
              dataset, self.samples_per_class)

      dataset = self.dataset_map[dataset]

    stats = collections.defaultdict(list)
    losses = []
    # TODO(lmetz) move this to ingraph control flow?
    for _ in xrange(self.averages):
      loss, stat = self._build_once(dataset, feature_transformer)
      losses.append(loss)
      for k, v in stat.items():
        stats[k].append(v)
    stats = {k: tf.add_n(v) / float(len(v)) for k, v in stats.items()}

    summary_updates = []
    for k, v in stats.items():
      tf.summary.scalar(k, v)

    with tf.control_dependencies(summary_updates):
      return tf.add_n(losses) / float(len(losses))

  def _build_once(self, dataset, feature_transformer):
    with tf.device(self._local_device):
      batch = dataset()
      num_classes = batch.label_onehot.shape.as_list()[1]

      regression_mod = snt.Linear(num_classes)

      if self.normalize_act:

        def normalize_transformer(x):
          unnorm_x = feature_transformer(x)
          return tf.nn.l2_normalize(unnorm_x, 0)

        feature_transformer_wrap = normalize_transformer
      else:
        feature_transformer_wrap = feature_transformer

      # construct the variables of the right shape in the sonnet module by
      # calling a forward pass through the regressor.
      with utils.assert_no_new_variables():
        dummy_features = feature_transformer_wrap(batch)
      regression_mod(dummy_features)
      reg_w = regression_mod.w
      reg_b = regression_mod.b

      batch_test = dataset()
      all_batch = utils.structure_map_multi(lambda x: tf.concat(x, 0), [batch, batch_test])
      #all_batch = tf.concat([batch, batch_test], 0)
      # Grab a new batch of data from the dataset.
      features = feature_transformer_wrap(all_batch)
      features, features_test = utils.structure_map_split(lambda x: tf.split(x, 2, axis=0), features)

      def center_y(y):
        y -= tf.reduce_mean(y)
        y *= tf.rsqrt(tf.reduce_mean(tf.reduce_sum(y**2, axis=[1], keep_dims=True)))
        return y
      def get_y_vec(batch):
        y_pieces = []
        if hasattr(batch, "label_onehot"):
          if self.zero_one_labels:
            y_pieces += [batch.label_onehot]
          else:
            y_pieces += [2. * batch.label_onehot - 1.]
        if hasattr(batch, "regression_target"):
          y_pieces += [batch.regression_target]
        y = tf.concat(y_pieces, 1)
        if self.center_y:
          y = center_y(y)
        return y

      y_train = get_y_vec(batch)

      w = solve_ridge(features, y_train, self.ridge_factor)

      # Generate features from another batch to evaluate loss on the validation
      # set. This provide a less overfit signal to the learned optimizer.
      y_test = get_y_vec(batch_test)

      def compute_logit(features):
        # We have updated the classifier mod in previous steps, we need to
        # substitute out those variables to get new values.
        replacement = collections.OrderedDict([(reg_w, w[:-1]), (reg_b, w[-1])])
        with variable_replace.variable_replace(replacement):
          logits = regression_mod(features)

        return logits

      batch_size = y_train.shape.as_list()[0]

      logit_train = compute_logit(features)
      logit_test_unnorm = compute_logit(features_test)
      if self.normalize_y_hat:
        logit_test = logit_test_unnorm / tf.sqrt(
            tf.reduce_sum(logit_test_unnorm**2, axis=[1], keep_dims=True))
      else:
        logit_test = logit_test_unnorm

      stats = {}

      if self.hinge_loss:
        # slightly closer to the true classification loss
        # any distance smaller than 1 is guaranteed to map to the correct class
        mse_test = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.square(logit_test - y_test), axis=1)-1.)) / batch_size
      else:
        mse_test = tf.reduce_sum(tf.square(logit_test - y_test)) / batch_size

      stats["mse_test"] = mse_test

      mse_train = tf.reduce_sum(tf.square(logit_train - y_train)) / batch_size
      stats["mse_train"] = mse_train

      is_correct_test = tf.equal(tf.argmax(logit_test, 1), tf.argmax(y_test, 1))
      accuracy_test = tf.reduce_mean(tf.cast(is_correct_test, tf.float32))
      stats["accuracy_test"] = accuracy_test

      def test_confusion_fn():
        test_confusion = tf.confusion_matrix(tf.argmax(y_test, 1), tf.argmax(logit_test, 1))
        test_confusion = tf.to_float(test_confusion) / tf.constant((logit_test.shape.as_list()[0] / float(logit_test.shape.as_list()[1])), dtype=tf.float32)
        test_confusion = tf.expand_dims(tf.expand_dims(test_confusion, 0), 3)
        return test_confusion
      tf.summary.image("test_confusion", test_confusion_fn())

      def train_confusion_fn():
        train_confusion = tf.confusion_matrix(tf.argmax(y_train, 1), tf.argmax(logit_train, 1))
        train_confusion = tf.to_float(train_confusion) / tf.constant((logit_train.shape.as_list()[0] / float(logit_train.shape.as_list()[1])), dtype=tf.float32)
        train_confusion = tf.expand_dims(tf.expand_dims(train_confusion, 0), 3)
        return train_confusion
      tf.summary.image("train_confusion", train_confusion_fn())

      is_correct = tf.equal(tf.argmax(logit_train, 1), tf.argmax(y_train, 1))
      accuracy_train = tf.reduce_mean(tf.cast(is_correct, tf.float32))
      stats["accuracy_train"] = accuracy_train

      reg = self.ridge_factor * tf.reduce_sum(tf.square(w[:-1])) / batch_size
      stats["ridge_component"] = reg

      stats["total_loss"] = mse_test + reg

      loss_to_train_at = (reg+ mse_test) * self.test_train_scalar + (mse_train + reg)*(1 - self.test_train_scalar)

      loss_to_train_at = tf.identity(loss_to_train_at)

      # Minimizing the test loss should not require regurization because the
      # metaobjective is solved for the training loss
      return loss_to_train_at, stats

  def local_variables(self):
    """List of variables that need to be updated for each evaluation.

    These variables should not be stored on a parameter server and
    should be reset every computation of a meta_objective loss.

    Returns:
      vars: list of tf.Variable
    """
    return list(
        snt.get_variables_in_module(self, tf.GraphKeys.TRAINABLE_VARIABLES))

  def remote_variables(self):
    return []
