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


"""

Can NOT be differentiated through.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.python.framework import function

from learning_unsupervised_learning import utils

from learning_unsupervised_learning.meta_objective import utils as meta_obj_utils

from sklearn import svm
from sklearn import linear_model


def build_fit(device, model_fn, num_classes, probs=True):

  def _py_fit_predict(trX, trY, teX):
    assert len(np.unique(trY)) == num_classes
    model = model_fn()
    model.fit(trX, trY)
    trP = model.predict(trX)
    teP = model.predict(teX)
    if probs:
      teP_probs = model.predict_log_proba(teX)
      return trP.astype(np.int64), teP.astype(np.int64), teP_probs.astype(
          np.float32)
    else:
      teP = model.predict(teX)
      return trP.astype(np.int64), teP.astype(np.int64)

  def return_fn(trX, trY, teX):
    with tf.device(device):
      with tf.device("/cpu:0"):
        if probs:
          return tf.py_func(
              _py_fit_predict,
              [tf.identity(trX),
               tf.identity(trY),
               tf.identity(teX)], [tf.int64, tf.int64, tf.float32])
        else:
          return tf.py_func(
              _py_fit_predict,
              [tf.identity(trX),
               tf.identity(trY),
               tf.identity(teX)], [tf.int64, tf.int64])

  return return_fn


class SKLearn(meta_obj_utils.MultiTrialMetaObjective):

  def __init__(
      self,
      local_device=None,
      remote_device=None,
      averages=1,
      samples_per_class=10,
      probs=False,
      stddev=0.01,
      n_samples=10,
      name="SKLearn",
  ):
    self._local_device = local_device
    self._remote_device = remote_device
    self.name = name
    self.probs = probs
    self.n_samples = n_samples
    self.stddev = stddev

    super(SKLearn, self).__init__(
        name=name, samples_per_class=samples_per_class, averages=averages)

  def _get_model(self):
    raise NotImplemented()

  def _build_once(self, dataset, feature_transformer):
    with tf.device(self._local_device):
      tr_batch = dataset()
      te_batch = dataset()
      num_classes = tr_batch.label_onehot.shape.as_list()[1]
      all_batch = utils.structure_map_multi(lambda x: tf.concat(x, 0),
                                            [tr_batch, te_batch])
      features = feature_transformer(all_batch)
      trX, teX = utils.structure_map_split(lambda x: tf.split(x, 2, axis=0),
                                           features)
      trY = tf.to_int64(tr_batch.label)
      trY_onehot = tf.to_int32(tr_batch.label_onehot)
      teY = tf.to_int64(te_batch.label)
      teY_shape = teY.shape.as_list()

      def blackbox((trX, trY, teX, teY)):
        trY = tf.to_int32(tf.rint(trY))
        teY = tf.to_int32(tf.rint(teY))
        tf_fn = build_fit(
            self._local_device,
            self._get_model,
            num_classes=num_classes,
            probs=self.probs)
        if self.probs:
          trP, teP, teP_probs = tf_fn(trX, trY, teX)
        else:
          trP, teP = tf_fn(trX, trY, teX)

        teY.set_shape(teY_shape)
        if self.probs:
          onehot = tf.one_hot(teY, num_classes)
          crossent = -tf.reduce_sum(onehot * teP_probs, [1])
          return tf.reduce_mean(crossent)
        else:
          # use error rate as the loss if no surrogate is avalible.
          return 1 - tf.reduce_mean(
              tf.to_float(tf.equal(teY, tf.to_int32(teP))))

      test_loss = blackbox((trX, tf.to_float(trY), teX, tf.to_float(teY)))

      stats = {}

      tf_fn = build_fit(
          self._local_device,
          self._get_model,
          num_classes=num_classes,
          probs=self.probs)
      if self.probs:
        trP, teP, teP_probs = tf_fn(trX, trY, teX)
      else:
        trP, teP = tf_fn(trX, trY, teX)
      stats["%s/accuracy_train" % self.name] = tf.reduce_mean(
          tf.to_float(tf.equal(tf.to_int32(trY), tf.to_int32(trP))))
      stats["%s/accuracy_test" % self.name] = tf.reduce_mean(
          tf.to_float(tf.equal(tf.to_int32(teY), tf.to_int32(teP))))
      stats["%s/test_loss" % self.name] = test_loss
      return test_loss, stats


class LogisticRegression(SKLearn):

  def __init__(self, C=1.0, name="LogisticRegression", probs=True, **kwargs):
    self.C = C
    super(LogisticRegression, self).__init__(name=name, probs=probs, **kwargs)

  def _get_model(self):
    return linear_model.LogisticRegression(C=self.C)
