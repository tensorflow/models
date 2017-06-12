# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

""" Learning Model for finding weights of random feature vectors. """

import cPickle
import math
import os.path
import subprocess
import sys
from datetime import datetime
import time

import tensorflow as tf
import numpy as np
import random

import utils
import skeleton as sk
import random_features as rf

def Loss(predictions, labels, number_of_classes, train_params, l2_reg_param):
  l2_norm = tf.nn.l2_loss(train_params)
  if number_of_classes == 1:
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        predictions, labels)) + l2_reg_param * l2_norm
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=tf.squeeze(predictions),
          labels=labels)) / math.log(number_of_classes) + l2_reg_param * l2_norm

class RandomFeaturesModel(object):
  """ Create a graph for training and testing a random features model. """
  def __init__(self,
               config,
               tf_examples=None,
               tf_labels=None,
               tf_test_examples=None,
               tf_test_labels=None,
               init_params=None,
               logfile = None):

    # Check that the number of class was set
    assert config.number_of_classes > 0, \
      'Number of classes was not set'

    self.config = config
    self.logfile = logfile

    # Read a skeleton and create its corresponding kernel
    assert hasattr(config, 'skeleton_proto'), 'Skeleton proto must exist'
    skeleton = sk.Skeleton()
    skeleton.Load(config.skeleton_proto)
    skeleton.SetActivationCoeffs(config.activation_coeffs)
    if config.remove_dual_bias:
      skeleton.RemoveDualBias()

    # allocate the parameter vectors
    random_features = rf.GenerateOrLoadRF(config).features
    kernel_width = random_features[2][0]
    if init_params == None:
      init_params = np.zeros(
          (kernel_width, config.number_of_classes)).astype(np.float32)
    self.tf_rf_params = tf.Variable(init_params, trainable=True)

    clip = config.clip_value
    if clip > 0:
      self.tf_rf_params = tf.clip_by_value(self.tf_rf_params, -clip, clip)

    # Allocate TF <sparse> matrix for the random features
    self.tf_rf_vectors = tf.SparseTensor(random_features[0],
                                         random_features[1],
                                         random_features[2])

    # Training iteration variable
    self.global_step = tf.Variable(0, trainable=False, name='step')

    if tf_examples != None:
      self.SetupTraining(tf_examples, tf_labels, skeleton, kernel_width)
    if tf_test_examples != None:
      self.SetupTesting(tf_test_examples, tf_test_labels, skeleton,
                        kernel_width)

    # Op for initializing parameters.
    self.init = tf.global_variables_initializer()

    # And a saver...
    self.saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out
    self.merged = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(config.model_file_path)


  def SetupTraining(self, tf_examples, tf_labels, skeleton, kernel_width):
    # Generate prediction graph from random features set
    self.predictions, _ = rf.RandomFeaturesGraph(
        skeleton, self.config.number_of_classes, \
        tf_examples, kernel_width, self.tf_rf_vectors, self.tf_rf_params)

    decay_steps = int(float(self.config.number_of_examples) /
                      float(self.config.batch_size) *
                      self.config.epochs_per_decay)

    # Learning rate setting
    lr = tf.train.exponential_decay(
        self.config.learning_rate,
        self.global_step,
        decay_steps=decay_steps,
        decay_rate=self.config.decay_rate,
        staircase=True,
        name='learning_rate')

    self.loss = Loss(self.predictions, tf_labels,
                     self.config.number_of_classes,
                     self.tf_rf_params, self.config.l2_reg_param)

    if self.config.number_of_classes == 1:
      self.accuracy = utils.Accuracy_binary(self.predictions, tf_labels)
    else:
      self.accuracy = utils.Accuracy(self.predictions, tf_labels)

    grads = tf.gradients(self.loss, tf.trainable_variables())

    optimizer = utils.GetOptimizer(self.config, lr)

    self.update = optimizer.apply_gradients(
        zip(grads, tf.trainable_variables()), global_step=self.global_step)

    # make summary writers for tensorboard
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    tf.summary.scalar('learning_rate', lr)


  def SetupTesting(self, tf_test_examples, tf_test_labels, skeleton,
                   kernel_width):
    self.test_predictions, _ = rf.RandomFeaturesGraph(
        skeleton, self.config.number_of_classes,
        tf_test_examples, kernel_width, self.tf_rf_vectors, self.tf_rf_params)

    self.test_loss = Loss(self.test_predictions, tf_test_labels,
                          self.config.number_of_classes,
                          self.tf_rf_params, self.config.l2_reg_param)

    if self.config.number_of_classes == 1:
      self.test_accuracy = utils.Accuracy_binary(self.test_predictions,
                                                 tf_test_labels)
    else:
      self.test_accuracy = utils.Accuracy(self.test_predictions, tf_test_labels)

    # make summary writers for tensorboard
    tf.summary.scalar('test_loss', self.test_loss)
    tf.summary.scalar('test_accuracy', self.test_accuracy)


  def Init(self, sess):
    """ Initialize the model.  Should be done before Train,
    but supervisor does it too. """
    self.init.run(session=sess)
    if len(self.config.model_file_path) > 0:
      if len(self.config.init_file) > 0:
        init_file = os.path.join(self.config.model_file_path,
                                 self.config.init_file)
        self.saver.restore(sess, init_file)


  def Train(self, sess):
    """ Simple epoch-based training. """
    assert self.config.number_of_examples > 0
    assert self.config.number_of_classes > 0

    number_of_batches = \
      self.config.number_of_examples / self.config.batch_size

    if number_of_batches * self.config.batch_size < \
      self.config.number_of_examples:
      print 'Warning: #examples is not divisible by batch-size (truncating)'

    test_frequency = self.config.model_test_frequency

    max_steps = self.config.number_of_epochs * number_of_batches
    steps = sess.run(self.global_step)
    eloss = eaccu = 0.0
    current_steps = 0
    estr = 'Epoch: {0:3d} ({1:3d} batches)  Loss: {2:9.6f}  Accuracy: {3:7.4f}  Time: {4:7.4f}'
    start_time = time.time()
    epoch = 0
    while steps < max_steps:
      _, loss, accuracy, steps = sess.run([
          self.update, self.loss, self.accuracy, self.global_step])
      eloss += loss
      eaccu += accuracy
      current_steps += 1
      epoch = steps / number_of_batches
      if current_steps % 100 == 0:
        utils.PrintAndLog(self.logfile, estr.format(
            epoch, current_steps, eloss / current_steps,
            eaccu / current_steps, time.time() - start_time))
      if steps % number_of_batches == 0:
        epoch = steps / number_of_batches
        utils.PrintAndLog(self.logfile, estr.format(
            epoch, current_steps, eloss / current_steps,
            eaccu / current_steps, time.time() - start_time))
        if test_frequency > 0 and epoch % test_frequency == 0:
          self.TestEval(sess)
        eloss = eaccu = 0.0
        current_steps = 0
        start_time = time.time()

    return sess.run(self.tf_rf_params)

  def TrainEval(self, sess):
    """ Eval loss and accuracy on entire train set. """
    utils.PrintAndLog(self.logfile, "Final training eval")
    assert self.config.number_of_examples > 0
    assert self.config.number_of_classes > 0

    number_of_batches = \
      self.config.number_of_examples / self.config.batch_size

    if number_of_batches * self.config.batch_size < \
      self.config.number_of_examples:
      print 'Warning: #examples is not divisible by batch-size (truncating)'

    eloss = eaccu = 0.0
    estr = '(Train) Batches:{0:5d}    Loss: {1:7.4f}   Accuracy: {2:7.4f}'
    for batch in range(number_of_batches):
      accuracy, loss = sess.run([self.accuracy, self.loss])
      eloss += loss
      eaccu += accuracy
      if (1 + batch) % 100 == 0:
        utils.PrintAndLog(self.logfile, estr.format(1 + batch,
                                                    eloss/(1.0 + batch),
                                                    eaccu/(1.0 + batch)))
    eloss /= number_of_batches
    eaccu /= number_of_batches
    utils.PrintAndLog(self.logfile,
                      estr.format(number_of_batches, eloss, eaccu))

    return eaccu

  def TestEval(self, sess):
    """ Eval loss and accuracy on test / validation set. """
    utils.PrintAndLog(self.logfile, "Testing now")
    assert self.config.number_of_test_examples > 0
    assert self.config.number_of_classes > 0

    number_of_batches = \
      self.config.number_of_test_examples / \
      self.config.batch_size

    if number_of_batches * self.config.batch_size < \
      self.config.number_of_test_examples:
      print 'Warning: #examples is not divisible by batch-size (truncating)'

    eloss = eaccu = 0.0
    estr = '(Test) Batches:{0:5d}    Loss: {1:7.4f}   Accuracy: {2:7.4f}'
    for batch in range(number_of_batches):
      accuracy, loss = sess.run([self.test_accuracy, self.test_loss])
      eloss += loss
      eaccu += accuracy
      if (1 + batch) % 100 == 0:
        utils.PrintAndLog(self.logfile, estr.format(1 + batch,
                                                    eloss/(1.0 + batch),
                                                    eaccu/(1.0 + batch)))
    eloss /= number_of_batches
    eaccu /= number_of_batches
    utils.PrintAndLog(self.logfile,
                      estr.format(number_of_batches, eloss, eaccu))

    return eaccu
