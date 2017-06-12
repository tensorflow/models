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

"""Neural network trainer.
"""
import math
import os.path
import time

import numpy as np
import tensorflow as tf

import neural_net as NN
import skeleton as sk
import utils


class NeuralNetModel(object):
  """Neural network trainer class."""

  def __init__(self,
               learning_params,
               examples,
               labels,
               test_examples,
               test_labels):
    self.learning_params = learning_params
    skeleton = sk.Skeleton()
    skeleton.Load(self.learning_params.skeleton_proto)
    self.SetupTraining(examples, labels, skeleton)
    self.SetupTesting(test_examples, test_labels, skeleton)
    self.init = tf.variables_initializer(tf.global_variables())
    self.saver = tf.train.Saver(tf.global_variables())

  def SetupLoss(self, predictions, labels):
    """Compute the loss and accuracy of the predictions given the labels."""
    loss_vec = tf.nn.softmax_cross_entropy_with_logits(
        logits=tf.squeeze(predictions), labels=labels)
    loss = (tf.reduce_mean(loss_vec) /
            math.log(self.learning_params.number_of_classes))
    accuracy = utils.Accuracy(predictions, labels)
    return loss, accuracy

  def SetupTraining(self, examples, labels, skeleton):
    """Set tensorflow graph for training."""
    # Builds the graph.
    predictions, self.trainable = NN.NeuralNet(skeleton, self.learning_params,
                                               examples)

    self.loss, self.accuracy = self.SetupLoss(predictions, labels)

    # Set trained variables.
    var_list = self.trainable
    if self.learning_params.trained_layers:
      var_list = []
      for i in str.split(self.learning_params.trained_layers, ','):
        var_list.append(self.trainable[2*int(i)])
        var_list.append(self.trainable[2*int(i)+1])

    # Set the optimizer.
    self.global_step = tf.Variable(0, trainable=False)
    num_batches_per_epoch = (self.learning_params.number_of_examples /
                             self.learning_params.batch_size)
    decay_steps = int(num_batches_per_epoch *
                      self.learning_params.epochs_per_decay)
    self.lr = tf.train.exponential_decay(
        self.learning_params.learning_rate,
        self.global_step,
        decay_steps,
        self.learning_params.decay_rate,
        staircase=True)
    optimizer = utils.GetOptimizer(self.learning_params, self.lr)
    self.update = optimizer.minimize(
        self.loss,
        var_list=var_list,
        gate_gradients=True,
        global_step=self.global_step)

    # make summary writers for tensorboard
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)

  def SetupTesting(self, test_examples, test_labels, skeleton):
    """Set tensorflow graph for training."""
    # Builds the graph.
    predictions, _ = NN.NeuralNet(skeleton, self.learning_params, test_examples,
                                  tf_params=self.trainable)
    self.test_loss, self.test_accuracy = self.SetupLoss(predictions,
                                                        test_labels)
    # make summary writers for tensorboard
    tf.summary.scalar('test_loss', self.test_loss)
    tf.summary.scalar('test_accuracy', self.test_accuracy)

  def Init(self, sess):
    """Initialize model. Must be done before Train. Supervisor does it too."""

    self.init.run(session=sess)
    if self.learning_params.model_file_path:
      if self.learning_params.init_file:
        init_file = os.path.join(self.learning_params.model_file_path,
                                 self.learning_params.init_file)
        self.saver.restore(sess, init_file)
      if self.learning_params.model_file_name:
        self.model_file = os.path.join(self.learning_params.model_file_path,
                                       self.learning_params.model_file_name)

  def Train(self, sess):
    """Train the network."""
    number_of_batches_per_epoch = (self.learning_params.number_of_examples /
                                   self.learning_params.batch_size)
    test_frequency = self.learning_params.model_test_frequency
    save_frequency = self.learning_params.model_save_frequency
    epoch = 0
    steps = 0
    total_steps = (self.learning_params.number_of_epochs *
                   number_of_batches_per_epoch)
    epoch_loss = epoch_accu = 0.0
    epoch_str = ('\nEpoch: {0:3d} ({1:3d} batches)  Loss: {2:7.4f}'
                 ' Accuracy: {3:7.4f} Time: {4:7.4f}')
    stat_str = ('Layer {0:d} statistics:  Shape: {1:20}'
                ' Spectral Norm: {2:7.2f} {3:7.2f} (absolute, relative)'
                ' Squared Frobenius Norm: {4:7.2f} {5:7.2f}'
                ' (absolute, relative) Trace Norm: {6:7.2f} {7:7.2f}'
                ' (absolute, relative)')
    start_time = time.time()
    initial_trainable = []
    for i in range(0, len(self.trainable), 2):
      cur_layer = sess.run(self.trainable[i])
      cur_shape = cur_layer.shape
      initial_trainable.append(
          cur_layer.reshape(cur_shape[0]*cur_shape[1]*cur_shape[2],
                            cur_shape[3]))

    while steps < total_steps:
      _, loss, accuracy, _ = sess.run([self.update, self.loss,
                                       self.accuracy, self.global_step])
      epoch_loss += loss
      epoch_accu += accuracy
      steps += 1
      if steps % number_of_batches_per_epoch == 0:
        epoch = steps / number_of_batches_per_epoch
        utils.PrintAndLog(self.learning_params.log_file,
                          epoch_str.format(
                              epoch, number_of_batches_per_epoch,
                              epoch_loss / number_of_batches_per_epoch,
                              epoch_accu / number_of_batches_per_epoch,
                              time.time() - start_time))
        epoch_loss = epoch_accu = 0.0
        for i in range(0, len(self.trainable), 2):
          cur_layer = sess.run(self.trainable[i])
          cur_shape = cur_layer.shape
          cur_layer_reshaped = cur_layer.reshape(
              cur_shape[0]*cur_shape[1]*cur_shape[2], cur_shape[3])
          s_abs = np.linalg.svd(cur_layer_reshaped, full_matrices=False,
                                compute_uv=False)
          s_rel = np.linalg.svd(cur_layer_reshaped - initial_trainable[i/2],
                                full_matrices=False,
                                compute_uv=False)
          utils.PrintAndLog(self.learning_params.log_file,
                            stat_str.format(i/2,
                                            cur_shape,
                                            s_abs.max(),
                                            s_rel.max(),
                                            (s_abs*s_abs).sum(),
                                            (s_rel*s_rel).sum(),
                                            s_abs.sum(),
                                            s_rel.sum()))
        if test_frequency > 0 and epoch % test_frequency == 0:
          self.TestEval(sess)
        if (save_frequency > 0 and self.learning_params.model_file_name
            and epoch % save_frequency == 0):
          self.saver.save(sess, self.model_file)
        start_time = time.time()

  def TestEval(self, sess):
    """Eval loss and accuracy on test / validation set."""
    number_of_batches_per_epoch = (self.learning_params.number_of_test_examples
                                   / self.learning_params.batch_size)
    eloss = eaccu = 0.0
    for _ in range(number_of_batches_per_epoch):
      accuracy, loss = sess.run([self.test_accuracy, self.test_loss])
      eloss += loss
      eaccu += accuracy

    eloss /= number_of_batches_per_epoch
    eaccu /= number_of_batches_per_epoch
    tstr = '(Test)      Loss: {0:7.4f}   Accuracy: {1:7.4f}'
    utils.PrintAndLog(self.learning_params.log_file, tstr.format(eloss, eaccu))
