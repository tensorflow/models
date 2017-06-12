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

""" Tests the accuracy of the neural net and random features at approximating
the kernel induced by the skeleton """


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
import neural_net as NN
import kernel as kf

class Stat(object):
  def __init__(self):
    self.count = 0
    self.diff_sum = 0
    self.diff_sq_sum = 0
    self.diff_max = 0
    self.X_sum = 0
    self.Y_sum = 0
    self.X_sq_sum = 0
    self.Y_sq_sum = 0
    self.XY_sum = 0

  def AddToStat(self, mat1, mat2):
    diff = np.absolute(mat1 - mat2)
    self.count += mat1.size
    self.diff_sum += diff.sum()
    diff_sq = diff * diff
    self.diff_sq_sum += diff_sq.sum()
    self.diff_max = max(self.diff_max, diff.max())
    self.X_sum += mat1.sum()
    self.Y_sum += mat2.sum()
    self.X_sq_sum += (mat1 * mat1).sum()
    self.Y_sq_sum += (mat2 * mat2).sum()
    self.XY_sum += (mat1 * mat2).sum()

  def AddStat(self, other):
    for (attr, val) in self.__dict__.iteritems():
      setattr(self, attr, val + getattr(other, attr))

  def PrintStat(self, prefix):
    if self.count == 0:
      print "No data yet"
      return
    mean = self.diff_sum / self.count
    mean_sq = self.diff_sq_sum / self.count
    E_X = self.X_sum / self.count
    E_Y = self.Y_sum / self.count
    E_Xsq = self.X_sq_sum / self.count
    E_Ysq = self.Y_sq_sum / self.count
    E_XY = self.XY_sum / self.count

    corr = (E_XY - E_X * E_Y) / math.sqrt((E_Xsq - E_X * E_X) *
                                          (E_Ysq - E_Y * E_Y))
    print prefix, "Mean:", mean, "RMS:", math.sqrt(mean_sq), "Max:", self.diff_max, "Correlation:", corr


class ConcentrationExperiment(object):
  """ Create a graph for carrying out a concentration experiment """

  def __init__(self, config, get_inputs, logfile = None):
    self.config = config
    self.logfile = logfile
    self.get_inputs = get_inputs
    self.config.SetValue('number_of_classes', 0)  # No labels needed here.

    # Read a skeleton
    assert hasattr(config, 'skeleton_proto'), 'Skeleton proto must exist'
    self.skeleton = sk.Skeleton()
    self.skeleton.Load(self.config.skeleton_proto)

    self.original_replication = [None] * (len(self.skeleton.layers) - 1)
    for i in range(1, len(self.skeleton.layers)):
      self.original_replication[i-1] = self.skeleton.layers[i].replication


  def DoOneRun(self, run_id, rf_number, nn_replication, prefix='', seed=0,
               batch_count=1):
    batch_size = self.config.batch_size

    self.config.rf_number = rf_number
    self.config.rf_file_name = ('features_' + prefix + '_' + str(rf_number) +
                                '_' + str(run_id) + '.pkl')
    srf = rf.GenerateOrLoadRF(self.config, seed=run_id + 2718281828 + seed)

    if isinstance(nn_replication, (list, tuple)):
      self.skeleton.SetReplication(nn_replication)
    else:
      self.skeleton.SetReplication([int(x * nn_replication) for x in
                                    self.original_replication])
    with tf.Graph().as_default(), tf.Session('') as sess:
      examples = self.get_inputs(batch_size)

      # Calculate the exact gram matrix for the batch
      gram = tf.reshape(
          kf.Kernel(self.skeleton, examples, examples),
          [batch_size, batch_size])

      # Calculate the approximate gram matrix using a neural net
      rep, _ = NN.NeuralNet(self.skeleton, self.config, examples)
      srep = tf.squeeze(rep)
      approx_gram = tf.matmul(srep, tf.transpose(srep))

      # Normalize the approximate gram matrix to so that the norm of
      # each element is 1.
      norms = tf.reshape(tf.sqrt(tf.diag_part(approx_gram)), [-1, 1])
      nn_gram = tf.div(approx_gram,
                       tf.matmul(norms, tf.transpose(norms)))

      # Compute the approximate gram matrix using random features
      parameters = tf.constant(np.zeros(
          (rf_number, self.config.number_of_classes)).astype(np.float32))
      rand_features = tf.SparseTensor(srf.features[0], srf.features[1],
                                      srf.features[2])
      _, rf_vectors = rf.RandomFeaturesGraph(
          self.skeleton, self.config.number_of_classes, examples, rf_number,
          rand_features, parameters, srf.weights)
      rf_gram = tf.matmul(rf_vectors, rf_vectors, transpose_b=True)
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess, coord)
      RF_K_stat = Stat()
      NN_K_stat = Stat()
      for i in xrange(batch_count):
        gram_np, nn_gram_np, rf_gram_np, approx_gram_np = sess.run(
            [gram, nn_gram, rf_gram, approx_gram])
        RF_K_stat.AddToStat(gram_np, rf_gram_np)
        NN_K_stat.AddToStat(gram_np, nn_gram_np)
      coord.request_stop()
      coord.join(threads)
      return NN_K_stat, RF_K_stat


  def RF_Experiment(self, min_rf, rf_factor, rf_runs, run_count, batch_count):
    """ Run an experiment comparing concentration of RF.
    min_rf, rf_factor, rf_runs determine the number of random features to try.
    run_count is the number of runs for each number of random features --- the
    random features are regenerated for each run.
    batch_count is the number of batches per run, for the same set of RFs """

    for i in range(rf_runs):
      rf_number = min_rf * (rf_factor ** i)
      RF_K_stat = Stat()
      for j in range(run_count):
        _, stat = self.DoOneRun(j, rf_number, 1, prefix=str(i),
                                seed=i, batch_count=batch_count)
        RF_K_stat.AddStat(stat)
      RF_K_stat.PrintStat("Average diffs for " + str(rf_number) + " RFs")


  def NN_Experiment(self, rep_factor, rep_runs, run_count, batch_count):
    """ Run an experiment comparing concentration of Neural Nets.
    rep_factor, rep_runs determine the replications to try
    run_count is the number of runs for each replication --- the
    neural net is recreated for each run.
    batch_count is the number of batches per run, for the same NN """
    
    print "Comparing Neural Nets with Kernels"
    for i in range(rep_runs):
      rep = rep_factor ** i
      NN_K_stat = Stat()
      for j in range(run_count):
        # set 5 as the default number of RFs, as we will ignore this stat.
        stat, _ = self.DoOneRun(j, 5, rep, prefix=str(i),
                                seed=i, batch_count=batch_count)
        NN_K_stat.AddStat(stat)

      replication = [None] * (len(self.skeleton.layers) - 1)
      for i in range(1, len(self.skeleton.layers)):
        replication[i-1] = self.skeleton.layers[i].replication
      NN_K_stat.PrintStat("Average diffs for replication " + str(replication))
