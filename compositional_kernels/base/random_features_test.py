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

import os
import random
import stat
import sys
import tensorflow as tf

from tensorflow.python.framework import test_util
from google.protobuf import text_format
from google.protobuf import descriptor_pb2

import config
import skeleton
import skeleton_pb2
import random_features

import numpy as np
import scipy.stats as stat

def Convolve(q, p):
  r = np.zeros(q.size + p.size)
  for i in range(q.size):
    r[i:i + p.size] += q[i] * p
  return r[0:q.size]

def EqArray(a, b):
  return np.absolute(a - b).max() < 1e-8

def EstimateDistribution(skeleton, config):
  """ Estimate the distribution of the number of non-zero elements in the
  deep random features generated from the skeleton. """
  res = np.zeros((1, config.activation_coeffs * len(skeleton.layers)))
  prob_mat = np.zeros((res.size, res.size))
  res[0, 1] = 1
  layer_id = len(skeleton.layers) - 1
  while layer_id > 0:
    if (layer_id == len(skeleton.layers) - 1 or
        not EqArray(skeleton.layers[layer_id].activation.dual_coeffs,
                    dual_coeffs)):
      dual_coeffs = skeleton.layers[layer_id].activation.dual_coeffs
      prob_mat[0, 0] = 1.0
      for n in range(1, res.size):
        prob_mat[n] = Convolve(prob_mat[n-1], dual_coeffs)
    res = np.dot(res, prob_mat)
    layer_id -= 1
  return res


class TestRandomFeatures(test_util.TensorFlowTestCase):

  def testComputeRandomFeature(self):
    test_config = config.LearningParams()
    testfile_path = os.path.dirname(__file__)
    testfile = "test_skeleton.txt"
    testfile_path = os.path.join(testfile_path, testfile)
    test_skeleton = skeleton.Skeleton()
    test_skeleton.Load(testfile_path)
    test_skeleton.SetActivationCoeffs(test_config.activation_coeffs)
    random.seed(57721)
    np.random.seed(seed=57721)

    featureCount = 1000
    counts = [0] * (test_config.activation_coeffs * len(test_skeleton.layers))
    for i in range(featureCount):
      v = random_features.ComputeRandomFeature(test_skeleton, test_config)
      n = np.count_nonzero(v)
      counts[n] += 1
    print counts
    self.assertEqual([513, 294, 128, 34, 17, 5, 4, 4, 1, 0],
                     counts[0:10])

    guess = EstimateDistribution(test_skeleton, test_config) * featureCount
    categories = (guess > 5).sum()
    chi, p = stat.chisquare(np.array(counts[0:categories]),
                            guess[0,0:categories])
    print chi, p
    self.assertGreater(p, 0.5)


if __name__ == '__main__':
  tf.test.main()
