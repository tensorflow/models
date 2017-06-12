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
import tempfile
import sys
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
import skeleton
import skeleton_pb2
import kernel

class TestSkeleton(test_util.TensorFlowTestCase):

  def testKernel(self):
    testfile_path = os.path.dirname(__file__)
    test_skeleton = skeleton.Skeleton()
    test_skeleton.Load(os.path.join(testfile_path, "simple_skeleton.pb.txt"))
    s1 = np.zeros((2,4,4,2), dtype=np.float32)
    s1[0,:,:,0] = [[1.0,0,1,1],[1,0,1,0],[1,1,1,0],[1,1,1,1]]
    s1[0,:,:,1] = [[0.0,1,0,0],[0,1,0,1],[0,0,0,1],[0,0,0,0]]
    s1[1,:,:,0] = np.ones((4,4))
    s2 = np.zeros((3,4,4,2), dtype=np.float32)
    s2[0,:,:,0] = [[1.0,0,1,1],[1,0,1,0],[1,1,1,0],[1,1,1,1]]
    s2[0,:,:,1] = [[0.0,1,0,0],[0,1,0,1],[0,0,0,1],[0,0,0,0]]
    s2[1,:,:,0] = np.ones((4,4))
    s2[2,:,:,0] = np.ones((4,4))
    with tf.Graph().as_default(), tf.Session('') as sess:
      s1_tf = tf.constant(s1, dtype=tf.float32)
      s2_tf = tf.constant(s2, dtype=tf.float32)
      ker = tf.reshape(kernel.Kernel(test_skeleton, s1_tf, s2_tf),
                       [s1.shape[0], s2.shape[0]])
    kernel_tf = sess.run([ker])[0]

    print kernel_tf
    self.assertAlmostEqual(0.80775118, kernel_tf[0,1])
    self.assertAlmostEqual(0.80775118, kernel_tf[0,2])
    self.assertAlmostEqual(0.80775118, kernel_tf[1,0])
    self.assertAlmostEqual(1.0, kernel_tf[0,0], 6)
    self.assertAlmostEqual(1.0, kernel_tf[1,1], 6)
    self.assertAlmostEqual(1.0, kernel_tf[1,2], 6)

if __name__ == '__main__':
  tf.test.main()
