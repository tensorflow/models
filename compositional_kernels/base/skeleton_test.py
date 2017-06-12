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

class TestSkeleton(test_util.TensorFlowTestCase):

  def setUp(self):
    self.tempfiles = []

  def MakeTempfile(self):
    """Returns the name of a new temporary file."""
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    self.tempfiles.append(filename)
    return filename

  def tearDown(self):
    for filename in self.tempfiles:
      try:
        os.remove(filename)
      except OSError:
        pass

  def testLoad(self):
    testfile_path = os.path.dirname(__file__)
    test_skeleton = skeleton.Skeleton()
    test_skeleton.Load(os.path.join(testfile_path, "test_skeleton.txt"))
    print "Test skeleton loaded successfully"
    self.assertEqual(3, len(test_skeleton.layers))
    layer = test_skeleton.layers[0]
    self.assertEqual(2, layer.dimensions.size)
    self.assertEqual(24, layer.size)
    layer = test_skeleton.layers[1]
    self.assertEqual(12, layer.size)
    layer = test_skeleton.layers[2]
    self.assertEqual(2, layer.dimensions.size)
    self.assertEqual(1, layer.size)

  def testReplication(self):
    testfile_path = os.path.dirname(__file__)
    test_skeleton = skeleton.Skeleton()
    test_skeleton.Load(os.path.join(testfile_path, "test_skeleton.txt"))
    test_skeleton.SetReplication([100, 200])
    self.assertEqual(test_skeleton.layers[1].replication, 100)
    self.assertEqual(test_skeleton.layers[2].replication, 200)


if __name__ == '__main__':
  tf.test.main()
