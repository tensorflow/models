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
"""Tests for datum_io, the python interface of DatumProto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from delf import datum_io


class DatumIoTest(tf.test.TestCase):

  def Conversion2dTestWithType(self, dtype):
    original_data = np.arange(9).reshape(3, 3).astype(dtype)
    serialized = datum_io.SerializeToString(original_data)
    retrieved_data = datum_io.ParseFromString(serialized)
    self.assertTrue(np.array_equal(original_data, retrieved_data))

  def Conversion3dTestWithType(self, dtype):
    original_data = np.arange(24).reshape(2, 3, 4).astype(dtype)
    serialized = datum_io.SerializeToString(original_data)
    retrieved_data = datum_io.ParseFromString(serialized)
    self.assertTrue(np.array_equal(original_data, retrieved_data))

  def testConversion2dWithType(self):
    self.Conversion2dTestWithType(np.int8)
    self.Conversion2dTestWithType(np.int16)
    self.Conversion2dTestWithType(np.int32)
    self.Conversion2dTestWithType(np.int64)
    self.Conversion2dTestWithType(np.float16)
    self.Conversion2dTestWithType(np.float32)
    self.Conversion2dTestWithType(np.float64)

  def testConversion3dWithType(self):
    self.Conversion3dTestWithType(np.int8)
    self.Conversion3dTestWithType(np.int16)
    self.Conversion3dTestWithType(np.int32)
    self.Conversion3dTestWithType(np.int64)
    self.Conversion3dTestWithType(np.float16)
    self.Conversion3dTestWithType(np.float32)
    self.Conversion3dTestWithType(np.float64)

  def testWriteAndReadToFile(self):
    data = np.array([[[-1.0, 125.0, -2.5], [14.5, 3.5, 0.0]],
                     [[20.0, 0.0, 30.0], [25.5, 36.0, 42.0]]])
    tmpdir = tf.test.get_temp_dir()
    filename = os.path.join(tmpdir, 'test.datum')
    datum_io.WriteToFile(data, filename)
    data_read = datum_io.ReadFromFile(filename)
    self.assertAllEqual(data_read, data)


if __name__ == '__main__':
  tf.test.main()
