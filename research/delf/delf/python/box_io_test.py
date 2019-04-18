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
"""Tests for box_io, the python interface of Boxes proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from delf import box_io


class BoxesIoTest(tf.test.TestCase):

  def _create_data(self):
    """Creates data to be used in tests.

    Returns:
      boxes: [N, 4] float array denoting bounding box coordinates, in format
      [top,
        left, bottom, right].
      scores: [N] float array with detection scores.
      class_indices: [N] int array with class indices.
    """
    boxes = np.arange(24, dtype=np.float32).reshape(6, 4)
    scores = np.arange(6, dtype=np.float32)
    class_indices = np.arange(6, dtype=np.int32)

    return boxes, scores, class_indices

  def testConversionAndBack(self):
    boxes, scores, class_indices = self._create_data()

    serialized = box_io.SerializeToString(boxes, scores, class_indices)
    parsed_data = box_io.ParseFromString(serialized)

    self.assertAllEqual(boxes, parsed_data[0])
    self.assertAllEqual(scores, parsed_data[1])
    self.assertAllEqual(class_indices, parsed_data[2])

  def testWriteAndReadToFile(self):
    boxes, scores, class_indices = self._create_data()

    tmpdir = tf.test.get_temp_dir()
    filename = os.path.join(tmpdir, 'test.boxes')
    box_io.WriteToFile(filename, boxes, scores, class_indices)
    data_read = box_io.ReadFromFile(filename)

    self.assertAllEqual(boxes, data_read[0])
    self.assertAllEqual(scores, data_read[1])
    self.assertAllEqual(class_indices, data_read[2])

  def testWriteAndReadToFileEmptyFile(self):
    tmpdir = tf.test.get_temp_dir()
    filename = os.path.join(tmpdir, 'test.box')
    box_io.WriteToFile(filename, np.array([]), np.array([]), np.array([]))
    data_read = box_io.ReadFromFile(filename)

    self.assertAllEqual(np.array([]), data_read[0])
    self.assertAllEqual(np.array([]), data_read[1])
    self.assertAllEqual(np.array([]), data_read[2])


if __name__ == '__main__':
  tf.test.main()
