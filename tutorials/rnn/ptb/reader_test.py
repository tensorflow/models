# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for models.tutorials.rnn.ptb.reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

import reader


class PtbReaderTest(tf.test.TestCase):

  def setUp(self):
    self._string_data = "\n".join(
        [" hello there i am",
         " rain as day",
         " want some cheesy puffs ?"])

  def testPtbRawData(self):
    tmpdir = tf.test.get_temp_dir()
    for suffix in "train", "valid", "test":
      filename = os.path.join(tmpdir, "ptb.%s.txt" % suffix)
      with tf.gfile.GFile(filename, "w") as fh:
        fh.write(self._string_data)
    # Smoke test
    output = reader.ptb_raw_data(tmpdir)
    self.assertEqual(len(output), 4)

  def testPtbProducer(self):
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
    with self.test_session() as session:
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(session, coord=coord)
      try:
        xval, yval = session.run([x, y])
        self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
        self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
        xval, yval = session.run([x, y])
        self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
        self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
      finally:
        coord.request_stop()
        coord.join()


if __name__ == "__main__":
  tf.test.main()
