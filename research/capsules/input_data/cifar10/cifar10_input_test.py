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

"""Tests for cifar10_input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import cifar10_input


class CIFAR10InputTest(tf.test.TestCase):

  def _record(self, label, colors):
    image_size = 32 * 32
    record = bytes(
        bytearray([label] + [colors[0]] * image_size +
                  [colors[1]] * image_size + [colors[2]] * image_size))
    expected = [[colors] * 32] * 32
    return record, expected

  def testRead(self):
    """Tests if the records are read in the expected order and value."""
    labels = [0, 1, 9]
    colors = [[0, 0, 0], [255, 255, 255], [1, 100, 253]]
    records = []
    expecteds = []
    for i in range(3):
      record, expected = self._record(labels[i], colors[i])
      records.append(record)
      expecteds.append(expected)
    filename = os.path.join(self.get_temp_dir(), "cifar_test")
    open(filename, "wb").write(b"".join(records))

    with self.test_session() as sess:
      q = tf.FIFOQueue(100, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      image_tensor, label_tensor = cifar10_input._read_input(q)

      for i in range(3):
        image, label = sess.run([image_tensor, label_tensor])
        self.assertEqual(labels[i], label)
        self.assertAllEqual(expecteds[i], image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(image_tensor)

  def testBatchedOuput(self):
    """Tests if the final output of batching works properly."""
    record, _ = self._record(5, [255, 0, 128])
    batch_size = 10
    expected_labels = [5 for _ in range(batch_size)]
    data_dir = self.get_temp_dir()
    filename = os.path.join(data_dir, "test_batch.bin")
    open(filename, "wb").write(b"".join([record]))
    features = cifar10_input.inputs("test", data_dir, batch_size)

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      labels = sess.run(features["recons_label"])
      self.assertAllEqual(expected_labels, labels)
      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == "__main__":
  tf.test.main()
