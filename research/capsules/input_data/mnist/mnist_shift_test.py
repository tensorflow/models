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

"""Tests for mnist_shift."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import mnist_input_record
import mnist_shift


class MnistShiftTest(tf.test.TestCase):

  def testReadImage(self):
    """Tests if the records are read in the expected order and value.

    Writes 3 images of size 28*28 into a temporary file. Calls the read_file
    function with the temporary file. Checks whether the order and value of
    pixels are correct for all 3 images.
    """
    colors = [0, 255, 100]
    height = 28
    size = height * height
    records = bytes(
        bytearray([1] * 16 + [colors[0]] * size + [colors[1]] * size +
                  [colors[2]] * size))
    expecteds = [
        np.zeros((height, height)) + colors[0],
        np.zeros((height, height)) + colors[1],
        np.zeros((height, height)) + colors[2]
    ]
    image_filename = os.path.join(self.get_temp_dir(), "mnist_image")
    open(image_filename, "wb").write(b"".join(records))

    with open(image_filename, "r") as f:
      images = mnist_shift.read_file(f, 4, len(colors) * size)
      images = images.reshape(len(colors), height, height)
      for i in range(len(colors)):
        self.assertAllEqual(expecteds[i], images[i])

  def testShift2d(self):
    """Tests if shifting of the image work as expected.

    Shifts the image in all direction with both positive and negative values.
    """
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected_shifted_up_right_one = [[0, 4, 5], [0, 7, 8], [0, 0, 0]]
    expected_shifted_down_left_two = [[0, 0, 0], [0, 0, 0], [3, 0, 0]]
    shifted_one = mnist_shift.shift_2d(image, (-1, 1), 3)
    shifted_two = mnist_shift.shift_2d(image, (2, -2), 2)
    self.assertAllEqual(expected_shifted_up_right_one, shifted_one)
    self.assertAllEqual(expected_shifted_down_left_two, shifted_two)

  def testShift2dZero(self):
    """Tests if shifting of the image with max_shift 0 returns the image."""
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    shifted_zero = mnist_shift.shift_2d(image, (0, 0), 0)
    self.assertAllEqual(image, shifted_zero)

  def testMultiShiftWrite(self):
    """Tests whether the output of shift_write_multi_mnist is readable.

    Writes a dataset of size two and reads the tfrecords files with
    mnist_input_record to check the integrity of the workflow.
    """
    colors = [255, 128]
    labels = [3, 4]
    dataset = []
    for i in range(2):
      dataset.append((np.zeros((28, 28)) + colors[i], labels[i]))
    expected_pair = np.minimum(dataset[0][0] + dataset[1][0], 255)
    file_prefix = os.path.join(self.get_temp_dir(), "mnist_multi")
    with self.test_session(graph=tf.Graph()) as session:
      mnist_shift.shift_write_multi_mnist(dataset, file_prefix, 0, 4, 1, 1)
      filenames = ['{}-{:0>5d}-of-{:0>5d}'.format(file_prefix, i, 2)
                   for i in range(2)]
      for filename in filenames:
        self.assertTrue(tf.gfile.Exists(filename))

      filename_queue = tf.train.string_input_producer(filenames)
      data, _ = mnist_input_record._multi_read_and_decode(filename_queue)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      pair_0 = session.run(data["images"])
      pair_1 = session.run(data["images"])
      self.assertAllEqual(pair_0, pair_1)
      pair_image = np.reshape(pair_0, (36, 36))
      self.assertAllClose(expected_pair / 255.0, pair_image[4:-4, 4:-4])

      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == "__main__":
  tf.test.main()
