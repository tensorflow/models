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

"""Tests for norb_input_record."""
import os

import numpy as np
import tensorflow as tf


import norb_input_record

FLAGS = tf.flags.FLAGS
DATA_DIR = '../testdata/norb/'


class NorbInputRecordTest(tf.test.TestCase):

  def testImageResize(self):
    """Checks the returned image is resized to the given dimmensions."""
    with self.test_session(graph=tf.Graph()) as session:
      features = norb_input_record.inputs(
          data_dir=os.path.join(DATA_DIR),
          batch_size=1,
          split='train',
          height=32,
          distort=False,
          batch_capacity=6)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, labels = session.run([features['images'], features['labels']])
      self.assertEqual((1, 5), labels.shape)
      self.assertEqual(1, np.sum(labels))
      self.assertItemsEqual([0, 1], np.unique(labels))
      self.assertEqual(32, features['height'])
      self.assertEqual((1, 2, 32, 32), images.shape)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testDistort(self):
    """Checks the dimmensions of the distorted image."""
    with self.test_session(graph=tf.Graph()) as sess:
      features = norb_input_record.inputs(
          data_dir=os.path.join(DATA_DIR),
          batch_size=1,
          split='test',
          height=32,
          distort=True,
          batch_capacity=6)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images = sess.run(features['images'])
      self.assertEqual(22, features['height'])
      self.assertEqual((1, 2, 22, 22), images.shape)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testImageRange(self):
    """Checks the image to be zero meaned with std of one."""
    with self.test_session(graph=tf.Graph()) as sess:
      features = norb_input_record.inputs(
          data_dir=os.path.join(DATA_DIR),
          batch_size=1,
          split='train',
          batch_capacity=6)
      image_mean = tf.reduce_mean(features['images'])
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      image, mean = sess.run([features['images'], image_mean])
      self.assertAllClose(1.0, np.std(image), atol=1e-2)
      self.assertAllClose(0, mean, atol=1e-3)

      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == '__main__':
  tf.test.main()
