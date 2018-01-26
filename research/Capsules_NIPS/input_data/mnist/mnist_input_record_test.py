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

"""Tests for mnist_input_record."""

import numpy as np
import tensorflow as tf

import mnist_input_record

MNIST_MULTI_TRAIN_FILE = 'multitrain_4shifted_mnist.tfrecords@3'
MNIST_MULTI_TEST_FILE = 'multitest_4shifted_mnist.tfrecords@2'
MNIST_DATA_DIR = '../testdata/mnist/'


class MnistInputRecordTest(tf.test.TestCase):

  def testSingleTrain(self):
    with self.test_session(graph=tf.Graph()) as sess:
      features = mnist_input_record.inputs(
          data_dir=MNIST_DATA_DIR,
          batch_size=1,
          split='train',
          num_targets=1,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, labels, recons_image = sess.run(
          [features['images'], features['labels'], features['recons_image']])
      self.assertEqual((1, 10), labels.shape)
      self.assertEqual(1, np.sum(labels))
      self.assertItemsEqual([0, 1], np.unique(labels))
      self.assertEqual(28, features['height'])
      self.assertEqual((1, 28, 28, 1), images.shape)
      self.assertEqual(recons_image.shape, images.shape)
      self.assertAllEqual(recons_image, images)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testSingleTrainDistorted(self):
    with self.test_session(graph=tf.Graph()) as sess:
      features = mnist_input_record.inputs(
          data_dir=MNIST_DATA_DIR,
          batch_size=1,
          split='train',
          num_targets=1,
          distort=True,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, labels, recons_image = sess.run(
          [features['images'], features['labels'], features['recons_image']])
      self.assertEqual((1, 10), labels.shape)
      self.assertEqual(1, np.sum(labels))
      self.assertItemsEqual([0, 1], np.unique(labels))
      self.assertEqual(24, features['height'])
      self.assertEqual((1, 24, 24, 1), images.shape)
      self.assertEqual(recons_image.shape, images.shape)
      self.assertAllEqual(recons_image, images)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testSingleTestDistorted(self):
    with self.test_session(graph=tf.Graph()) as sess:
      features = mnist_input_record.inputs(
          data_dir=MNIST_DATA_DIR,
          batch_size=1,
          split='test',
          num_targets=1,
          distort=True,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, recons_image, recons_label = sess.run([
          features['images'], features['recons_image'], features['recons_label']
      ])
      self.assertEqual([7], recons_label)
      self.assertEqual(24, features['height'])
      self.assertEqual((1, 24, 24, 1), images.shape)
      self.assertAllEqual(recons_image, images)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testMultiTrain(self):
    data_dir = MNIST_DATA_DIR + MNIST_MULTI_TRAIN_FILE
    with self.test_session(graph=tf.Graph()) as sess:
      features = mnist_input_record.inputs(
          data_dir=data_dir,
          batch_size=1,
          split='train',
          num_targets=2,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, labels, recons_image, spare_image = sess.run([
          features['images'], features['labels'], features['recons_image'],
          features['spare_image']
      ])
      self.assertEqual((1, 10), labels.shape)
      self.assertEqual(2, np.sum(labels))
      self.assertItemsEqual([0, 1], np.unique(labels))
      self.assertEqual(36, features['height'])
      self.assertEqual((1, 36 * 36), images.shape)
      self.assertEqual(recons_image.shape, images.shape)
      self.assertEqual(spare_image.shape, images.shape)

      coord.request_stop()
      for thread in threads:
        thread.join()

  def testMultiTest(self):
    data_dir = MNIST_DATA_DIR + MNIST_MULTI_TEST_FILE
    with self.test_session(graph=tf.Graph()) as sess:
      test_features = mnist_input_record.inputs(
          data_dir=data_dir,
          batch_size=1,
          split='test',
          num_targets=2,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      test_label = sess.run([test_features['recons_label']])
      self.assertEqual([7], test_label)

      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == '__main__':
  tf.test.main()
