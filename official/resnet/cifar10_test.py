# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import mkstemp

import numpy as np
import tensorflow as tf

import cifar10_main

tf.logging.set_verbosity(tf.logging.ERROR)

_BATCH_SIZE = 128


class BaseTest(tf.test.TestCase):

  def test_dataset_input_fn(self):
    fake_data = bytearray()
    fake_data.append(7)
    for i in range(3):
      for _ in range(1024):
        fake_data.append(i)

    _, filename = mkstemp(dir=self.get_temp_dir())
    data_file = open(filename, 'wb')
    data_file.write(fake_data)
    data_file.close()

    fake_dataset = tf.data.FixedLengthRecordDataset(
        filename, cifar10_main._RECORD_BYTES)
    fake_dataset = fake_dataset.map(
        lambda val: cifar10_main.parse_record(val, False))
    image, label = fake_dataset.make_one_shot_iterator().get_next()

    self.assertEqual(label.get_shape().as_list(), [10])
    self.assertEqual(image.get_shape().as_list(), [32, 32, 3])

    with self.test_session() as sess:
      image, label = sess.run([image, label])

      self.assertAllEqual(label, np.array([int(i == 7) for i in range(10)]))

      for row in image:
        for pixel in row:
          self.assertAllClose(pixel, np.array([-1.225, 0., 1.225]), rtol=1e-3)

  def input_fn(self):
    features = tf.random_uniform([_BATCH_SIZE, 32, 32, 3])
    labels = tf.random_uniform(
        [_BATCH_SIZE], maxval=9, dtype=tf.int32)
    return features, tf.one_hot(labels, 10)

  def cifar10_model_fn_helper(self, mode):
    features, labels = self.input_fn()
    spec = cifar10_main.cifar10_model_fn(
        features, labels, mode, {
            'resnet_size': 32,
            'data_format': 'channels_last',
            'batch_size': _BATCH_SIZE,
        })

    predictions = spec.predictions
    self.assertAllEqual(predictions['probabilities'].shape,
                        (_BATCH_SIZE, 10))
    self.assertEqual(predictions['probabilities'].dtype, tf.float32)
    self.assertAllEqual(predictions['classes'].shape, (_BATCH_SIZE,))
    self.assertEqual(predictions['classes'].dtype, tf.int64)

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = spec.loss
      self.assertAllEqual(loss.shape, ())
      self.assertEqual(loss.dtype, tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = spec.eval_metric_ops
      self.assertAllEqual(eval_metric_ops['accuracy'][0].shape, ())
      self.assertAllEqual(eval_metric_ops['accuracy'][1].shape, ())
      self.assertEqual(eval_metric_ops['accuracy'][0].dtype, tf.float32)
      self.assertEqual(eval_metric_ops['accuracy'][1].dtype, tf.float32)

  def test_cifar10_model_fn_train_mode(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.TRAIN)

  def test_cifar10_model_fn_eval_mode(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.EVAL)

  def test_cifar10_model_fn_predict_mode(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.PREDICT)


if __name__ == '__main__':
  tf.test.main()
