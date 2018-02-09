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
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3


class BaseTest(tf.test.TestCase):

  def test_dataset_input_fn(self):
    fake_data = bytearray()
    fake_data.append(7)
    for i in range(_NUM_CHANNELS):
      for _ in range(_HEIGHT * _WIDTH):
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

    self.assertAllEqual(label.shape, (10,))
    self.assertAllEqual(image.shape, (_HEIGHT, _WIDTH, _NUM_CHANNELS))

    with self.test_session() as sess:
      image, label = sess.run([image, label])

      self.assertAllEqual(label, np.array([int(i == 7) for i in range(10)]))

      for row in image:
        for pixel in row:
          self.assertAllClose(pixel, np.array([-1.225, 0., 1.225]), rtol=1e-3)

  def input_fn(self):
    features = tf.random_uniform([_BATCH_SIZE, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    labels = tf.random_uniform(
        [_BATCH_SIZE], maxval=9, dtype=tf.int32)
    return features, tf.one_hot(labels, 10)

  def cifar10_model_fn_helper(self, mode, multi_gpu=False):
    features, labels = self.input_fn()
    spec = cifar10_main.cifar10_model_fn(
        features, labels, mode, {
            'resnet_size': 32,
            'data_format': 'channels_last',
            'batch_size': _BATCH_SIZE,
            'multi_gpu': multi_gpu
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

  def test_cifar10_model_fn_train_mode_multi_gpu(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.TRAIN, multi_gpu=True)

  def test_cifar10_model_fn_eval_mode(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.EVAL)

  def test_cifar10_model_fn_predict_mode(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.PREDICT)

  def test_cifar10model_shape(self):
    batch_size = 135
    num_classes = 246

    model = cifar10_main.Cifar10Model(
        32, data_format='channels_last', num_classes=num_classes)
    fake_input = tf.random_uniform([batch_size, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    output = model(fake_input, training=True)

    self.assertAllEqual(output.shape, (batch_size, num_classes))


if __name__ == '__main__':
  tf.test.main()
