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
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main
from official.utils.misc import keras_utils
from official.utils.testing import integration

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

_BATCH_SIZE = 128
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3


class BaseTest(tf.test.TestCase):
  """Tests for the Cifar10 version of Resnet.
  """

  _num_validation_images = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(BaseTest, cls).setUpClass()
    if keras_utils.is_v2_0:
      tf.compat.v1.disable_eager_execution()
    cifar10_main.define_cifar_flags()

  def setUp(self):
    super(BaseTest, self).setUp()
    self._num_validation_images = cifar10_main.NUM_IMAGES['validation']
    cifar10_main.NUM_IMAGES['validation'] = 4

  def tearDown(self):
    super(BaseTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    cifar10_main.NUM_IMAGES['validation'] = self._num_validation_images

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
        filename, cifar10_main._RECORD_BYTES)  # pylint: disable=protected-access
    fake_dataset = fake_dataset.map(
        lambda val: cifar10_main.parse_record(val, False, tf.float32))
    image, label = tf.compat.v1.data.make_one_shot_iterator(
        fake_dataset).get_next()

    self.assertAllEqual(label.shape, ())
    self.assertAllEqual(image.shape, (_HEIGHT, _WIDTH, _NUM_CHANNELS))

    with self.session() as sess:
      image, label = sess.run([image, label])

      self.assertEqual(label, 7)

      for row in image:
        for pixel in row:
          self.assertAllClose(pixel, np.array([-1.225, 0., 1.225]), rtol=1e-3)

  def cifar10_model_fn_helper(self, mode, resnet_version, dtype):
    input_fn = cifar10_main.get_synth_input_fn(dtype)
    dataset = input_fn(True, '', _BATCH_SIZE)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    features, labels = iterator.get_next()
    spec = cifar10_main.cifar10_model_fn(
        features, labels, mode, {
            'dtype': dtype,
            'resnet_size': 32,
            'data_format': 'channels_last',
            'batch_size': _BATCH_SIZE,
            'resnet_version': resnet_version,
            'loss_scale': 128 if dtype == tf.float16 else 1,
            'fine_tune': False,
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

  def test_cifar10_model_fn_train_mode_v1(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.TRAIN, resnet_version=1,
                                 dtype=tf.float32)

  def test_cifar10_model_fn_trainmode__v2(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.TRAIN, resnet_version=2,
                                 dtype=tf.float32)

  def test_cifar10_model_fn_eval_mode_v1(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.EVAL, resnet_version=1,
                                 dtype=tf.float32)

  def test_cifar10_model_fn_eval_mode_v2(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.EVAL, resnet_version=2,
                                 dtype=tf.float32)

  def test_cifar10_model_fn_predict_mode_v1(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.PREDICT,
                                 resnet_version=1, dtype=tf.float32)

  def test_cifar10_model_fn_predict_mode_v2(self):
    self.cifar10_model_fn_helper(tf.estimator.ModeKeys.PREDICT,
                                 resnet_version=2, dtype=tf.float32)

  def _test_cifar10model_shape(self, resnet_version):
    batch_size = 135
    num_classes = 246

    model = cifar10_main.Cifar10Model(32, data_format='channels_last',
                                      num_classes=num_classes,
                                      resnet_version=resnet_version)
    fake_input = tf.random.uniform([batch_size, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    output = model(fake_input, training=True)

    self.assertAllEqual(output.shape, (batch_size, num_classes))

  def test_cifar10model_shape_v1(self):
    self._test_cifar10model_shape(resnet_version=1)

  def test_cifar10model_shape_v2(self):
    self._test_cifar10model_shape(resnet_version=2)

  def test_cifar10_end_to_end_synthetic_v1(self):
    integration.run_synthetic(
        main=cifar10_main.run_cifar, tmp_root=self.get_temp_dir(),
        extra_flags=['-resnet_version', '1', '-batch_size', '4']
    )

  def test_cifar10_end_to_end_synthetic_v2(self):
    integration.run_synthetic(
        main=cifar10_main.run_cifar, tmp_root=self.get_temp_dir(),
        extra_flags=['-resnet_version', '2', '-batch_size', '4']
    )


if __name__ == '__main__':
  tf.test.main()
