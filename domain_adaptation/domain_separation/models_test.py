# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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
"""Tests for DSN components."""

import numpy as np
import tensorflow as tf

#from models.domain_adaptation.domain_separation
import models


class SharedEncodersTest(tf.test.TestCase):

  def _testSharedEncoder(self,
                         input_shape=[5, 28, 28, 1],
                         model=models.dann_mnist,
                         is_training=True):
    images = tf.to_float(np.random.rand(*input_shape))

    with self.test_session() as sess:
      logits, _ = model(images)
      sess.run(tf.global_variables_initializer())
      logits_np = sess.run(logits)
    return logits_np

  def testBuildGRLMnistModel(self):
    logits = self._testSharedEncoder(model=getattr(models,
                                                   'dann_mnist'))
    self.assertEqual(logits.shape, (5, 10))
    self.assertTrue(np.any(logits))

  def testBuildGRLSvhnModel(self):
    logits = self._testSharedEncoder(model=getattr(models,
                                                   'dann_svhn'))
    self.assertEqual(logits.shape, (5, 10))
    self.assertTrue(np.any(logits))

  def testBuildGRLGtsrbModel(self):
    logits = self._testSharedEncoder([5, 40, 40, 3],
                                     getattr(models, 'dann_gtsrb'))
    self.assertEqual(logits.shape, (5, 43))
    self.assertTrue(np.any(logits))

  def testBuildPoseModel(self):
    logits = self._testSharedEncoder([5, 64, 64, 4],
                                     getattr(models, 'dsn_cropped_linemod'))
    self.assertEqual(logits.shape, (5, 11))
    self.assertTrue(np.any(logits))

  def testBuildPoseModelWithBatchNorm(self):
    images = tf.to_float(np.random.rand(10, 64, 64, 4))

    with self.test_session() as sess:
      logits, _ = getattr(models, 'dsn_cropped_linemod')(
          images, batch_norm_params=models.default_batch_norm_params(True))
      sess.run(tf.global_variables_initializer())
      logits_np = sess.run(logits)
    self.assertEqual(logits_np.shape, (10, 11))
    self.assertTrue(np.any(logits_np))


class EncoderTest(tf.test.TestCase):

  def _testEncoder(self, batch_norm_params=None, channels=1):
    images = tf.to_float(np.random.rand(10, 28, 28, channels))

    with self.test_session() as sess:
      end_points = models.default_encoder(
          images, 128, batch_norm_params=batch_norm_params)
      sess.run(tf.global_variables_initializer())
      private_code = sess.run(end_points['fc3'])
    self.assertEqual(private_code.shape, (10, 128))
    self.assertTrue(np.any(private_code))
    self.assertTrue(np.all(np.isfinite(private_code)))

  def testEncoder(self):
    self._testEncoder()

  def testEncoderMultiChannel(self):
    self._testEncoder(None, 4)

  def testEncoderIsTrainingBatchNorm(self):
    self._testEncoder(models.default_batch_norm_params(True))

  def testEncoderBatchNorm(self):
    self._testEncoder(models.default_batch_norm_params(False))


class DecoderTest(tf.test.TestCase):

  def _testDecoder(self,
                   height=64,
                   width=64,
                   channels=4,
                   batch_norm_params=None,
                   decoder=models.small_decoder):
    codes = tf.to_float(np.random.rand(32, 100))

    with self.test_session() as sess:
      output = decoder(
          codes,
          height=height,
          width=width,
          channels=channels,
          batch_norm_params=batch_norm_params)
      sess.run(tf.global_variables_initializer())
      output_np = sess.run(output)
    self.assertEqual(output_np.shape, (32, height, width, channels))
    self.assertTrue(np.any(output_np))
    self.assertTrue(np.all(np.isfinite(output_np)))

  def testSmallDecoder(self):
    self._testDecoder(28, 28, 4, None, getattr(models, 'small_decoder'))

  def testSmallDecoderThreeChannels(self):
    self._testDecoder(28, 28, 3)

  def testSmallDecoderBatchNorm(self):
    self._testDecoder(28, 28, 4, models.default_batch_norm_params(False))

  def testSmallDecoderIsTrainingBatchNorm(self):
    self._testDecoder(28, 28, 4, models.default_batch_norm_params(True))

  def testLargeDecoder(self):
    self._testDecoder(32, 32, 4, None, getattr(models, 'large_decoder'))

  def testLargeDecoderThreeChannels(self):
    self._testDecoder(32, 32, 3, None, getattr(models, 'large_decoder'))

  def testLargeDecoderBatchNorm(self):
    self._testDecoder(32, 32, 4,
                      models.default_batch_norm_params(False),
                      getattr(models, 'large_decoder'))

  def testLargeDecoderIsTrainingBatchNorm(self):
    self._testDecoder(32, 32, 4,
                      models.default_batch_norm_params(True),
                      getattr(models, 'large_decoder'))

  def testGtsrbDecoder(self):
    self._testDecoder(40, 40, 3, None, getattr(models, 'large_decoder'))

  def testGtsrbDecoderBatchNorm(self):
    self._testDecoder(40, 40, 4,
                      models.default_batch_norm_params(False),
                      getattr(models, 'gtsrb_decoder'))

  def testGtsrbDecoderIsTrainingBatchNorm(self):
    self._testDecoder(40, 40, 4,
                      models.default_batch_norm_params(True),
                      getattr(models, 'gtsrb_decoder'))


if __name__ == '__main__':
  tf.test.main()
