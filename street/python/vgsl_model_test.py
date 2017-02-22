# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for vgsl_model."""
import os

import numpy as np
import tensorflow as tf
import vgsl_input
import vgsl_model


def _testdata(filename):
  return os.path.join('../testdata/', filename)


def _rand(*size):
  return np.random.uniform(size=size).astype('f')


class VgslModelTest(tf.test.TestCase):

  def testParseInputSpec(self):
    """The parser must return the numbers in the correct order.
    """
    shape = vgsl_model._ParseInputSpec(input_spec='32,42,256,3')
    self.assertEqual(
        shape,
        vgsl_input.ImageShape(
            batch_size=32, height=42, width=256, depth=3))
    # Nones must be inserted for zero sizes.
    shape = vgsl_model._ParseInputSpec(input_spec='1,0,0,3')
    self.assertEqual(
        shape,
        vgsl_input.ImageShape(
            batch_size=1, height=None, width=None, depth=3))

  def testParseOutputSpec(self):
    """The parser must return the correct args in the correct order.
    """
    out_dims, out_func, num_classes = vgsl_model._ParseOutputSpec(
        output_spec='O1c142')
    self.assertEqual(out_dims, 1)
    self.assertEqual(out_func, 'c')
    self.assertEqual(num_classes, 142)
    out_dims, out_func, num_classes = vgsl_model._ParseOutputSpec(
        output_spec='O2s99')
    self.assertEqual(out_dims, 2)
    self.assertEqual(out_func, 's')
    self.assertEqual(num_classes, 99)
    out_dims, out_func, num_classes = vgsl_model._ParseOutputSpec(
        output_spec='O0l12')
    self.assertEqual(out_dims, 0)
    self.assertEqual(out_func, 'l')
    self.assertEqual(num_classes, 12)

  def testPadLabels2d(self):
    """Must pad timesteps in labels to match logits.
    """
    with self.test_session() as sess:
      # Make placeholders for logits and labels.
      ph_logits = tf.placeholder(tf.float32, shape=(None, None, 42))
      ph_labels = tf.placeholder(tf.int64, shape=(None, None))
      padded_labels = vgsl_model._PadLabels2d(tf.shape(ph_logits)[1], ph_labels)
      # Make actual inputs.
      real_logits = _rand(4, 97, 42)
      real_labels = _rand(4, 85)
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (4, 97))
      real_labels = _rand(4, 97)
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (4, 97))
      real_labels = _rand(4, 100)
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (4, 97))

  def testPadLabels3d(self):
    """Must pad height and width in labels to match logits.

    The tricky thing with 3-d is that the rows and columns need to remain
    intact, so we'll test it with small known data.
    """
    with self.test_session() as sess:
      # Make placeholders for logits and labels.
      ph_logits = tf.placeholder(tf.float32, shape=(None, None, None, 42))
      ph_labels = tf.placeholder(tf.int64, shape=(None, None, None))
      padded_labels = vgsl_model._PadLabels3d(ph_logits, ph_labels)
      # Make actual inputs.
      real_logits = _rand(1, 3, 4, 42)
      # Test all 9 combinations of height x width in [small, ok, big]
      real_labels = np.arange(6).reshape((1, 2, 3))  # Height small, width small
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 0], [3, 4, 5, 0], [0, 0, 0, 0]])
      real_labels = np.arange(8).reshape((1, 2, 4))  # Height small, width ok
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [4, 5, 6, 7], [0, 0, 0, 0]])
      real_labels = np.arange(10).reshape((1, 2, 5))  # Height small, width big
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [5, 6, 7, 8], [0, 0, 0, 0]])
      real_labels = np.arange(9).reshape((1, 3, 3))  # Height ok, width small
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]])
      real_labels = np.arange(12).reshape((1, 3, 4))  # Height ok, width ok
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
      real_labels = np.arange(15).reshape((1, 3, 5))  # Height ok, width big
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]])
      real_labels = np.arange(12).reshape((1, 4, 3))  # Height big, width small
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]])
      real_labels = np.arange(16).reshape((1, 4, 4))  # Height big, width ok
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
      real_labels = np.arange(20).reshape((1, 4, 5))  # Height big, width big
      np_array = sess.run([padded_labels],
                          feed_dict={ph_logits: real_logits,
                                     ph_labels: real_labels})[0]
      self.assertEqual(tuple(np_array.shape), (1, 3, 4))
      self.assertAllEqual(np_array[0, :, :],
                          [[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]])

  def testEndToEndSizes0d(self):
    """Tests that the output sizes match when training/running real 0d data.

    Uses mnist with dual summarizing LSTMs to reduce to a single value.
    """
    filename = _testdata('mnist-tiny')
    with self.test_session() as sess:
      model = vgsl_model.InitNetwork(
          filename,
          model_spec='4,0,0,1[Cr5,5,16 Mp3,3 Lfys16 Lfxs16]O0s12',
          mode='train')
      tf.global_variables_initializer().run(session=sess)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      _, step = model.TrainAStep(sess)
      self.assertEqual(step, 1)
      output, labels = model.RunAStep(sess)
      self.assertEqual(len(output.shape), 2)
      self.assertEqual(len(labels.shape), 1)
      self.assertEqual(output.shape[0], labels.shape[0])
      self.assertEqual(output.shape[1], 12)

  # TODO(rays) Support logistic and test with Imagenet (as 0d, multi-object.)

  def testEndToEndSizes1dCTC(self):
    """Tests that the output sizes match when training with CTC.

    Basic bidi LSTM on top of convolution and summarizing LSTM with CTC.
    """
    filename = _testdata('arial-32-tiny')
    with self.test_session() as sess:
      model = vgsl_model.InitNetwork(
          filename,
          model_spec='2,0,0,1[Cr5,5,16 Mp3,3 Lfys16 Lbx100]O1c105',
          mode='train')
      tf.global_variables_initializer().run(session=sess)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      _, step = model.TrainAStep(sess)
      self.assertEqual(step, 1)
      output, labels = model.RunAStep(sess)
      self.assertEqual(len(output.shape), 3)
      self.assertEqual(len(labels.shape), 2)
      self.assertEqual(output.shape[0], labels.shape[0])
      # This is ctc - the only cast-iron guarantee is labels <= output.
      self.assertLessEqual(labels.shape[1], output.shape[1])
      self.assertEqual(output.shape[2], 105)

  def testEndToEndSizes1dFixed(self):
    """Tests that the output sizes match when training/running 1 data.

    Convolution, summarizing LSTM with fwd rev fwd to allow no CTC.
    """
    filename = _testdata('numbers-16-tiny')
    with self.test_session() as sess:
      model = vgsl_model.InitNetwork(
          filename,
          model_spec='8,0,0,1[Cr5,5,16 Mp3,3 Lfys16 Lfx64 Lrx64 Lfx64]O1s12',
          mode='train')
      tf.global_variables_initializer().run(session=sess)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      _, step = model.TrainAStep(sess)
      self.assertEqual(step, 1)
      output, labels = model.RunAStep(sess)
      self.assertEqual(len(output.shape), 3)
      self.assertEqual(len(labels.shape), 2)
      self.assertEqual(output.shape[0], labels.shape[0])
      # Not CTC, output lengths match.
      self.assertEqual(output.shape[1], labels.shape[1])
      self.assertEqual(output.shape[2], 12)

  # TODO(rays) Get a 2-d dataset and support 2d (heat map) outputs.


if __name__ == '__main__':
  tf.test.main()
