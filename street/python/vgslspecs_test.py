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
"""Tests for vgslspecs."""

import numpy as np
import tensorflow as tf
import vgslspecs


def _rand(*size):
  return np.random.uniform(size=size).astype('f')


class VgslspecsTest(tf.test.TestCase):

  def __init__(self, other):
    super(VgslspecsTest, self).__init__(other)
    self.max_width = 36
    self.max_height = 24
    self.batch_size = 4

  def SetupInputs(self):
    # Make placeholders for standard inputs.
    # Everything is variable in the input, except the depth.
    self.ph_image = tf.placeholder(
        tf.float32, shape=(None, None, None, 3), name='inputs')
    self.ph_widths = tf.placeholder(tf.int64, shape=(None,), name='w')
    self.ph_heights = tf.placeholder(tf.int64, shape=(None,), name='h')
    # Make actual inputs.
    self.in_image = _rand(self.batch_size, self.max_height, self.max_width, 3)
    self.in_widths = [24, 12, self.max_width, 30]
    self.in_heights = [self.max_height, 18, 12, 6]

  def ExpectScaledSize(self, spec, target_shape, factor=1):
    """Tests that the output of the graph of the given spec has target_shape."""
    with tf.Graph().as_default():
      with self.test_session() as sess:
        self.SetupInputs()
        # Only the placeholders are given at construction time.
        vgsl = vgslspecs.VGSLSpecs(self.ph_widths, self.ph_heights, True)
        outputs = vgsl.Build(self.ph_image, spec)
        # Compute the expected output widths from the given scale factor.
        target_widths = tf.div(self.in_widths, factor).eval()
        target_heights = tf.div(self.in_heights, factor).eval()
        # Run with the 'real' data.
        tf.initialize_all_variables().run()
        res_image, res_widths, res_heights = sess.run(
            [outputs, vgsl.GetLengths(2), vgsl.GetLengths(1)],
            feed_dict={self.ph_image: self.in_image,
                       self.ph_widths: self.in_widths,
                       self.ph_heights: self.in_heights})
        self.assertEqual(tuple(res_image.shape), target_shape)
        if target_shape[1] > 1:
          self.assertEqual(tuple(res_heights), tuple(target_heights))
        if target_shape[2] > 1:
          self.assertEqual(tuple(res_widths), tuple(target_widths))

  def testSameSizeConv(self):
    """Test all types of Conv. There is no scaling."""
    self.ExpectScaledSize(
        '[Cs{MyConv}5,5,16 Ct3,3,12 Cr4,4,24 Cl5,5,64]',
        (self.batch_size, self.max_height, self.max_width, 64))

  def testSameSizeLSTM(self):
    """Test all non-reducing LSTMs. Output depth is doubled with BiDi."""
    self.ExpectScaledSize('[Lfx16 Lrx8 Do Lbx24 Lfy12 Do{MyDo} Lry7 Lby32]',
                          (self.batch_size, self.max_height, self.max_width,
                           64))

  def testSameSizeParallel(self):
    """Parallel affects depth, but not scale."""
    self.ExpectScaledSize('[Cs5,5,16 (Lfx{MyLSTM}32 Lrx32 Lbx16)]',
                          (self.batch_size, self.max_height, self.max_width,
                           96))

  def testScalingOps(self):
    """Test a heterogeneous series with scaling."""
    self.ExpectScaledSize('[Cs5,5,16 Mp{MyPool}2,2 Ct3,3,32 Mp3,3 Lfx32 Lry64]',
                          (self.batch_size, self.max_height / 6,
                           self.max_width / 6, 64), 6)

  def testXReduction(self):
    """Test a heterogeneous series with reduction of x-dimension."""
    self.ExpectScaledSize('[Cr5,5,16 Mp2,2 Ct3,3,32 Mp3,3 Lfxs32 Lry64]',
                          (self.batch_size, self.max_height / 6, 1, 64), 6)

  def testYReduction(self):
    """Test a heterogeneous series with reduction of y-dimension."""
    self.ExpectScaledSize('[Cl5,5,16 Mp2,2 Ct3,3,32 Mp3,3 Lfys32 Lfx64]',
                          (self.batch_size, 1, self.max_width / 6, 64), 6)

  def testXYReduction(self):
    """Test a heterogeneous series with reduction to 0-d."""
    self.ExpectScaledSize(
        '[Cr5,5,16 Lfys32 Lfxs64 Fr{MyFC}16 Ft20 Fl12 Fs32 Fm40]',
        (self.batch_size, 1, 1, 40))

  def testReshapeTile(self):
    """Tests that a tiled input can be reshaped to the batch dimension."""
    self.ExpectScaledSize('[S2(3x0)0,2 Cr5,5,16 Lfys16]',
                          (self.batch_size * 3, 1, self.max_width / 3, 16), 3)

  def testReshapeDepth(self):
    """Tests that depth can be reshaped to the x dimension."""
    self.ExpectScaledSize('[Cl5,5,16 Mp3,3 (Lrys32 Lbys16 Lfys32) S3(3x0)2,3]',
                          (self.batch_size, 1, self.max_width, 32))


if __name__ == '__main__':
  tf.test.main()
