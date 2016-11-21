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
"""Tests for errorcounter."""
import tensorflow as tf
import errorcounter as ec


class ErrorcounterTest(tf.test.TestCase):

  def testComputeErrorRate(self):
    """Tests that the percent calculation works as expected.
    """
    rate = ec.ComputeErrorRate(error_count=0, truth_count=0)
    self.assertEqual(rate, 100.0)
    rate = ec.ComputeErrorRate(error_count=1, truth_count=0)
    self.assertEqual(rate, 100.0)
    rate = ec.ComputeErrorRate(error_count=10, truth_count=1)
    self.assertEqual(rate, 100.0)
    rate = ec.ComputeErrorRate(error_count=0, truth_count=1)
    self.assertEqual(rate, 0.0)
    rate = ec.ComputeErrorRate(error_count=3, truth_count=12)
    self.assertEqual(rate, 25.0)

  def testCountErrors(self):
    """Tests that the error counter works as expected.
    """
    truth_str = 'farm barn'
    counts = ec.CountErrors(ocr_text=truth_str, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=0, truth_count=9, test_count=9))
    # With a period on the end, we get a char error.
    dot_str = 'farm barn.'
    counts = ec.CountErrors(ocr_text=dot_str, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=1, truth_count=9, test_count=10))
    counts = ec.CountErrors(ocr_text=truth_str, truth_text=dot_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=0, truth_count=10, test_count=9))
    # Space is just another char.
    no_space = 'farmbarn'
    counts = ec.CountErrors(ocr_text=no_space, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=0, truth_count=9, test_count=8))
    counts = ec.CountErrors(ocr_text=truth_str, truth_text=no_space)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=1, truth_count=8, test_count=9))
    # Lose them all.
    counts = ec.CountErrors(ocr_text='', truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=9, fp=0, truth_count=9, test_count=0))
    counts = ec.CountErrors(ocr_text=truth_str, truth_text='')
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=9, truth_count=0, test_count=9))

  def testCountWordErrors(self):
    """Tests that the error counter works as expected.
    """
    truth_str = 'farm barn'
    counts = ec.CountWordErrors(ocr_text=truth_str, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=0, truth_count=2, test_count=2))
    # With a period on the end, we get a word error.
    dot_str = 'farm barn.'
    counts = ec.CountWordErrors(ocr_text=dot_str, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=1, truth_count=2, test_count=2))
    counts = ec.CountWordErrors(ocr_text=truth_str, truth_text=dot_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=1, truth_count=2, test_count=2))
    # Space is special.
    no_space = 'farmbarn'
    counts = ec.CountWordErrors(ocr_text=no_space, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=2, fp=1, truth_count=2, test_count=1))
    counts = ec.CountWordErrors(ocr_text=truth_str, truth_text=no_space)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=2, truth_count=1, test_count=2))
    # Lose them all.
    counts = ec.CountWordErrors(ocr_text='', truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=2, fp=0, truth_count=2, test_count=0))
    counts = ec.CountWordErrors(ocr_text=truth_str, truth_text='')
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=0, fp=2, truth_count=0, test_count=2))
    # With a space in ba rn, there is an extra add.
    sp_str = 'farm ba rn'
    counts = ec.CountWordErrors(ocr_text=sp_str, truth_text=truth_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=1, fp=2, truth_count=2, test_count=3))
    counts = ec.CountWordErrors(ocr_text=truth_str, truth_text=sp_str)
    self.assertEqual(
        counts, ec.ErrorCounts(
            fn=2, fp=1, truth_count=3, test_count=2))


if __name__ == '__main__':
  tf.test.main()
