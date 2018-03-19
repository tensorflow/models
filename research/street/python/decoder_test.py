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
"""Tests for decoder."""
import os

import tensorflow as tf
import decoder


def _testdata(filename):
  return os.path.join('../testdata/', filename)


class DecoderTest(tf.test.TestCase):

  def testCodesFromCTC(self):
    """Tests that the simple CTC decoder drops nulls and duplicates.
    """
    ctc_labels = [9, 9, 9, 1, 9, 2, 2, 3, 9, 9, 0, 0, 1, 9, 1, 9, 9, 9]
    decode = decoder.Decoder(filename=None)
    non_null_labels = decode._CodesFromCTC(
        ctc_labels, merge_dups=False, null_label=9)
    self.assertEqual(non_null_labels, [1, 2, 2, 3, 0, 0, 1, 1])
    idempotent_labels = decode._CodesFromCTC(
        non_null_labels, merge_dups=False, null_label=9)
    self.assertEqual(idempotent_labels, non_null_labels)
    collapsed_labels = decode._CodesFromCTC(
        ctc_labels, merge_dups=True, null_label=9)
    self.assertEqual(collapsed_labels, [1, 2, 3, 0, 1, 1])
    non_idempotent_labels = decode._CodesFromCTC(
        collapsed_labels, merge_dups=True, null_label=9)
    self.assertEqual(non_idempotent_labels, [1, 2, 3, 0, 1])

  def testStringFromCTC(self):
    """Tests that the decoder can decode sequences including multi-codes.
    """
    #             -  f  -  a  r  -  m(1/2)m     -junk sp b  a  r  -  n  -
    ctc_labels = [9, 6, 9, 1, 3, 9, 4, 9, 5, 5, 9, 5, 0, 2, 1, 3, 9, 4, 9]
    decode = decoder.Decoder(filename=_testdata('charset_size_10.txt'))
    text = decode.StringFromCTC(ctc_labels, merge_dups=True, null_label=9)
    self.assertEqual(text, 'farm barn')


if __name__ == '__main__':
  tf.test.main()
