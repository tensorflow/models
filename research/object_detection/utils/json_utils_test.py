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
"""Tests for google3.image.understanding.object_detection.utils.json_utils."""
import os

import tensorflow.compat.v1 as tf

from object_detection.utils import json_utils


class JsonUtilsTest(tf.test.TestCase):

  def testDumpReasonablePrecision(self):
    output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
    with tf.gfile.GFile(output_path, 'w') as f:
      json_utils.Dump(1.0, f, float_digits=2)
    with tf.gfile.GFile(output_path, 'r') as f:
      self.assertEqual(f.read(), '1.00')

  def testDumpPassExtraParams(self):
    output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
    with tf.gfile.GFile(output_path, 'w') as f:
      json_utils.Dump([1.12345], f, float_digits=2, indent=3)
    with tf.gfile.GFile(output_path, 'r') as f:
      self.assertEqual(f.read(), '[\n   1.12\n]')

  def testDumpZeroPrecision(self):
    output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
    with tf.gfile.GFile(output_path, 'w') as f:
      json_utils.Dump(1.0, f, float_digits=0, indent=3)
    with tf.gfile.GFile(output_path, 'r') as f:
      self.assertEqual(f.read(), '1')

  def testDumpUnspecifiedPrecision(self):
    output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
    with tf.gfile.GFile(output_path, 'w') as f:
      json_utils.Dump(1.012345, f)
    with tf.gfile.GFile(output_path, 'r') as f:
      self.assertEqual(f.read(), '1.012345')

  def testDumpsReasonablePrecision(self):
    s = json_utils.Dumps(1.12545, float_digits=2)
    self.assertEqual(s, '1.13')

  def testDumpsPassExtraParams(self):
    s = json_utils.Dumps([1.0], float_digits=2, indent=3)
    self.assertEqual(s, '[\n   1.00\n]')

  def testDumpsZeroPrecision(self):
    s = json_utils.Dumps(1.0, float_digits=0)
    self.assertEqual(s, '1')

  def testDumpsUnspecifiedPrecision(self):
    s = json_utils.Dumps(1.012345)
    self.assertEqual(s, '1.012345')

  def testPrettyParams(self):
    s = json_utils.Dumps({'v': 1.012345, 'n': 2}, **json_utils.PrettyParams())
    self.assertEqual(s, '{\n  "n": 2,\n  "v": 1.0123\n}')

  def testPrettyParamsExtraParamsInside(self):
    s = json_utils.Dumps(
        {'v': 1.012345,
         'n': float('nan')}, **json_utils.PrettyParams(allow_nan=True))
    self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')

    with self.assertRaises(ValueError):
      s = json_utils.Dumps(
          {'v': 1.012345,
           'n': float('nan')}, **json_utils.PrettyParams(allow_nan=False))

  def testPrettyParamsExtraParamsOutside(self):
    s = json_utils.Dumps(
        {'v': 1.012345,
         'n': float('nan')}, allow_nan=True, **json_utils.PrettyParams())
    self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')

    with self.assertRaises(ValueError):
      s = json_utils.Dumps(
          {'v': 1.012345,
           'n': float('nan')}, allow_nan=False, **json_utils.PrettyParams())


if __name__ == '__main__':
  tf.test.main()
