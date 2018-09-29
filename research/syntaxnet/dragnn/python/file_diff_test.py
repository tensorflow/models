# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Diff test that compares two files are identical."""

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('actual_file', None, 'File to test.')
flags.DEFINE_string('expected_file', None, 'File with expected contents.')


class DiffTest(tf.test.TestCase):

  def testEqualFiles(self):
    content_actual = None
    content_expected = None

    try:
      with open(FLAGS.actual_file) as actual:
        content_actual = actual.read()
    except IOError as e:
      self.fail("Error opening '%s': %s" % (FLAGS.actual_file, e.strerror))

    try:
      with open(FLAGS.expected_file) as expected:
        content_expected = expected.read()
    except IOError as e:
      self.fail("Error opening '%s': %s" % (FLAGS.expected_file, e.strerror))

    self.assertTrue(content_actual == content_expected)


if __name__ == '__main__':
  tf.test.main()
