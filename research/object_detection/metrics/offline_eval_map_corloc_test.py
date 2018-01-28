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
"""Tests for utilities in offline_eval_map_corloc binary."""

import tensorflow as tf

from object_detection.metrics import offline_eval_map_corloc as offline_eval


class OfflineEvalMapCorlocTest(tf.test.TestCase):

  def test_generateShardedFilenames(self):
    test_filename = '/path/to/file'
    result = offline_eval._generate_sharded_filenames(test_filename)
    self.assertEqual(result, [test_filename])

    test_filename = '/path/to/file-00000-of-00050'
    result = offline_eval._generate_sharded_filenames(test_filename)
    self.assertEqual(result, [test_filename])

    result = offline_eval._generate_sharded_filenames('/path/to/@3.record')
    self.assertEqual(result, [
        '/path/to/-00000-of-00003.record', '/path/to/-00001-of-00003.record',
        '/path/to/-00002-of-00003.record'
    ])

    result = offline_eval._generate_sharded_filenames('/path/to/abc@3')
    self.assertEqual(result, [
        '/path/to/abc-00000-of-00003', '/path/to/abc-00001-of-00003',
        '/path/to/abc-00002-of-00003'
    ])

    result = offline_eval._generate_sharded_filenames('/path/to/@1')
    self.assertEqual(result, ['/path/to/-00000-of-00001'])

  def test_generateFilenames(self):
    test_filenames = ['/path/to/file', '/path/to/@3.record']
    result = offline_eval._generate_filenames(test_filenames)
    self.assertEqual(result, [
        '/path/to/file', '/path/to/-00000-of-00003.record',
        '/path/to/-00001-of-00003.record', '/path/to/-00002-of-00003.record'
    ])


if __name__ == '__main__':
  tf.test.main()
