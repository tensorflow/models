# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for file_writers."""

import os
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.core import file_writers
from official.core import tf_example_builder


class FileWritersTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_bytes_feature('foo', 'Hello World!')
    self._example = example_builder.example

  @parameterized.parameters('tfrecord', 'TFRecord', 'tfrecords',
                            'tfrecord_compressed', 'TFRecord_Compressed',
                            'tfrecords_gzip')
  def test_write_small_dataset_success(self, file_type):
    temp_dir = self.create_tempdir()
    temp_dataset_file = os.path.join(temp_dir.full_path, 'train')
    file_writers.write_small_dataset([self._example], temp_dataset_file,
                                     file_type)
    self.assertTrue(os.path.exists(temp_dataset_file))

  def test_write_small_dataset_unrecognized_format(self):
    file_type = 'bar'
    temp_dir = self.create_tempdir()
    temp_dataset_file = os.path.join(temp_dir.full_path, 'train')
    with self.assertRaises(ValueError):
      file_writers.write_small_dataset([self._example], temp_dataset_file,
                                       file_type)


if __name__ == '__main__':
  tf.test.main()
