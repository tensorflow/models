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

"""Tests for object_detection.utils.dataset_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util


class DatasetUtilTest(tf.test.TestCase):

  def test_read_examples_list(self):
    example_list_data = """example1 1\nexample2 2"""
    example_list_path = os.path.join(self.get_temp_dir(), 'examples.txt')
    with tf.gfile.Open(example_list_path, 'wb') as f:
      f.write(example_list_data)

    examples = dataset_util.read_examples_list(example_list_path)
    self.assertListEqual(['example1', 'example2'], examples)


if __name__ == '__main__':
  tf.test.main()
