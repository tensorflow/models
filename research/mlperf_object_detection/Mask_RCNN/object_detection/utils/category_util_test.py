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

"""Tests for object_detection.utils.category_util."""
import os

import tensorflow as tf

from object_detection.utils import category_util


class EvalUtilTest(tf.test.TestCase):

  def test_load_categories_from_csv_file(self):
    csv_data = """
        0,"cat"
        1,"dog"
        2,"bird"
    """.strip(' ')
    csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
    with tf.gfile.Open(csv_path, 'wb') as f:
      f.write(csv_data)

    categories = category_util.load_categories_from_csv_file(csv_path)
    self.assertTrue({'id': 0, 'name': 'cat'} in categories)
    self.assertTrue({'id': 1, 'name': 'dog'} in categories)
    self.assertTrue({'id': 2, 'name': 'bird'} in categories)

  def test_save_categories_to_csv_file(self):
    categories = [
        {'id': 0, 'name': 'cat'},
        {'id': 1, 'name': 'dog'},
        {'id': 2, 'name': 'bird'},
    ]
    csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
    category_util.save_categories_to_csv_file(categories, csv_path)
    saved_categories = category_util.load_categories_from_csv_file(csv_path)
    self.assertEqual(saved_categories, categories)


if __name__ == '__main__':
  tf.test.main()
