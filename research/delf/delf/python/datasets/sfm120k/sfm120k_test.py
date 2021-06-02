# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Sfm120k dataset module."""

import tensorflow as tf

from delf.python.datasets.sfm120k import sfm120k


class Sfm120kTest(tf.test.TestCase):
  """Tests for Sfm120k dataset module."""

  def testId2Filename(self):
    """Tests conversion of image id to full path mapping."""
    image_id = "29fdc243aeb939388cfdf2d081dc080e"
    prefix = "train/retrieval-SfM-120k/ims/"
    path = sfm120k.id2filename(image_id, prefix)
    expected_path = "train/retrieval-SfM-120k/ims/0e/08/dc" \
                    "/29fdc243aeb939388cfdf2d081dc080e"
    self.assertEqual(path, expected_path)


if __name__ == '__main__':
  tf.test.main()
