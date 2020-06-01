# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf_record_creation_util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import contextlib2
import six
from six.moves import range
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util


class OpenOutputTfrecordsTests(tf.test.TestCase):

  def test_sharded_tfrecord_writes(self):
    with contextlib2.ExitStack() as tf_record_close_stack:
      output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
          tf_record_close_stack,
          os.path.join(tf.test.get_temp_dir(), 'test.tfrec'), 10)
      for idx in range(10):
        output_tfrecords[idx].write(six.ensure_binary('test_{}'.format(idx)))

    for idx in range(10):
      tf_record_path = '{}-{:05d}-of-00010'.format(
          os.path.join(tf.test.get_temp_dir(), 'test.tfrec'), idx)
      records = list(tf.python_io.tf_record_iterator(tf_record_path))
      self.assertAllEqual(records, ['test_{}'.format(idx)])


if __name__ == '__main__':
  tf.test.main()
