# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tfrecord_lib."""

import os

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from official.vision.data import tfrecord_lib


FLAGS = flags.FLAGS


def process_sample(x):
  d = {'x': x}
  return tf.train.Example(features=tf.train.Features(feature=d)), 0


def parse_function(example_proto):

  feature_description = {
      'x': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
  }
  return tf.io.parse_single_example(example_proto, feature_description)


class TfrecordLibTest(parameterized.TestCase):

  def test_write_tf_record_dataset(self):
    data = [(tfrecord_lib.convert_to_feature(i),) for i in range(17)]

    path = os.path.join(FLAGS.test_tmpdir, 'train')

    tfrecord_lib.write_tf_record_dataset(
        path, data, process_sample, 3, use_multiprocessing=False)
    tfrecord_files = tf.io.gfile.glob(path + '*')

    self.assertLen(tfrecord_files, 3)

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_function)

    read_values = set(d['x'] for d in dataset.as_numpy_iterator())
    self.assertSetEqual(read_values, set(range(17)))

  def test_convert_to_feature_float(self):

    proto = tfrecord_lib.convert_to_feature(0.0)
    self.assertEqual(proto.float_list.value[0], 0.0)

  def test_convert_to_feature_int(self):

    proto = tfrecord_lib.convert_to_feature(0)
    self.assertEqual(proto.int64_list.value[0], 0)

  def test_convert_to_feature_bytes(self):

    proto = tfrecord_lib.convert_to_feature(b'123')
    self.assertEqual(proto.bytes_list.value[0], b'123')

  def test_convert_to_feature_float_list(self):

    proto = tfrecord_lib.convert_to_feature([0.0, 1.0])
    self.assertSequenceAlmostEqual(proto.float_list.value, [0.0, 1.0])

  def test_convert_to_feature_int_list(self):

    proto = tfrecord_lib.convert_to_feature([0, 1])
    self.assertSequenceAlmostEqual(proto.int64_list.value, [0, 1])

  def test_convert_to_feature_bytes_list(self):

    proto = tfrecord_lib.convert_to_feature([b'123', b'456'])
    self.assertSequenceAlmostEqual(proto.bytes_list.value, [b'123', b'456'])


if __name__ == '__main__':
  tf.test.main()
