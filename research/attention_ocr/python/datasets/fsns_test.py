# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for FSNS datasets module."""

import collections
import os
import tensorflow as tf
from tensorflow.contrib import slim

from datasets import fsns
from datasets import unittest_utils
from tensorflow.compat.v1 import flags

FLAGS = flags.FLAGS


def get_test_split():
  config = fsns.DEFAULT_CONFIG.copy()
  config['splits'] = {'test': {'size': 5, 'pattern': 'fsns-00000-of-00001'}}
  return fsns.get_split('test', dataset_dir(), config)


def dataset_dir():
  return os.path.join(os.path.dirname(__file__), 'testdata/fsns')


class FsnsTest(tf.test.TestCase):
  def test_decodes_example_proto(self):
    expected_label = range(37)
    expected_image, encoded = unittest_utils.create_random_image(
        'PNG', shape=(150, 600, 3))
    serialized = unittest_utils.create_serialized_example({
        'image/encoded': [encoded],
        'image/format': [b'PNG'],
        'image/class':
        expected_label,
        'image/unpadded_class':
        range(10),
        'image/text': [b'Raw text'],
        'image/orig_width': [150],
        'image/width': [600]
    })

    decoder = fsns.get_split('train', dataset_dir()).decoder
    with self.test_session() as sess:
      data_tuple = collections.namedtuple('DecodedData', decoder.list_items())
      data = sess.run(data_tuple(*decoder.decode(serialized)))

    self.assertAllEqual(expected_image, data.image)
    self.assertAllEqual(expected_label, data.label)
    self.assertEqual([b'Raw text'], data.text)
    self.assertEqual([1], data.num_of_views)

  def test_label_has_shape_defined(self):
    serialized = 'fake'
    decoder = fsns.get_split('train', dataset_dir()).decoder

    [label_tf] = decoder.decode(serialized, ['label'])

    self.assertEqual(label_tf.get_shape().dims[0], 37)

  def test_dataset_tuple_has_all_extra_attributes(self):
    dataset = fsns.get_split('train', dataset_dir())

    self.assertTrue(dataset.charset)
    self.assertTrue(dataset.num_char_classes)
    self.assertTrue(dataset.num_of_views)
    self.assertTrue(dataset.max_sequence_length)
    self.assertTrue(dataset.null_code)

  def test_can_use_the_test_data(self):
    batch_size = 1
    dataset = get_test_split()
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=True,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)
    image_tf, label_tf = provider.get(['image', 'label'])

    with self.test_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      with slim.queues.QueueRunners(sess):
        image_np, label_np = sess.run([image_tf, label_tf])

    self.assertEqual((150, 600, 3), image_np.shape)
    self.assertEqual((37, ), label_np.shape)


if __name__ == '__main__':
  tf.test.main()
