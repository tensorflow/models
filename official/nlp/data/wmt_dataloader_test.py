# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for official.nlp.data.wmt_dataloader."""
import os
import random
from absl import logging

import numpy as np
import tensorflow as tf

from official.nlp.data import wmt_dataloader


def _create_fake_dataset(output_path):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  for _ in range(20):
    features = {}
    seq_length = random.randint(20, 40)
    input_ids = np.random.randint(100, size=(seq_length))
    features['inputs'] = create_int_feature(input_ids)
    seq_length = random.randint(10, 80)
    targets = np.random.randint(100, size=(seq_length))
    features['targets'] = create_int_feature(targets)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class WMTDataLoaderTest(tf.test.TestCase):

  def test_load_dataset(self):
    batch_tokens_size = 100
    train_data_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(train_data_path)
    data_config = wmt_dataloader.WMTDataConfig(
        input_path=train_data_path,
        max_seq_length=35,
        global_batch_size=batch_tokens_size,
        static_batch=False)
    dataset = wmt_dataloader.WMTDataLoader(data_config).load()
    examples = next(iter(dataset))
    inputs, targets = examples['inputs'], examples['targets']
    logging.info('dynamic inputs=%s targets=%s', inputs, targets)
    data_config = wmt_dataloader.WMTDataConfig(
        input_path=train_data_path,
        max_seq_length=35,
        global_batch_size=batch_tokens_size,
        static_batch=True)
    dataset = wmt_dataloader.WMTDataLoader(data_config).load()
    examples = next(iter(dataset))
    inputs, targets = examples['inputs'], examples['targets']
    logging.info('static inputs=%s targets=%s', inputs, targets)
    self.assertEqual(inputs.shape, (2, 35))
    self.assertEqual(targets.shape, (2, 35))

  def test_load_dataset_raise_invalid_window(self):
    batch_tokens_size = 10  # this is too small to form buckets.
    train_data_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(train_data_path)
    data_config = wmt_dataloader.WMTDataConfig(
        input_path=train_data_path,
        max_seq_length=100,
        global_batch_size=batch_tokens_size)
    with self.assertRaisesRegex(
        ValueError, 'The token budget, global batch size, is too small.*'):
      _ = wmt_dataloader.WMTDataLoader(data_config).load()


if __name__ == '__main__':
  tf.test.main()
