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

"""Tests for official.nlp.data.tagging_data_loader."""
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.data import tagging_dataloader


def _create_fake_dataset(output_path, seq_length, include_sentence_id):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  for i in range(100):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features['input_ids'] = create_int_feature(input_ids)
    features['input_mask'] = create_int_feature(np.ones_like(input_ids))
    features['segment_ids'] = create_int_feature(np.ones_like(input_ids))
    features['label_ids'] = create_int_feature(
        np.random.randint(10, size=(seq_length)))
    if include_sentence_id:
      features['sentence_id'] = create_int_feature([i])
      features['sub_sentence_id'] = create_int_feature([0])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class TaggingDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_load_dataset(self, include_sentence_id):
    seq_length = 16
    batch_size = 10
    train_data_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(train_data_path, seq_length, include_sentence_id)
    data_config = tagging_dataloader.TaggingDataConfig(
        input_path=train_data_path,
        seq_length=seq_length,
        global_batch_size=batch_size,
        include_sentence_id=include_sentence_id)

    dataset = tagging_dataloader.TaggingDataLoader(data_config).load()
    features, labels = next(iter(dataset))

    expected_keys = ['input_word_ids', 'input_mask', 'input_type_ids']
    if include_sentence_id:
      expected_keys.extend(['sentence_id', 'sub_sentence_id'])
    self.assertCountEqual(expected_keys, features.keys())

    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(labels.shape, (batch_size, seq_length))
    if include_sentence_id:
      self.assertEqual(features['sentence_id'].shape, (batch_size,))
      self.assertEqual(features['sub_sentence_id'].shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
