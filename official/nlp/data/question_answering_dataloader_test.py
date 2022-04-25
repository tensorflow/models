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

"""Tests for official.nlp.data.question_answering_dataloader."""
import os

import numpy as np
import tensorflow as tf

from official.nlp.data import question_answering_dataloader


def _create_fake_dataset(output_path, seq_length):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  for _ in range(100):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features['input_ids'] = create_int_feature(input_ids)
    features['input_mask'] = create_int_feature(np.ones_like(input_ids))
    features['segment_ids'] = create_int_feature(np.ones_like(input_ids))
    features['start_positions'] = create_int_feature(np.array([0]))
    features['end_positions'] = create_int_feature(np.array([10]))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class QuestionAnsweringDataTest(tf.test.TestCase):

  def test_load_dataset(self):
    seq_length = 128
    batch_size = 10
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(input_path, seq_length)
    data_config = question_answering_dataloader.QADataConfig(
        is_training=True,
        input_path=input_path,
        seq_length=seq_length,
        global_batch_size=batch_size)
    dataset = question_answering_dataloader.QuestionAnsweringDataLoader(
        data_config).load()
    features, labels = next(iter(dataset))

    self.assertCountEqual(['input_word_ids', 'input_mask', 'input_type_ids'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))

    self.assertCountEqual(['start_positions', 'end_positions'], labels.keys())
    self.assertEqual(labels['start_positions'].shape, (batch_size,))
    self.assertEqual(labels['end_positions'].shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
