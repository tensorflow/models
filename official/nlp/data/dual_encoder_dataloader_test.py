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

"""Tests for official.nlp.data.dual_encoder_dataloader."""
import os

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.data import dual_encoder_dataloader


_LEFT_FEATURE_NAME = 'left_input'
_RIGHT_FEATURE_NAME = 'right_input'


def _create_fake_dataset(output_path):
  """Creates a fake dataset contains examples for training a dual encoder model.

    The created dataset contains examples with two byteslist features keyed by
    _LEFT_FEATURE_NAME and _RIGHT_FEATURE_NAME.

  Args:
    output_path: The output path of the fake dataset.
  """
  def create_str_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

  with tf.io.TFRecordWriter(output_path) as writer:
    for _ in range(100):
      features = {}
      features[_LEFT_FEATURE_NAME] = create_str_feature([b'hello world.'])
      features[_RIGHT_FEATURE_NAME] = create_str_feature([b'world hello.'])

      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())


def _make_vocab_file(vocab, output_path):
  with tf.io.gfile.GFile(output_path, 'w') as f:
    f.write('\n'.join(vocab + ['']))


class DualEncoderDataTest(tf.test.TestCase, parameterized.TestCase):

  def test_load_dataset(self):
    seq_length = 16
    batch_size = 10
    train_data_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    vocab_path = os.path.join(self.get_temp_dir(), 'vocab.txt')

    _create_fake_dataset(train_data_path)
    _make_vocab_file(
        ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'he', '#llo', 'world'], vocab_path)

    data_config = dual_encoder_dataloader.DualEncoderDataConfig(
        input_path=train_data_path,
        seq_length=seq_length,
        vocab_file=vocab_path,
        lower_case=True,
        left_text_fields=(_LEFT_FEATURE_NAME,),
        right_text_fields=(_RIGHT_FEATURE_NAME,),
        global_batch_size=batch_size)
    dataset = dual_encoder_dataloader.DualEncoderDataLoader(
        data_config).load()
    features = next(iter(dataset))
    self.assertCountEqual(
        ['left_word_ids', 'left_mask', 'left_type_ids', 'right_word_ids',
         'right_mask', 'right_type_ids'],
        features.keys())
    self.assertEqual(features['left_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['left_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['left_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_type_ids'].shape, (batch_size, seq_length))

  @parameterized.parameters(False, True)
  def test_load_tfds(self, use_preprocessing_hub):
    seq_length = 16
    batch_size = 10
    if use_preprocessing_hub:
      vocab_path = ''
      preprocessing_hub = (
          'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3')
    else:
      vocab_path = os.path.join(self.get_temp_dir(), 'vocab.txt')
      _make_vocab_file(
          ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'he', '#llo', 'world'],
          vocab_path)
      preprocessing_hub = ''

    data_config = dual_encoder_dataloader.DualEncoderDataConfig(
        tfds_name='para_crawl/enmt',
        tfds_split='train',
        seq_length=seq_length,
        vocab_file=vocab_path,
        lower_case=True,
        left_text_fields=('en',),
        right_text_fields=('mt',),
        preprocessing_hub_module_url=preprocessing_hub,
        global_batch_size=batch_size)
    dataset = dual_encoder_dataloader.DualEncoderDataLoader(
        data_config).load()
    features = next(iter(dataset))
    self.assertCountEqual(
        ['left_word_ids', 'left_mask', 'left_type_ids', 'right_word_ids',
         'right_mask', 'right_type_ids'],
        features.keys())
    self.assertEqual(features['left_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['left_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['left_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['right_type_ids'].shape, (batch_size, seq_length))


if __name__ == '__main__':
  tf.test.main()
