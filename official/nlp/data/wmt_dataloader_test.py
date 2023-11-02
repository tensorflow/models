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

"""Tests for official.nlp.data.wmt_dataloader."""
import os
from absl.testing import parameterized

import tensorflow as tf, tf_keras

from sentencepiece import SentencePieceTrainer
from official.nlp.data import wmt_dataloader


def _generate_line_file(filepath, lines):
  with tf.io.gfile.GFile(filepath, 'w') as f:
    for l in lines:
      f.write('{}\n'.format(l))


def _generate_record_file(filepath, src_lines, tgt_lines, unique_id=False):
  writer = tf.io.TFRecordWriter(filepath)
  for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
    features = {
        'en': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[src.encode()])),
        'reverse_en': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tgt.encode()])),
    }
    if unique_id:
      features['unique_id'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[i]))
    example = tf.train.Example(
        features=tf.train.Features(
            feature=features))
    writer.write(example.SerializeToString())
  writer.close()


def _train_sentencepiece(input_path, vocab_size, model_path, eos_id=1):
  argstr = ' '.join([
      f'--input={input_path}', f'--vocab_size={vocab_size}',
      '--character_coverage=0.995',
      f'--model_prefix={model_path}', '--model_type=bpe',
      '--bos_id=-1', '--pad_id=0', f'--eos_id={eos_id}', '--unk_id=2'
  ])
  SentencePieceTrainer.Train(argstr)


class WMTDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(WMTDataLoaderTest, self).setUp()
    self._temp_dir = self.get_temp_dir()
    src_lines = [
        'abc ede fg',
        'bbcd ef a g',
        'de f a a g'
    ]
    tgt_lines = [
        'dd cc a ef  g',
        'bcd ef a g',
        'gef cd ba'
    ]
    self._record_train_input_path = os.path.join(self._temp_dir, 'train.record')
    _generate_record_file(self._record_train_input_path, src_lines, tgt_lines)
    self._record_test_input_path = os.path.join(self._temp_dir, 'test.record')
    _generate_record_file(self._record_test_input_path, src_lines, tgt_lines,
                          unique_id=True)
    self._sentencepeice_input_path = os.path.join(self._temp_dir, 'inputs.txt')
    _generate_line_file(self._sentencepeice_input_path, src_lines + tgt_lines)
    sentencepeice_model_prefix = os.path.join(self._temp_dir, 'sp')
    _train_sentencepiece(self._sentencepeice_input_path, 20,
                         sentencepeice_model_prefix)
    self._sentencepeice_model_path = '{}.model'.format(
        sentencepeice_model_prefix)

  @parameterized.named_parameters(
      ('train_static', True, True, 100, (2, 35)),
      ('train_non_static', True, False, 100, (12, 7)),
      ('non_train_static', False, True, 3, (3, 35)),
      ('non_train_non_static', False, False, 50, (2, 7)),)
  def test_load_dataset(
      self, is_training, static_batch, batch_size, expected_shape):
    data_config = wmt_dataloader.WMTDataConfig(
        input_path=self._record_train_input_path
        if is_training else self._record_test_input_path,
        max_seq_length=35,
        global_batch_size=batch_size,
        is_training=is_training,
        static_batch=static_batch,
        src_lang='en',
        tgt_lang='reverse_en',
        sentencepiece_model_path=self._sentencepeice_model_path)
    dataset = wmt_dataloader.WMTDataLoader(data_config).load()
    examples = next(iter(dataset))
    inputs, targets = examples['inputs'], examples['targets']
    self.assertEqual(inputs.shape, expected_shape)
    self.assertEqual(targets.shape, expected_shape)

  def test_load_dataset_raise_invalid_window(self):
    batch_tokens_size = 10  # this is too small to form buckets.
    data_config = wmt_dataloader.WMTDataConfig(
        input_path=self._record_train_input_path,
        max_seq_length=100,
        global_batch_size=batch_tokens_size,
        is_training=True,
        static_batch=False,
        src_lang='en',
        tgt_lang='reverse_en',
        sentencepiece_model_path=self._sentencepeice_model_path)
    with self.assertRaisesRegex(
        ValueError, 'The token budget, global batch size, is too small.*'):
      _ = wmt_dataloader.WMTDataLoader(data_config).load()


if __name__ == '__main__':
  tf.test.main()
