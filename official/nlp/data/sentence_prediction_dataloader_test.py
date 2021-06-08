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

"""Tests for official.nlp.data.sentence_prediction_dataloader."""
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from sentencepiece import SentencePieceTrainer
from official.nlp.data import sentence_prediction_dataloader as loader


def _create_fake_preprocessed_dataset(output_path, seq_length, label_type):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  for _ in range(100):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features['input_ids'] = create_int_feature(input_ids)
    features['input_mask'] = create_int_feature(np.ones_like(input_ids))
    features['segment_ids'] = create_int_feature(np.ones_like(input_ids))

    if label_type == 'int':
      features['label_ids'] = create_int_feature([1])
    elif label_type == 'float':
      features['label_ids'] = create_float_feature([0.5])
    else:
      raise ValueError('Unsupported label_type: %s' % label_type)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _create_fake_raw_dataset(output_path, text_fields, label_type):
  """Creates a fake tf record file."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_str_feature(value):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    return f

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  for _ in range(100):
    features = {}
    for text_field in text_fields:
      features[text_field] = create_str_feature([b'hello world'])

    if label_type == 'int':
      features['label'] = create_int_feature([0])
    elif label_type == 'float':
      features['label'] = create_float_feature([0.5])
    else:
      raise ValueError('Unexpected label_type: %s' % label_type)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _create_fake_sentencepiece_model(output_dir):
  vocab = ['a', 'b', 'c', 'd', 'e', 'abc', 'def', 'ABC', 'DEF']
  model_prefix = os.path.join(output_dir, 'spm_model')
  input_text_file_path = os.path.join(output_dir, 'train_input.txt')
  with tf.io.gfile.GFile(input_text_file_path, 'w') as f:
    f.write(' '.join(vocab + ['\n']))
  # Add 7 more tokens: <pad>, <unk>, [CLS], [SEP], [MASK], <s>, </s>.
  full_vocab_size = len(vocab) + 7
  flags = dict(
      model_prefix=model_prefix,
      model_type='word',
      input=input_text_file_path,
      pad_id=0,
      unk_id=1,
      control_symbols='[CLS],[SEP],[MASK]',
      vocab_size=full_vocab_size,
      bos_id=full_vocab_size - 2,
      eos_id=full_vocab_size - 1)
  SentencePieceTrainer.Train(' '.join(
      ['--{}={}'.format(k, v) for k, v in flags.items()]))
  return model_prefix + '.model'


def _create_fake_vocab_file(vocab_file_path):
  tokens = ['[PAD]']
  for i in range(1, 100):
    tokens.append('[unused%d]' % i)
  tokens.extend(['[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world'])
  with tf.io.gfile.GFile(vocab_file_path, 'w') as outfile:
    outfile.write('\n'.join(tokens))


class SentencePredictionDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('int', tf.int32), ('float', tf.float32))
  def test_load_dataset(self, label_type, expected_label_type):
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    batch_size = 10
    seq_length = 128
    _create_fake_preprocessed_dataset(input_path, seq_length, label_type)
    data_config = loader.SentencePredictionDataConfig(
        input_path=input_path,
        seq_length=seq_length,
        global_batch_size=batch_size,
        label_type=label_type)
    dataset = loader.SentencePredictionDataLoader(data_config).load()
    features, labels = next(iter(dataset))
    self.assertCountEqual(['input_word_ids', 'input_mask', 'input_type_ids'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(labels.shape, (batch_size,))
    self.assertEqual(labels.dtype, expected_label_type)

  def test_load_dataset_as_dict(self):
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    batch_size = 10
    seq_length = 128
    _create_fake_preprocessed_dataset(input_path, seq_length, 'int')
    data_config = loader.SentencePredictionDataConfig(
        input_path=input_path,
        seq_length=seq_length,
        global_batch_size=batch_size,
        label_type='int',
        outputs_as_dict=True)
    dataset = loader.SentencePredictionDataLoader(data_config).load()
    features = next(iter(dataset))
    self.assertCountEqual([
        'input_word_ids', 'input_mask', 'input_type_ids', 'next_sentence_labels'
    ], features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['next_sentence_labels'].shape, (batch_size,))
    self.assertEqual(features['next_sentence_labels'].dtype, tf.int32)


class SentencePredictionTfdsDataLoaderTest(tf.test.TestCase,
                                           parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_python_wordpiece_preprocessing(self, use_tfds):
    batch_size = 10
    seq_length = 256  # Non-default value.
    lower_case = True

    tf_record_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    text_fields = ['sentence1', 'sentence2']
    if not use_tfds:
      _create_fake_raw_dataset(tf_record_path, text_fields, label_type='int')

    vocab_file_path = os.path.join(self.get_temp_dir(), 'vocab.txt')
    _create_fake_vocab_file(vocab_file_path)

    data_config = loader.SentencePredictionTextDataConfig(
        input_path='' if use_tfds else tf_record_path,
        tfds_name='glue/mrpc' if use_tfds else '',
        tfds_split='train' if use_tfds else '',
        text_fields=text_fields,
        global_batch_size=batch_size,
        seq_length=seq_length,
        is_training=True,
        lower_case=lower_case,
        vocab_file=vocab_file_path)
    dataset = loader.SentencePredictionTextDataLoader(data_config).load()
    features, labels = next(iter(dataset))
    self.assertCountEqual(['input_word_ids', 'input_type_ids', 'input_mask'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(labels.shape, (batch_size,))

  @parameterized.parameters(True, False)
  def test_python_sentencepiece_preprocessing(self, use_tfds):
    batch_size = 10
    seq_length = 256  # Non-default value.
    lower_case = True

    tf_record_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    text_fields = ['sentence1', 'sentence2']
    if not use_tfds:
      _create_fake_raw_dataset(tf_record_path, text_fields, label_type='int')

    sp_model_file_path = _create_fake_sentencepiece_model(self.get_temp_dir())
    data_config = loader.SentencePredictionTextDataConfig(
        input_path='' if use_tfds else tf_record_path,
        tfds_name='glue/mrpc' if use_tfds else '',
        tfds_split='train' if use_tfds else '',
        text_fields=text_fields,
        global_batch_size=batch_size,
        seq_length=seq_length,
        is_training=True,
        lower_case=lower_case,
        tokenization='SentencePiece',
        vocab_file=sp_model_file_path,
    )
    dataset = loader.SentencePredictionTextDataLoader(data_config).load()
    features, labels = next(iter(dataset))
    self.assertCountEqual(['input_word_ids', 'input_type_ids', 'input_mask'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(labels.shape, (batch_size,))

  @parameterized.parameters(True, False)
  def test_saved_model_preprocessing(self, use_tfds):
    batch_size = 10
    seq_length = 256  # Non-default value.

    tf_record_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    text_fields = ['sentence1', 'sentence2']
    if not use_tfds:
      _create_fake_raw_dataset(tf_record_path, text_fields, label_type='float')

    vocab_file_path = os.path.join(self.get_temp_dir(), 'vocab.txt')
    _create_fake_vocab_file(vocab_file_path)
    data_config = loader.SentencePredictionTextDataConfig(
        input_path='' if use_tfds else tf_record_path,
        tfds_name='glue/mrpc' if use_tfds else '',
        tfds_split='train' if use_tfds else '',
        text_fields=text_fields,
        global_batch_size=batch_size,
        seq_length=seq_length,
        is_training=True,
        preprocessing_hub_module_url=(
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'),
        label_type='int' if use_tfds else 'float',
    )
    dataset = loader.SentencePredictionTextDataLoader(data_config).load()
    features, labels = next(iter(dataset))
    self.assertCountEqual(['input_word_ids', 'input_type_ids', 'input_mask'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_mask'].shape, (batch_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, seq_length))
    self.assertEqual(labels.shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
