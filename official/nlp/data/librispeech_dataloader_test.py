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

"""Tests for loader."""

import tempfile

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_datasets as tfds

from official.nlp.data import librispeech_dataloader


class LoaderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    vocab_tokens = [
        '[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn',
        '##ing', ','
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      vocab_writer.write(''.join([x + '\n' for x in vocab_tokens
                                 ]).encode('utf-8'))
    vocab_file = vocab_writer.name
    self.vocab_file = vocab_file

  def test_data_loading(self):
    data_config = librispeech_dataloader.LibriSpeechDataConfig(
        tfds_name='librispeech',
        tfds_split='dev_clean',
        global_batch_size=4,
        is_training=True,
        frame_length=800,
        fft_length=800,
        frame_step=400,
        sample_rate=16000,
        num_mel_bins=128,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        audio_samples=160000,
        vocab_file=self.vocab_file,
    )
    logging.info('data_config: %s', data_config)
    with tfds.testing.mock_data(num_examples=5):
      dataset = librispeech_dataloader.LibriSpeechDataLoader(data_config).load()
      features = next(iter(dataset))
      logging.info('features: %s', features)
      self.assertSequenceEqual(features['audio'].shape,
                               [4, data_config.audio_samples])
      self.assertEqual(features['audio'].dtype, tf.float32)
      self.assertSequenceEqual(features['tokens'].shape,
                               [4, data_config.seq_lengths])
      self.assertSequenceEqual(features['mask'].shape,
                               [4, data_config.seq_lengths])
      self.assertEqual(features['tokens'].dtype, tf.int32)
      mfcc_shape = (data_config.global_batch_size,
                    data_config.audio_samples / data_config.frame_step,
                    13)
      self.assertSequenceEqual(features['mfcc'].shape, mfcc_shape)
      log_mel_shape = (data_config.global_batch_size,
                       data_config.audio_samples / data_config.frame_step,
                       data_config.num_mel_bins)
      self.assertSequenceEqual(features['log_mel'].shape, log_mel_shape)


if __name__ == '__main__':
  tf.test.main()
