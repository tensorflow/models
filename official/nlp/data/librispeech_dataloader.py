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

r"""Loads dataset from raw text for the unified model bert task.

TODO(dalejohnson): Consider use of dmvr/processors.py for more generalized
capabilities.

See:
https://github.com/tensorflow/tensorflow/issues/16465#issuecomment-396494851
for a discussion on variance between tf.signal MFCC calculations and those of
librosa.

https://stackoverflow.com/questions/60492462/mfcc-python-completely-different-result-from-librosa-vs-python-speech-features
for some discussion on hyperparameter tuning in MFCC.
"""

import dataclasses
from typing import Dict, Mapping, List, Optional, Tuple

from absl import logging
import numpy as np
import tensorflow.google as tf
import tensorflow_text as tf_text

from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

_SCALE_FACTOR = 32768.0


@dataclasses.dataclass
class LibriSpeechDataConfig(cfg.DataConfig):
  """A data config for LibriSpeech dataset."""

  model = None
  global_batch_size: int = 32
  drop_remainder: bool = True
  loader_debug: bool = True
  is_training: bool = True

  input_path: str = ''
  tfds_name: str = 'librispeech'  # use: librispeech
  tfds_split: str = ''  # try: dev_clean

  # MFCC and Mel settings
  frame_length: int = 800   # 50ms @ 16kHz
  fft_length: int = 800     # 50ms @ 16kHz
  frame_step: int = 400     # 25ms @ 16kHz
  sample_rate: int = 16000  # 16kHz
  num_mel_bins: int = 128
  lower_edge_hertz: float = 125.0
  upper_edge_hertz: float = 7500.0

  audio_samples: int = 160000  # about 10 seconds

  seq_lengths: int = 512
  vocab_file: str = ''


@data_loader_factory.register_data_loader_cls(LibriSpeechDataConfig)
class LibriSpeechDataLoader(data_loader.DataLoader):
  """Loads dataset (already tokenized) and pass it along to the model."""

  def __init__(self, params: LibriSpeechDataConfig):
    self._params = params
    logging.info('params: %s', params)

  def _transform_and_batch(self, dataset: tf.data.Dataset,
                           input_context) -> Dict[str, tf.Tensor]:
    tokenizer = tf_text.WordpieceTokenizer(
        self._params.vocab_file, token_out_type=tf.int32)

    def _tokenize_and_pad(example_text: List[str],
                          seq_length: int) -> tf.Tensor:
      """Invoke tensorflow wordpiece tokenization."""

      # Normalize text.
      example_text = tf_text.normalize_utf8(example_text)
      example_text = tf_text.case_fold_utf8(example_text)

      # Tokenize into words.
      word_tokenizer = tf_text.WhitespaceTokenizer()
      tokens = word_tokenizer.tokenize(example_text)

      # Tokenize into subwords.
      subtokens = tokenizer.tokenize(tokens).merge_dims(1, -1)

      # Apply padding.
      return tf_text.pad_model_inputs(subtokens, seq_length)

    def _spectrogram(pcm: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Calculates mel spectogram and MFCC."""
      # Time to frequency domain.
      stfts = tf.signal.stft(
          pcm,
          frame_length=self._params.frame_length,
          frame_step=self._params.frame_step,
          fft_length=self._params.fft_length,
          window_fn=tf.signal.hann_window,
          pad_end=True)
      spectrograms = tf.abs(stfts)

      # We can't get these shapes when tracing, but they reflect
      # spectrogram.shape.
      downsampled = int(
          np.ceil(self._params.audio_samples / self._params.frame_step))
      logging.info('downsampled: %d', downsampled)
      num_spectrogram_bins = self._params.fft_length // 2 + 1
      logging.info('num_spectrogram_bins: %d', num_spectrogram_bins)

      # To log frequency domain.
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=self._params.num_mel_bins,
          num_spectrogram_bins=num_spectrogram_bins,
          sample_rate=self._params.sample_rate,
          lower_edge_hertz=self._params.lower_edge_hertz,
          upper_edge_hertz=self._params.upper_edge_hertz)
      mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix,
                                      1)
      mel_spectrograms.set_shape((downsampled, self._params.num_mel_bins))

      # To log frequency domain with log magnitude.
      log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

      # Extract MFCC (Mel frequency capstral coefficients)
      mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[
          ..., :13]
      return mfccs, log_mel_spectrograms

    def _map(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      # deal with the audio
      paddings = [[0, self._params.audio_samples]]
      padded = tf.pad(example['speech'], paddings)
      audio = tf.RaggedTensor.from_tensor(tf.stack([padded]))
      audio, _ = tf_text.pad_model_inputs(audio, self._params.audio_samples)
      audio = tf.squeeze(audio)
      audio = tf.cast(audio, tf.float32) / _SCALE_FACTOR

      # tokenize the text
      text = example['text']
      tokens = _tokenize_and_pad([text], self._params.seq_lengths)
      tokens = tf.reshape(tokens, (2, self._params.seq_lengths))
      token_ids = tokens[0]
      mask_id = tokens[1]

      # audio features
      logging.info('audio: %s', audio)
      tf.print('audio', audio)
      mfcc, log_mel = _spectrogram(audio)

      return {
          'audio': audio,
          'text': token_ids,
          'mask': mask_id,
          'mfcc': mfcc,
          'log_mel': log_mel
      }

    dataset = dataset.map(
        _map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batch_size = self._params.global_batch_size
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        batch_size) if input_context else batch_size
    dataset = dataset.batch(
        per_replica_batch_size, drop_remainder=self._params.drop_remainder)
    return dataset

  def _postprocess(self, record: Mapping[str,
                                         tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Adjust a single batch."""
    model_inputs = {
        'audio': record['audio'],
        'tokens': tf.cast(record['text'], tf.int32),
        'mask': tf.cast(record['mask'], tf.int32),
        'mfcc': record['mfcc'],
        'log_mel': record['log_mel'],
    }
    if self._params.loader_debug:
      logging.info('final model inputs: %s', model_inputs)
    return model_inputs

  def load(self,
           input_context: Optional[tf.distribute.InputContext] = None) -> ...:
    reader = input_reader.InputReader(
        params=self._params,
        decoder_fn=self._decode if self._params.input_path else None,
        dataset_fn=tf.data.TFRecordDataset,
        transform_and_batch_fn=self._transform_and_batch,
        postprocess_fn=self._postprocess)
    return reader.read(input_context)
