#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""Generate tf.data.Dataset object for deep speech training/evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
# pylint: disable=g-bad-import-order
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import soundfile
import tensorflow as tf
# pylint: enable=g-bad-import-order

import data.featurizer as featurizer  # pylint: disable=g-bad-import-order


class AudioConfig(object):
  """Configs for spectrogram extraction from audio."""

  def __init__(self,
               sample_rate,
               window_ms,
               stride_ms,
               normalize=False):
    """Initialize the AudioConfig class.

    Args:
      sample_rate: an integer denoting the sample rate of the input waveform.
      window_ms: an integer for the length of a spectrogram frame, in ms.
      stride_ms: an integer for the frame stride, in ms.
      normalize: a boolean for whether apply normalization on the audio feature.
    """

    self.sample_rate = sample_rate
    self.window_ms = window_ms
    self.stride_ms = stride_ms
    self.normalize = normalize


class DatasetConfig(object):
  """Config class for generating the DeepSpeechDataset."""

  def __init__(self, audio_config, data_path, vocab_file_path, sortagrad):
    """Initialize the configs for deep speech dataset.

    Args:
      audio_config: AudioConfig object specifying the audio-related configs.
      data_path: a string denoting the full path of a manifest file.
      vocab_file_path: a string specifying the vocabulary file path.
      sortagrad: a boolean, if set to true, audio sequences will be fed by
                increasing length in the first training epoch, which will
                expedite network convergence.

    Raises:
      RuntimeError: file path not exist.
    """

    self.audio_config = audio_config
    assert tf.gfile.Exists(data_path)
    assert tf.gfile.Exists(vocab_file_path)
    self.data_path = data_path
    self.vocab_file_path = vocab_file_path
    self.sortagrad = sortagrad


def _normalize_audio_feature(audio_feature):
  """Perform mean and variance normalization on the spectrogram feature.

  Args:
    audio_feature: a numpy array for the spectrogram feature.

  Returns:
    a numpy array of the normalized spectrogram.
  """
  mean = np.mean(audio_feature, axis=0)
  var = np.var(audio_feature, axis=0)
  normalized = (audio_feature - mean) / (np.sqrt(var) + 1e-6)

  return normalized


def _preprocess_audio(audio_file_path, audio_featurizer, normalize):
  """Load the audio file and compute spectrogram feature."""
  data, _ = soundfile.read(audio_file_path)
  feature = featurizer.compute_spectrogram_feature(
      data, audio_featurizer.sample_rate, audio_featurizer.stride_ms,
      audio_featurizer.window_ms)
  # Feature normalization
  if normalize:
    feature = _normalize_audio_feature(feature)

  # Adding Channel dimension for conv2D input.
  feature = np.expand_dims(feature, axis=2)
  return feature


def _preprocess_data(file_path):
  """Generate a list of tuples (wav_filename, wav_filesize, transcript).

  Each dataset file contains three columns: "wav_filename", "wav_filesize",
  and "transcript". This function parses the csv file and stores each example
  by the increasing order of audio length (indicated by wav_filesize).
  AS the waveforms are ordered in increasing length, audio samples in a
  mini-batch have similar length.

  Args:
    file_path: a string specifying the csv file path for a dataset.

  Returns:
    A list of tuples (wav_filename, wav_filesize, transcript) sorted by
    file_size.
  """
  tf.logging.info("Loading data set {}".format(file_path))
  with tf.gfile.Open(file_path, "r") as f:
    lines = f.read().splitlines()
  # Skip the csv header in lines[0].
  lines = lines[1:]
  # The metadata file is tab separated.
  lines = [line.split("\t", 2) for line in lines]
  # Sort input data by the length of audio sequence.
  lines.sort(key=lambda item: int(item[1]))

  return [tuple(line) for line in lines]


class DeepSpeechDataset(object):
  """Dataset class for training/evaluation of DeepSpeech model."""

  def __init__(self, dataset_config):
    """Initialize the DeepSpeechDataset class.

    Args:
      dataset_config: DatasetConfig object.
    """
    self.config = dataset_config
    # Instantiate audio feature extractor.
    self.audio_featurizer = featurizer.AudioFeaturizer(
        sample_rate=self.config.audio_config.sample_rate,
        window_ms=self.config.audio_config.window_ms,
        stride_ms=self.config.audio_config.stride_ms)
    # Instantiate text feature extractor.
    self.text_featurizer = featurizer.TextFeaturizer(
        vocab_file=self.config.vocab_file_path)

    self.speech_labels = self.text_featurizer.speech_labels
    self.entries = _preprocess_data(self.config.data_path)
    # The generated spectrogram will have 161 feature bins.
    self.num_feature_bins = 161


def batch_wise_dataset_shuffle(entries, epoch_index, sortagrad, batch_size):
  """Batch-wise shuffling of the data entries.

  Each data entry is in the format of (audio_file, file_size, transcript).
  If epoch_index is 0 and sortagrad is true, we don't perform shuffling and
  return entries in sorted file_size order. Otherwise, do batch_wise shuffling.

  Args:
    entries: a list of data entries.
    epoch_index: an integer of epoch index
    sortagrad: a boolean to control whether sorting the audio in the first
      training epoch.
    batch_size: an integer for the batch size.

  Returns:
    The shuffled data entries.
  """
  shuffled_entries = []
  if epoch_index == 0 and sortagrad:
    # No need to shuffle.
    shuffled_entries = entries
  else:
    # Shuffle entries batch-wise.
    max_buckets = int(math.floor(len(entries) / batch_size))
    total_buckets = [i for i in xrange(max_buckets)]
    random.shuffle(total_buckets)
    shuffled_entries = []
    for i in total_buckets:
      shuffled_entries.extend(entries[i * batch_size : (i + 1) * batch_size])
    # If the last batch doesn't contain enough batch_size examples,
    # just append it to the shuffled_entries.
    shuffled_entries.extend(entries[max_buckets * batch_size:])

  return shuffled_entries


def input_fn(batch_size, deep_speech_dataset, repeat=1):
  """Input function for model training and evaluation.

  Args:
    batch_size: an integer denoting the size of a batch.
    deep_speech_dataset: DeepSpeechDataset object.
    repeat: an integer for how many times to repeat the dataset.

  Returns:
    a tf.data.Dataset object for model to consume.
  """
  # Dataset properties
  data_entries = deep_speech_dataset.entries
  num_feature_bins = deep_speech_dataset.num_feature_bins
  audio_featurizer = deep_speech_dataset.audio_featurizer
  feature_normalize = deep_speech_dataset.config.audio_config.normalize
  text_featurizer = deep_speech_dataset.text_featurizer

  def _gen_data():
    """Dataset generator function."""
    for audio_file, _, transcript in data_entries:
      features = _preprocess_audio(
          audio_file, audio_featurizer, feature_normalize)
      labels = featurizer.compute_label_feature(
          transcript, text_featurizer.token_to_index)
      input_length = [features.shape[0]]
      label_length = [len(labels)]
      # Yield a tuple of (features, labels) where features is a dict containing
      # all info about the actual data features.
      yield (
          {
              "features": features,
              "input_length": input_length,
              "label_length": label_length
          },
          labels)

  dataset = tf.data.Dataset.from_generator(
      _gen_data,
      output_types=(
          {
              "features": tf.float32,
              "input_length": tf.int32,
              "label_length": tf.int32
          },
          tf.int32),
      output_shapes=(
          {
              "features": tf.TensorShape([None, num_feature_bins, 1]),
              "input_length": tf.TensorShape([1]),
              "label_length": tf.TensorShape([1])
          },
          tf.TensorShape([None]))
  )

  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)

  # Padding the features to its max length dimensions.
  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes=(
          {
              "features": tf.TensorShape([None, num_feature_bins, 1]),
              "input_length": tf.TensorShape([1]),
              "label_length": tf.TensorShape([1])
          },
          tf.TensorShape([None]))
  )

  # Prefetch to improve speed of input pipeline.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset

