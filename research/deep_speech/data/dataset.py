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

import numpy as np
import scipy.io.wavfile as wavfile
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# pylint: disable=g-bad-import-order
from data.featurizer import AudioFeaturizer
from data.featurizer import TextFeaturizer


class AudioConfig(object):
  """Configs for spectrogram extraction from audio."""

  def __init__(self,
               sample_rate,
               frame_length,
               frame_step,
               fft_length=None,
               normalize=False,
               spect_type="linear"):
    """Initialize the AudioConfig class.

    Args:
      sample_rate: an integer denoting the sample rate of the input waveform.
      frame_length: an integer for the length of a spectrogram frame, in ms.
      frame_step: an integer for the frame stride, in ms.
      fft_length: an integer for the number of fft bins.
      normalize: a boolean for whether apply normalization on the audio tensor.
      spect_type: a string for the type of spectrogram to be extracted.
    """

    self.sample_rate = sample_rate
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.normalize = normalize
    self.spect_type = spect_type


class DatasetConfig(object):
  """Config class for generating the DeepSpeechDataset."""

  def __init__(self, audio_config, data_path, vocab_file_path):
    """Initialize the configs for deep speech dataset.

    Args:
      audio_config: AudioConfig object specifying the audio-related configs.
      data_path: a string denoting the full path of a manifest file.
      vocab_file_path: a string specifying the vocabulary file path.

    Raises:
      RuntimeError: file path not exist.
    """

    self.audio_config = audio_config
    assert tf.gfile.Exists(data_path)
    assert tf.gfile.Exists(vocab_file_path)
    self.data_path = data_path
    self.vocab_file_path = vocab_file_path


class DeepSpeechDataset(object):
  """Dataset class for training/evaluation of DeepSpeech model."""

  def __init__(self, dataset_config):
    """Initialize the class.

    Each dataset file contains three columns: "wav_filename", "wav_filesize",
    and "transcript". This function parses the csv file and stores each example
    by the increasing order of audio length (indicated by wav_filesize).

    Args:
      dataset_config: DatasetConfig object.
    """
    self.config = dataset_config
    # Instantiate audio feature extractor.
    self.audio_featurizer = AudioFeaturizer(
        sample_rate=self.config.audio_config.sample_rate,
        frame_length=self.config.audio_config.frame_length,
        frame_step=self.config.audio_config.frame_step,
        fft_length=self.config.audio_config.fft_length,
        spect_type=self.config.audio_config.spect_type)
    # Instantiate text feature extractor.
    self.text_featurizer = TextFeaturizer(
        vocab_file=self.config.vocab_file_path)

    self.speech_labels = self.text_featurizer.speech_labels
    self.features, self.labels = self._preprocess_data(self.config.data_path)
    self.num_feature_bins = (
        self.features[0].shape[1] if len(self.features) else None)

  def _preprocess_data(self, file_path):
    """Generate a list of waveform, transcript pair.

    Note that the waveforms are ordered in increasing length, so that audio
    samples in a mini-batch have similar length.

    Args:
      file_path: a string specifying the csv file path for a data set.

    Returns:
      features and labels array processed from the audio/text input.
    """

    with tf.gfile.Open(file_path, "r") as f:
      lines = f.read().splitlines()
    lines = [line.split("\t") for line in lines]
    # Skip the csv header.
    lines = lines[1:]
    # Sort input data by the length of waveform.
    lines.sort(key=lambda item: int(item[1]))
    features = [self._preprocess_audio(line[0]) for line in lines]
    labels = [self._preprocess_transcript(line[2]) for line in lines]
    return features, labels

  def _normalize_audio_tensor(self, audio_tensor):
    """Perform mean and variance normalization on the spectrogram tensor.

    Args:
      audio_tensor: a tensor for the spectrogram feature.

    Returns:
      a tensor for the normalized spectrogram.
    """
    mean, var = tf.nn.moments(audio_tensor, axes=[0])
    normalized = (audio_tensor - mean) / (tf.sqrt(var) + 1e-6)
    return normalized

  def _preprocess_audio(self, audio_file_path):
    """Load the audio file in memory."""
    tf.logging.info(
        "Extracting spectrogram feature for {}".format(audio_file_path))
    sample_rate, data = wavfile.read(audio_file_path)
    assert sample_rate == self.config.audio_config.sample_rate
    if data.dtype not in [np.float32, np.float64]:
      data = data.astype(np.float32) / np.iinfo(data.dtype).max
    feature = self.audio_featurizer.featurize(data)
    if self.config.audio_config.normalize:
      feature = self._normalize_audio_tensor(feature)
    return tf.Session().run(
        feature)  # return a numpy array rather than a tensor

  def _preprocess_transcript(self, transcript):
    return self.text_featurizer.featurize(transcript)


def input_fn(batch_size, deep_speech_dataset, repeat=1):
  """Input function for model training and evaluation.

  Args:
    batch_size: an integer denoting the size of a batch.
    deep_speech_dataset: DeepSpeechDataset object.
    repeat: an integer for how many times to repeat the dataset.

  Returns:
    a tf.data.Dataset object for model to consume.
  """
  features = deep_speech_dataset.features
  labels = deep_speech_dataset.labels
  num_feature_bins = deep_speech_dataset.num_feature_bins

  def _gen_data():
    for i in xrange(len(features)):
      feature = np.expand_dims(features[i], axis=2)
      input_length = [features[i].shape[0]]
      label_length = [len(labels[i])]
      yield {
          "features": feature,
          "labels": labels[i],
          "input_length": input_length,
          "label_length": label_length
      }

  dataset = tf.data.Dataset.from_generator(
      _gen_data,
      output_types={
          "features": tf.float32,
          "labels": tf.int32,
          "input_length": tf.int32,
          "label_length": tf.int32
      },
      output_shapes={
          "features": tf.TensorShape([None, num_feature_bins, 1]),
          "labels": tf.TensorShape([None]),
          "input_length": tf.TensorShape([1]),
          "label_length": tf.TensorShape([1])
      })

  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)
  # Padding the features to its max length dimensions.
  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes={
          "features": tf.TensorShape([None, num_feature_bins, 1]),
          "labels": tf.TensorShape([None]),
          "input_length": tf.TensorShape([1]),
          "label_length": tf.TensorShape([1])
      })

  # Prefetch to improve speed of input pipeline.
  dataset = dataset.prefetch(1)
  return dataset
