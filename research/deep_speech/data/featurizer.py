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
"""Utility class for extracting features from the text and audio input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import numpy as np


def compute_spectrogram_feature(samples, sample_rate, stride_ms=10.0,
                                window_ms=20.0, max_freq=None, eps=1e-14):
  """Compute the spectrograms for the input samples(waveforms).

  More about spectrogram computation, please refer to:
  https://en.wikipedia.org/wiki/Short-time_Fourier_transform.
  """
  if max_freq is None:
    max_freq = sample_rate / 2
  if max_freq > sample_rate / 2:
    raise ValueError("max_freq must not be greater than half of sample rate.")

  if stride_ms > window_ms:
    raise ValueError("Stride size must not be greater than window size.")

  stride_size = int(0.001 * sample_rate * stride_ms)
  window_size = int(0.001 * sample_rate * window_ms)

  # Extract strided windows
  truncate_size = (len(samples) - window_size) % stride_size
  samples = samples[:len(samples) - truncate_size]
  nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
  nstrides = (samples.strides[0], samples.strides[0] * stride_size)
  windows = np.lib.stride_tricks.as_strided(
      samples, shape=nshape, strides=nstrides)
  assert np.all(
      windows[:, 1] == samples[stride_size:(stride_size + window_size)])

  # Window weighting, squared Fast Fourier Transform (fft), scaling
  weighting = np.hanning(window_size)[:, None]
  fft = np.fft.rfft(windows * weighting, axis=0)
  fft = np.absolute(fft)
  fft = fft**2
  scale = np.sum(weighting**2) * sample_rate
  fft[1:-1, :] *= (2.0 / scale)
  fft[(0, -1), :] /= scale
  # Prepare fft frequency list
  freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

  # Compute spectrogram feature
  ind = np.where(freqs <= max_freq)[0][-1] + 1
  specgram = np.log(fft[:ind, :] + eps)
  return np.transpose(specgram, (1, 0))


class AudioFeaturizer(object):
  """Class to extract spectrogram features from the audio input."""

  def __init__(self,
               sample_rate=16000,
               window_ms=20.0,
               stride_ms=10.0):
    """Initialize the audio featurizer class according to the configs.

    Args:
      sample_rate: an integer specifying the sample rate of the input waveform.
      window_ms: an integer for the length of a spectrogram frame, in ms.
      stride_ms: an integer for the frame stride, in ms.
    """
    self.sample_rate = sample_rate
    self.window_ms = window_ms
    self.stride_ms = stride_ms


def compute_label_feature(text, token_to_idx):
  """Convert string to a list of integers."""
  tokens = list(text.strip().lower())
  feats = [token_to_idx[token] for token in tokens]
  return feats


class TextFeaturizer(object):
  """Extract text feature based on char-level granularity.

  By looking up the vocabulary table, each input string (one line of transcript)
  will be converted to a sequence of integer indexes.
  """

  def __init__(self, vocab_file):
    lines = []
    with codecs.open(vocab_file, "r", "utf-8") as fin:
      lines.extend(fin.readlines())
    self.token_to_index = {}
    self.index_to_token = {}
    self.speech_labels = ""
    index = 0
    for line in lines:
      line = line[:-1]  # Strip the '\n' char.
      if line.startswith("#"):
        # Skip from reading comment line.
        continue
      self.token_to_index[line] = index
      self.index_to_token[index] = line
      self.speech_labels += line
      index += 1
