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
from scipy import signal


def compute_spectrogram_feature(waveform, frame_length, frame_step, fft_length):
  """Compute the spectrograms for the input waveform."""
  _, _, stft = signal.stft(
      waveform,
      nperseg=frame_length,
      noverlap=frame_step,
      nfft=fft_length)

  # Perform transpose to set its shape as [time_steps, feature_num_bins]
  spectrogram = np.transpose(np.absolute(stft), (1, 0))
  return spectrogram


class AudioFeaturizer(object):
  """Class to extract spectrogram features from the audio input."""

  def __init__(self,
               sample_rate=16000,
               frame_length=25,
               frame_step=10,
               fft_length=None):
    """Initialize the audio featurizer class according to the configs.

    Args:
      sample_rate: an integer specifying the sample rate of the input waveform.
      frame_length: an integer for the length of a spectrogram frame, in ms.
      frame_step: an integer for the frame stride, in ms.
      fft_length: an integer for the number of fft bins.
    """
    self.frame_length = int(sample_rate * frame_length / 1e3)
    self.frame_step = int(sample_rate * frame_step / 1e3)
    self.fft_length = fft_length if fft_length else int(2**(np.ceil(
        np.log2(self.frame_length))))


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
    self.token_to_idx = {}
    self.idx_to_token = {}
    self.speech_labels = ""
    idx = 0
    for line in lines:
      line = line[:-1]  # Strip the '\n' char.
      if line.startswith("#"):
        # Skip from reading comment line.
        continue
      self.token_to_idx[line] = idx
      self.idx_to_token[idx] = line
      self.speech_labels += line
      idx += 1
