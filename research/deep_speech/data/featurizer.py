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
import functools
import numpy as np
import tensorflow as tf


class AudioFeaturizer(object):
  """Class to extract spectrogram features from the audio input."""

  def __init__(self,
               sample_rate=16000,
               frame_length=25,
               frame_step=10,
               fft_length=None,
               window_fn=functools.partial(
                   tf.contrib.signal.hann_window, periodic=True),
               spect_type="linear"):
    """Initialize the audio featurizer class according to the configs.

    Args:
      sample_rate: an integer specifying the sample rate of the input waveform.
      frame_length: an integer for the length of a spectrogram frame, in ms.
      frame_step: an integer for the frame stride, in ms.
      fft_length: an integer for the number of fft bins.
      window_fn: windowing function.
      spect_type: a string for the type of spectrogram to be extracted.
      Currently only support 'linear', otherwise will raise a value error.

    Raises:
      ValueError: In case of invalid arguments for `spect_type`.
    """
    if spect_type != "linear":
      raise ValueError("Unsupported spectrogram type: %s" % spect_type)
    self.window_fn = window_fn
    self.frame_length = int(sample_rate * frame_length / 1e3)
    self.frame_step = int(sample_rate * frame_step / 1e3)
    self.fft_length = fft_length if fft_length else int(2**(np.ceil(
        np.log2(self.frame_length))))

  def featurize(self, waveform):
    """Extract spectrogram feature tensors from the waveform."""
    return self._compute_linear_spectrogram(waveform)

  def _compute_linear_spectrogram(self, waveform):
    """Compute the linear-scale, magnitude spectrograms for the input waveform.

    Args:
      waveform: a float32 audio tensor.
    Returns:
      a float 32 tensor with shape [len, num_bins]
    """

    # `stfts` is a complex64 Tensor representing the Short-time Fourier
    # Transform of each signal in `signals`. Its shape is
    # [?, fft_unique_bins] where fft_unique_bins = fft_length // 2 + 1.
    stfts = tf.contrib.signal.stft(
        waveform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        fft_length=self.fft_length,
        window_fn=self.window_fn,
        pad_end=True)

    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [?, 257].
    magnitude_spectrograms = tf.abs(stfts)
    return magnitude_spectrograms

  def _compute_mel_filterbank_features(self, waveform):
    """Compute the mel filterbank features."""
    raise NotImplementedError("MFCC feature extraction not supported yet.")


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

  def featurize(self, text):
    """Convert string to a list of integers."""
    tokens = list(text.strip().lower())
    feats = [self.token_to_idx[token] for token in tokens]
    return feats
