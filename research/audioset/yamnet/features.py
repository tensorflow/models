# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Feature computation for YAMNet."""

import numpy as np
import tensorflow as tf


def waveform_to_log_mel_spectrogram_patches(waveform, params):
  """Compute log mel spectrogram patches of a 1-D waveform."""
  with tf.name_scope('log_mel_features'):
    # waveform has shape [<# samples>]

    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    window_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
    hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1
    magnitude_spectrogram = tf.abs(tf.signal.stft(
        signals=waveform,
        frame_length=window_length_samples,
        frame_step=hop_length_samples,
        fft_length=fft_length))
    # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]

    # Convert spectrogram into log mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.MEL_BANDS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.SAMPLE_RATE,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET)
    # log_mel_spectrogram has shape [<# STFT frames>, MEL_BANDS]

    # Frame spectrogram (shape [<# STFT frames>, MEL_BANDS]) into patches (the
    # input examples). Only complete frames are emitted, so if there is less
    # than PATCH_WINDOW_SECONDS of waveform then nothing is emitted (to avoid
    # this, zero-pad before processing).
    spectrogram_hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    spectrogram_sample_rate = params.SAMPLE_RATE / spectrogram_hop_length_samples
    patch_window_length_samples = int(
      round(spectrogram_sample_rate * params.PATCH_WINDOW_SECONDS))
    patch_hop_length_samples = int(
      round(spectrogram_sample_rate * params.PATCH_HOP_SECONDS))
    features = tf.signal.frame(
        signal=log_mel_spectrogram,
        frame_length=patch_window_length_samples,
        frame_step=patch_hop_length_samples,
        axis=0)
    # features has shape [<# patches>, <# STFT frames in an patch>, MEL_BANDS]

    return log_mel_spectrogram, features


def pad_waveform(waveform, params):
  """Pads waveform with silence if needed to get an integral number of patches."""
  # In order to produce one patch of log mel spectrogram input to YAMNet, we
  # need at least one patch window length of waveform plus enough extra samples
  # to complete the final STFT analysis window.
  min_waveform_seconds = (
      params.PATCH_WINDOW_SECONDS +
      params.STFT_WINDOW_SECONDS - params.STFT_HOP_SECONDS)
  min_num_samples = tf.cast(min_waveform_seconds * params.SAMPLE_RATE, tf.int32)
  num_samples = tf.size(waveform)
  num_padding_samples = tf.maximum(0, min_num_samples - num_samples)

  # In addition, there might be enough waveform for one or more additional
  # patches formed by hopping forward. If there are more samples than one patch,
  # round up to an integral number of hops.
  num_samples = tf.maximum(num_samples, min_num_samples)
  num_samples_after_first_patch = num_samples - min_num_samples
  hop_samples = tf.cast(params.PATCH_HOP_SECONDS * params.SAMPLE_RATE, tf.int32)
  num_hops_after_first_patch = tf.cast(tf.math.ceil(
      tf.math.divide(num_samples_after_first_patch, hop_samples)), tf.int32)
  num_padding_samples += (
      hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)

  padded_waveform = tf.pad(waveform, [[0, num_padding_samples]],
                           mode='CONSTANT', constant_values=0.0)
  return padded_waveform
