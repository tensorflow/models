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
      round(params.sample_rate * params.stft_window_seconds))
    hop_length_samples = int(
      round(params.sample_rate * params.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1
    if params.tflite_compatible:
      magnitude_spectrogram = _tflite_stft_magnitude(
          signal=waveform,
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length)
    else:
      magnitude_spectrogram = tf.abs(tf.signal.stft(
          signals=waveform,
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length))
    # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]

    # Convert spectrogram into log mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.sample_rate,
        lower_edge_hertz=params.mel_min_hz,
        upper_edge_hertz=params.mel_max_hz)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.log_offset)
    # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

    # Frame spectrogram (shape [<# STFT frames>, params.mel_bands]) into patches
    # (the input examples). Only complete frames are emitted, so if there is
    # less than params.patch_window_seconds of waveform then nothing is emitted
    # (to avoid this, zero-pad before processing).
    spectrogram_hop_length_samples = int(
      round(params.sample_rate * params.stft_hop_seconds))
    spectrogram_sample_rate = params.sample_rate / spectrogram_hop_length_samples
    patch_window_length_samples = int(
      round(spectrogram_sample_rate * params.patch_window_seconds))
    patch_hop_length_samples = int(
      round(spectrogram_sample_rate * params.patch_hop_seconds))
    features = tf.signal.frame(
        signal=log_mel_spectrogram,
        frame_length=patch_window_length_samples,
        frame_step=patch_hop_length_samples,
        axis=0)
    # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]

    return log_mel_spectrogram, features


def pad_waveform(waveform, params):
  """Pads waveform with silence if needed to get an integral number of patches."""
  # In order to produce one patch of log mel spectrogram input to YAMNet, we
  # need at least one patch window length of waveform plus enough extra samples
  # to complete the final STFT analysis window.
  min_waveform_seconds = (
      params.patch_window_seconds +
      params.stft_window_seconds - params.stft_hop_seconds)
  min_num_samples = tf.cast(min_waveform_seconds * params.sample_rate, tf.int32)
  num_samples = tf.shape(waveform)[0]
  num_padding_samples = tf.maximum(0, min_num_samples - num_samples)

  # In addition, there might be enough waveform for one or more additional
  # patches formed by hopping forward. If there are more samples than one patch,
  # round up to an integral number of hops.
  num_samples = tf.maximum(num_samples, min_num_samples)
  num_samples_after_first_patch = num_samples - min_num_samples
  hop_samples = tf.cast(params.patch_hop_seconds * params.sample_rate, tf.int32)
  num_hops_after_first_patch = tf.cast(tf.math.ceil(
          tf.cast(num_samples_after_first_patch, tf.float32) /
          tf.cast(hop_samples, tf.float32)), tf.int32)
  num_padding_samples += (
      hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)

  padded_waveform = tf.pad(waveform, [[0, num_padding_samples]],
                           mode='CONSTANT', constant_values=0.0)
  return padded_waveform


def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
  """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""
  def _hann_window():
    return tf.reshape(
      tf.constant(
          (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
          ).astype(np.float32),
          name='hann_window'), [1, frame_length])

  def _dft_matrix(dft_length):
    """Calculate the full DFT matrix in NumPy."""
    # See https://en.wikipedia.org/wiki/DFT_matrix
    omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

  def _rdft(framed_signal, fft_length):
    """Implement real-input Discrete Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping only the
    # first half ("positive frequencies").  So discard the second half of rows,
    # but transpose the array for right-multiplication.  The DFT matrix is
    # symmetric, so we could have done it more directly, but this reflects our
    # intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
        fft_length // 2 + 1), :].transpose()
    real_dft_matrix = tf.constant(
        np.real(complex_dft_matrix_kept_values).astype(np.float32),
        name='real_dft_matrix')
    imag_dft_matrix = tf.constant(
        np.imag(complex_dft_matrix_kept_values).astype(np.float32),
        name='imaginary_dft_matrix')
    signal_frame_length = tf.shape(framed_signal)[-1]
    half_pad = (fft_length - signal_frame_length) // 2
    padded_frames = tf.pad(
        framed_signal,
        [
            # Don't add any padding in the frame dimension.
            [0, 0],
            # Pad before and after the signal within each frame.
            [half_pad, fft_length - signal_frame_length - half_pad]
        ],
        mode='CONSTANT',
        constant_values=0.0)
    real_stft = tf.matmul(padded_frames, real_dft_matrix)
    imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
    return real_stft, imag_stft

  def _complex_abs(real, imag):
    return tf.sqrt(tf.add(real * real, imag * imag))

  framed_signal = tf.signal.frame(signal, frame_length, frame_step)
  windowed_signal = framed_signal * _hann_window()
  real_stft, imag_stft = _rdft(windowed_signal, fft_length)
  stft_magnitude = _complex_abs(real_stft, imag_stft)
  return stft_magnitude
