"""Exports VGGish as a SavedModel for publication to TF Hub.

The exported SavedModel accepts a 1-d float32 Tensor of arbitrary shape
containing an audio waveform (assumed to be mono 16 kHz samples in the [-1, +1]
range) and returns a 2-d float32 batch of 128-d VGGish embeddings, one per
0.96s example generated from the waveform.

Requires pip-installing tensorflow_hub.

Usage:
  vggish_export_tfhub.py <path/to/VGGish/checkpoint> <path/to/tfhub/export>
"""

import sys
sys.path.append('..')  # Lets us import yamnet modules from sibling directory.

import numpy as np
import resampy
import tensorflow as tf
assert tf.version.VERSION >= '2.0.0', (
    'Need at least TF 2.0, you have TF v{}'.format(tf.version.VERSION))
import tensorflow_hub as tfhub

import vggish_input
import vggish_params
import vggish_slim
from yamnet import features as yamnet_features
from yamnet import params as yamnet_params


def vggish_definer(variables, checkpoint_path):
  """Defines VGGish with variables tracked and initialized from a checkpoint."""
  reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)

  def var_tracker(next_creator, **kwargs):
    """Variable creation hook that assigns initial values from a checkpoint."""
    var_name = kwargs['name']
    var_value = reader.get_tensor(var_name)
    kwargs.update({'initial_value': var_value})
    var = next_creator(**kwargs)
    variables.append(var)
    return var

  def waveform_to_features(waveform):
    """Creates VGGish features using the YAMNet feature extractor."""
    params = yamnet_params.Params(
        sample_rate=vggish_params.SAMPLE_RATE,
        stft_window_seconds=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        stft_hop_seconds=vggish_params.STFT_HOP_LENGTH_SECONDS,
        mel_bands=vggish_params.NUM_MEL_BINS,
        mel_min_hz=vggish_params.MEL_MIN_HZ,
        mel_max_hz=vggish_params.MEL_MAX_HZ,
        log_offset=vggish_params.LOG_OFFSET,
        patch_window_seconds=vggish_params.EXAMPLE_WINDOW_SECONDS,
        patch_hop_seconds=vggish_params.EXAMPLE_HOP_SECONDS)
    log_mel_spectrogram, features = yamnet_features.waveform_to_log_mel_spectrogram_patches(
        waveform, params)
    return features

  def define_vggish(waveform):
    with tf.variable_creator_scope(var_tracker):
      features = waveform_to_features(waveform)
      return vggish_slim.define_vggish_slim(features, training=False)

  return define_vggish


class VGGish(tf.Module):
  """A TF2 Module wrapper around VGGish."""
  def __init__(self, checkpoint_path):
    super().__init__()
    self._variables = []
    self._vggish_fn = tf.compat.v1.wrap_function(
        vggish_definer(self._variables, checkpoint_path),
        signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))

  @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
  def __call__(self, waveform):
    return self._vggish_fn(waveform)


def check_model(model_fn):
  """Applies vggish_smoke_test's sanity check to an instance of VGGish."""
  num_secs = 3
  freq = 1000
  sr = 44100
  t = np.arange(0, num_secs, 1 / sr)
  waveform = np.sin(2 * np.pi * freq * t)

  waveform = resampy.resample(waveform, sr, vggish_params.SAMPLE_RATE)
  embeddings = model_fn(waveform)

  expected_embedding_mean = -0.0333
  expected_embedding_std = 0.380
  rel_error = 0.1
  np.testing.assert_allclose(
      [np.mean(embeddings), np.std(embeddings)],
      [expected_embedding_mean, expected_embedding_std],
      rtol=rel_error)


def main(args):
  # Create a TF2 wrapper around VGGish.
  vggish_checkpoint_path = args[0]
  vggish = VGGish(vggish_checkpoint_path)
  check_model(vggish)

  # Make TF-Hub export.
  vggish_tfhub_export_path = args[1]
  tf.saved_model.save(vggish, vggish_tfhub_export_path)

  # Check export in TF2.
  model = tfhub.load(vggish_tfhub_export_path)
  check_model(model)

  # Check export in TF1.
  with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
    model = tfhub.load(vggish_tfhub_export_path)
    sess.run(tf.compat.v1.global_variables_initializer())
    def run_model(waveform):
      embeddings = model(waveform)
      return sess.run(embeddings)
    check_model(run_model)

if __name__ == '__main__':
  main(sys.argv[1:])
