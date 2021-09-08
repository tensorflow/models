"""Exports YAMNet as: TF2 SavedModel, TF-Lite model, TF-JS model.

The exported models all accept as input:
- 1-d float32 Tensor of arbitrary shape containing an audio waveform
  (assumed to be mono 16 kHz samples in the [-1, +1] range)
and return as output:
- a 2-d float32 Tensor of shape [num_frames, num_classes] containing
  predicted class scores for each frame of audio extracted from the input.
- a 2-d float32 Tensor of shape [num_frames, embedding_size] containing
  embeddings of each frame of audio.
- a 2-d float32 Tensor of shape [num_spectrogram_frames, num_mel_bins]
  containing the log mel spectrogram of the entire waveform.
The SavedModels will also contain (as an asset) a class map CSV file that maps
class indices to AudioSet class names and Freebase MIDs. The path to the class
map is available as the 'class_map_path()' method of the restored model.

Requires pip-installing tensorflow_hub and tensorflowjs.

Usage:
  export.py <path/to/YAMNet/weights-hdf-file> <path/to/output/directory>
and the various exports will be created in subdirectories of the output directory.
Assumes that it will be run in the yamnet source directory from where it loads
the class map. Skips an export if the corresponding directory already exists.
"""

import os
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf
assert tf.version.VERSION >= '2.0.0', (
    'Need at least TF 2.0, you have TF v{}'.format(tf.version.VERSION))
import tensorflow_hub as tfhub
from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs_saved_model_converter

import params as yamnet_params
import yamnet


def log(msg):
  print('\n=====\n{} | {}\n=====\n'.format(time.asctime(), msg), flush=True)


class YAMNet(tf.Module):
  """A TF2 Module wrapper around YAMNet."""
  def __init__(self, weights_path, params):
    super().__init__()
    self._yamnet = yamnet.yamnet_frames_model(params)
    self._yamnet.load_weights(weights_path)
    self._class_map_asset = tf.saved_model.Asset('yamnet_class_map.csv')

  @tf.function(input_signature=[])
  def class_map_path(self):
    return self._class_map_asset.asset_path

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, waveform):
    predictions, embeddings, log_mel_spectrogram = self._yamnet(waveform)

    return {'predictions': predictions,
            'embeddings': embeddings, 
            'log_mel_spectrogram': log_mel_spectrogram}


def check_model(model_fn, class_map_path, params):
  yamnet_classes = yamnet.class_names(class_map_path)

  """Applies yamnet_test's sanity checks to an instance of YAMNet."""
  def clip_test(waveform, expected_class_name, top_n=10):
    results = model_fn(waveform=waveform)
    predictions = results['predictions']
    embeddings = results['embeddings']
    log_mel_spectrogram = results['log_mel_spectrogram']
    clip_predictions = np.mean(predictions, axis=0)
    top_n_indices = np.argsort(clip_predictions)[-top_n:]
    top_n_scores = clip_predictions[top_n_indices]
    top_n_class_names = yamnet_classes[top_n_indices]
    top_n_predictions = list(zip(top_n_class_names, top_n_scores))
    assert expected_class_name in top_n_class_names, (
        'Did not find expected class {} in top {} predictions: {}'.format(
            expected_class_name, top_n, top_n_predictions))

  clip_test(
      waveform=np.zeros((int(3 * params.sample_rate),), dtype=np.float32),
      expected_class_name='Silence')

  np.random.seed(51773)  # Ensure repeatability.
  clip_test(
      waveform=np.random.uniform(-1.0, +1.0,
                                 (int(3 * params.sample_rate),)).astype(np.float32),
      expected_class_name='White noise')

  clip_test(
      waveform=np.sin(2 * np.pi * 440 *
                      np.arange(0, 3, 1 / params.sample_rate), dtype=np.float32),
      expected_class_name='Sine wave')


def make_tf2_export(weights_path, export_dir):
  if os.path.exists(export_dir):
    log('TF2 export already exists in {}, skipping TF2 export'.format(
        export_dir))
    return

  # Create a TF2 Module wrapper around YAMNet.
  log('Building and checking TF2 Module ...')
  params = yamnet_params.Params()
  yamnet = YAMNet(weights_path, params)
  check_model(yamnet, yamnet.class_map_path(), params)
  log('Done')

  # Make TF2 SavedModel export.
  log('Making TF2 SavedModel export ...')
  tf.saved_model.save(
      yamnet, export_dir,
      signatures={'serving_default': yamnet.__call__.get_concrete_function()})
  log('Done')

  # Check export with TF-Hub in TF2.
  log('Checking TF2 SavedModel export in TF2 ...')
  model = tfhub.load(export_dir)
  check_model(model, model.class_map_path(), params)
  log('Done')

  # Check export with TF-Hub in TF1.
  log('Checking TF2 SavedModel export in TF1 ...')
  with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
    model = tfhub.load(export_dir)
    sess.run(tf.compat.v1.global_variables_initializer())
    def run_model(waveform):
      return sess.run(model(waveform))
    check_model(run_model, model.class_map_path().eval(), params)
  log('Done')


def make_tflite_export(weights_path, export_dir):
  if os.path.exists(export_dir):
    log('TF-Lite export already exists in {}, skipping TF-Lite export'.format(
        export_dir))
    return

  # Create a TF-Lite compatible Module wrapper around YAMNet.
  log('Building and checking TF-Lite Module ...')
  params = yamnet_params.Params(tflite_compatible=True)
  yamnet = YAMNet(weights_path, params)
  check_model(yamnet, yamnet.class_map_path(), params)
  log('Done')

  # Make TF-Lite SavedModel export.
  log('Making TF-Lite SavedModel export ...')
  saved_model_dir = os.path.join(export_dir, 'saved_model')
  os.makedirs(saved_model_dir)
  tf.saved_model.save(
      yamnet, saved_model_dir,
      signatures={'serving_default': yamnet.__call__.get_concrete_function()})
  log('Done')

  # Check that the export can be loaded and works.
  log('Checking TF-Lite SavedModel export in TF2 ...')
  model = tf.saved_model.load(saved_model_dir)
  check_model(model, model.class_map_path(), params)
  log('Done')

  # Make a TF-Lite model from the SavedModel.
  log('Making TF-Lite model ...')
  tflite_converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir, signature_keys=['serving_default'])
  tflite_model = tflite_converter.convert()
  tflite_model_path = os.path.join(export_dir, 'yamnet.tflite')
  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
  log('Done')

  # Check the TF-Lite export.
  log('Checking TF-Lite model ...')
  interpreter = tf.lite.Interpreter(tflite_model_path)
  runner = interpreter.get_signature_runner('serving_default')
  check_model(runner, 'yamnet_class_map.csv', params)
  log('Done')

  return saved_model_dir


def make_tfjs_export(tflite_saved_model_dir, export_dir):
  if os.path.exists(export_dir):
    log('TF-JS export already exists in {}, skipping TF-JS export'.format(
        export_dir))
    return

  # Make a TF-JS model from the TF-Lite SavedModel export.
  log('Making TF-JS model ...')
  os.makedirs(export_dir)
  tfjs_saved_model_converter.convert_tf_saved_model(
      tflite_saved_model_dir, export_dir)
  log('Done')


def main(args):
  weights_path = args[0]
  output_dir = args[1]

  tf2_export_dir = os.path.join(output_dir, 'tf2')
  make_tf2_export(weights_path, tf2_export_dir)

  tflite_export_dir = os.path.join(output_dir, 'tflite')
  tflite_saved_model_dir = make_tflite_export(weights_path, tflite_export_dir)

  tfjs_export_dir = os.path.join(output_dir, 'tfjs')
  make_tfjs_export(tflite_saved_model_dir, tfjs_export_dir)

if __name__ == '__main__':
  main(sys.argv[1:])
