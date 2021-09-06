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

"""Installation test for YAMNet."""

import pathlib

import numpy as np
import tensorflow as tf
from absl import flags

import export
import params
import yamnet


flags.DEFINE_bool('update', False, 'update the numpy reference result files')

FLAGS = flags.FLAGS

HERE = pathlib.Path(__file__).parent

class YAMNetTest(tf.test.TestCase):
  _params = params.Params()
  _yamnet = None
  _yamnet_classes = None
  _atol = 2e-5
  _rtol = 5e-6
  _check_outputs = ('log_mel_spectrogram',
                    'embeddings',
                    'predictions', 'logits')

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._yamnet = export.YAMNet(cls._params)
    weights_file = tf.keras.utils.get_file(
        fname='yamnet.h5',
        origin='https://storage.googleapis.com/audioset/yamnet.h5',
        cache_dir=HERE, cache_subdir='.')

    cls._yamnet.load_weights('yamnet.h5')
    cls._yamnet_classes = yamnet.class_names('yamnet_class_map.csv')

  def clip_test(self, waveform, expected_class_name, top_n=10):
    """Run the model on the waveform, check that expected class is in top-n."""
    outputs = self._yamnet(
        waveform, as_dict=True,
        returns=self._check_outputs)
    clip_predictions = np.mean(outputs['predictions'], axis=0)
    top_n_indices = np.argsort(clip_predictions)[-top_n:]
    top_n_scores = clip_predictions[top_n_indices]
    top_n_class_names = self._yamnet_classes[top_n_indices]
    top_n_predictions = list(zip(top_n_class_names, top_n_scores))
    self.assertIn(
        expected_class_name, top_n_class_names,
        'Did not find expected class {} in top {} predictions: {}'.format(
        expected_class_name, top_n, top_n_predictions))
    return outputs

  def _name(self):
    return self.id().replace("_","").replace('.','_')

  def check_close(self, outputs):
    file_path = HERE/'npz'/self._name()

    file_path = file_path.with_suffix(".npz")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if FLAGS.update:
      np.savez(file=file_path, **outputs)
    else:
      reference = np.load(file_path)
      for key in self._check_outputs:
        out_value = outputs[key].numpy()
        ref_value = reference[key]
        self.assertEqual(out_value.shape, ref_value.shape, msg=f"`{key}.shape` didn't match")
        self.assertAllClose(out_value, ref_value,
                            atol=self._atol, rtol=self._rtol,
                            msg=f"`{key}` didn't match")

  def testZeros(self):
    outputs = self.clip_test(
        waveform=np.zeros((int(3 * self._params.sample_rate),)),
        expected_class_name='Silence')

    self.check_close(outputs)

  def testRandom(self):
    np.random.seed(51773)  # Ensure repeatability.
    outputs = self.clip_test(
        waveform=np.random.uniform(-1.0, +1.0,
                                   (int(3 * self._params.sample_rate),)),
        expected_class_name='White noise')
    self.check_close(outputs)

  def testSine(self):
    outputs = self.clip_test(
        waveform=np.sin(2 * np.pi * 440 *
                        np.arange(0, 3, 1 / self._params.sample_rate)),
        expected_class_name='Sine wave')
    self.check_close(outputs)

  def testBatchOne(self):
    sine_wave = np.sin(2 * np.pi * 440 *
                        np.arange(0, 3, 1 / self._params.sample_rate))
    sine_outputs = self.clip_test(
        waveform=sine_wave, expected_class_name='Sine wave')


    batch_one = sine_wave[None, ...]
    batch_one_outputs = self._yamnet(
        waveform=batch_one, as_dict=True,
        returns=self._check_outputs)

    for name, value in batch_one_outputs.items():
      self.assertAllClose(value[0], sine_outputs[name],
                          msg=f"`{name}` didn't match")

  def testBatchReplicate(self):
    sine_wave = np.sin(2 * np.pi * 440 *
                        np.arange(0, 3, 1 / self._params.sample_rate))
    sine_outputs = self.clip_test(
        waveform=sine_wave, expected_class_name='Sine wave')


    batch_one = np.stack([sine_wave, sine_wave, sine_wave])
    batch_one_outputs = self._yamnet(
        waveform=batch_one, as_dict=True,
        returns=self._check_outputs)

    for name, value in batch_one_outputs.items():
      self.assertAllClose(value[0], sine_outputs[name],
                          msg=f"`{name}[0]` didn't match")
      self.assertAllClose(value[1], sine_outputs[name],
                          msg=f"`{name}[1]` didn't match")
      self.assertAllClose(value[2], sine_outputs[name],
                          msg=f"`{name}[2]` didn't match")

  def testBatch(self):
    sine_wave = np.sin(2 * np.pi * 440 *
                        np.arange(0, 3, 1 / self._params.sample_rate))
    sine_outputs = self.clip_test(
        waveform=sine_wave, expected_class_name='Sine wave')

    random_wave = np.random.uniform(-1.0, +1.0,
                                   (int(3 * self._params.sample_rate),))
    random_outputs = self.clip_test(
        waveform=random_wave,
        expected_class_name='White noise')

    stacked = np.stack([sine_wave, random_wave], axis=0)
    stacked_output = self._yamnet(
        waveform=stacked, as_dict=True,
        returns=self._check_outputs)

    for name, value in stacked_output.items():
      self.assertAllClose(value[0], sine_outputs[name],
                          msg=f"sine_outputs[{name}] didn't match")
      self.assertAllClose(value[1], random_outputs[name],
                          msg=f"random_outputs[{name}] didn't match")

class YAMNetLiteTest(YAMNetTest):
  # Same tests with the TFLite model. Use the files generated by the base class.

  _params = params.Params(tflite_compatible=True)
  _atol = 0.05
  _rtol = 0.01


  def _name(self):
    return super()._name().replace('YAMNetLite', 'YAMNet')

  def check_close(self, outputs):
    # Compare to the saved lite outputs
    if not FLAGS.update:
      super().check_close(outputs)


if __name__ == '__main__':
  tf.test.main()
