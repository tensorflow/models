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

  _params = None
  _yamnet = None
  _yamnet_classes = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._params = params.Params()
    cls._yamnet = export.YAMNet(cls._params)
    cls._yamnet.load_weights('yamnet.h5')
    cls._yamnet_classes = yamnet.class_names('yamnet_class_map.csv')

  def clip_test(self, waveform, expected_class_name, top_n=10):
    """Run the model on the waveform, check that expected class is in top-n."""
    outputs = YAMNetTest._yamnet(
        waveform, as_dict=True,
        returns=('predictions', 'embeddings',
                 'log_mel_spectrogram', 'logits'))
    clip_predictions = np.mean(outputs['predictions'], axis=0)
    top_n_indices = np.argsort(clip_predictions)[-top_n:]
    top_n_scores = clip_predictions[top_n_indices]
    top_n_class_names = YAMNetTest._yamnet_classes[top_n_indices]
    top_n_predictions = list(zip(top_n_class_names, top_n_scores))
    self.assertIn(
        expected_class_name, top_n_class_names,
        'Did not find expected class {} in top {} predictions: {}'.format(
        expected_class_name, top_n, top_n_predictions))
    return outputs

  def check_numpy(self, outputs):
    test_id = self.id().replace("_","").replace('.','_')
    file_path = HERE/'npz'/test_id
    file_path = file_path.with_suffix(".npz")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if FLAGS.update:
      np.savez(file=file_path, **outputs)
    else:
      reference = np.load(file_path)
      for key in outputs.keys():
        out_value = outputs[key]
        ref_value = reference[key]
        self.assertTrue(np.allclose(out_value, ref_value),
                        f"`{key}` didn't match")

  def testZeros(self):
    outputs = self.clip_test(
        waveform=np.zeros((int(3 * YAMNetTest._params.sample_rate),)),
        expected_class_name='Silence')
    self.check_numpy(outputs)

  def testRandom(self):
    np.random.seed(51773)  # Ensure repeatability.
    outputs = self.clip_test(
        waveform=np.random.uniform(-1.0, +1.0,
                                   (int(3 * YAMNetTest._params.sample_rate),)),
        expected_class_name='White noise')
    self.check_numpy(outputs)

  def testSine(self):
    outputs = self.clip_test(
        waveform=np.sin(2 * np.pi * 440 *
                        np.arange(0, 3, 1 / YAMNetTest._params.sample_rate)),
        expected_class_name='Sine wave')
    self.check_numpy(outputs)


if __name__ == '__main__':
  tf.test.main()
