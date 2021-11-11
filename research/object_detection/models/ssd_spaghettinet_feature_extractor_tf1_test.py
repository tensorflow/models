# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ssd_spaghettinet_feature_extractor."""
import unittest
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_spaghettinet_feature_extractor
from object_detection.utils import tf_version

try:
  from tensorflow.contrib import quantize as contrib_quantize  # pylint: disable=g-import-not-at-top
except:  # pylint: disable=bare-except
  pass


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class SSDSpaghettiNetFeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self, arch_name, is_training=True):
    return ssd_spaghettinet_feature_extractor.SSDSpaghettinetFeatureExtractor(
        is_training=is_training,
        spaghettinet_arch_name=arch_name,
        depth_multiplier=1.0,
        min_depth=4,
        pad_to_multiple=1,
        conv_hyperparams_fn=self.conv_hyperparams_fn)

  def _test_spaghettinet_returns_correct_shapes(self, arch_name,
                                                expected_feature_map_shapes):
    image = tf.random.normal((1, 320, 320, 3))
    feature_extractor = self._create_feature_extractor(arch_name)
    feature_maps = feature_extractor.extract_features(image)

    self.assertEqual(len(expected_feature_map_shapes), len(feature_maps))
    for expected_shape, x in zip(expected_feature_map_shapes, feature_maps):
      self.assertTrue(x.shape.is_compatible_with(expected_shape))

  def test_spaghettinet_edgetpu_s(self):
    expected_feature_map_shapes = [(1, 20, 20, 120), (1, 10, 10, 168),
                                   (1, 5, 5, 136), (1, 3, 3, 136),
                                   (1, 3, 3, 64)]
    self._test_spaghettinet_returns_correct_shapes('spaghettinet_edgetpu_s',
                                                   expected_feature_map_shapes)

  def test_spaghettinet_edgetpu_m(self):
    expected_feature_map_shapes = [(1, 20, 20, 120), (1, 10, 10, 168),
                                   (1, 5, 5, 136), (1, 3, 3, 136),
                                   (1, 3, 3, 64)]
    self._test_spaghettinet_returns_correct_shapes('spaghettinet_edgetpu_m',
                                                   expected_feature_map_shapes)

  def test_spaghettinet_edgetpu_l(self):
    expected_feature_map_shapes = [(1, 20, 20, 120), (1, 10, 10, 168),
                                   (1, 5, 5, 112), (1, 3, 3, 128),
                                   (1, 3, 3, 64)]
    self._test_spaghettinet_returns_correct_shapes('spaghettinet_edgetpu_l',
                                                   expected_feature_map_shapes)

  def _check_quantization(self, model_fn):
    checkpoint_dir = self.get_temp_dir()

    with tf.Graph().as_default() as training_graph:
      model_fn(is_training=True)
      contrib_quantize.experimental_create_training_graph(training_graph)
      with self.session(graph=training_graph) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, checkpoint_dir)

    with tf.Graph().as_default() as eval_graph:
      model_fn(is_training=False)
      contrib_quantize.experimental_create_eval_graph(eval_graph)
      with self.session(graph=eval_graph) as sess:
        tf.train.Saver().restore(sess, checkpoint_dir)

  def _test_spaghettinet_quantization(self, arch_name):
    def model_fn(is_training):
      image = tf.random.normal((1, 320, 320, 3))
      feature_extractor = self._create_feature_extractor(
          arch_name, is_training=is_training)
      feature_extractor.extract_features(image)
    self._check_quantization(model_fn)

  def test_spaghettinet_edgetpu_s_quantization(self):
    self._test_spaghettinet_quantization('spaghettinet_edgetpu_s')

  def test_spaghettinet_edgetpu_m_quantization(self):
    self._test_spaghettinet_quantization('spaghettinet_edgetpu_m')

  def test_spaghettinet_edgetpu_l_quantization(self):
    self._test_spaghettinet_quantization('spaghettinet_edgetpu_l')


if __name__ == '__main__':
  tf.test.main()
