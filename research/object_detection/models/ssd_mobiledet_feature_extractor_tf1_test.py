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
"""Tests for ssd_mobiledet_feature_extractor."""
import unittest
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobiledet_feature_extractor
from object_detection.utils import tf_version

try:
  from tensorflow.contrib import quantize as contrib_quantize  # pylint: disable=g-import-not-at-top
except:  # pylint: disable=bare-except
  pass


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class SSDMobileDetFeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                feature_extractor_cls,
                                is_training=False,
                                depth_multiplier=1.0,
                                pad_to_multiple=1,
                                use_explicit_padding=False,
                                use_keras=False):
    """Constructs a new MobileDet feature extractor.

    Args:
      feature_extractor_cls: feature extractor class.
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: If True, we will use 'VALID' padding for
        convolutions, but prepad inputs so that the output dimensions are the
        same as if 'SAME' padding were used.
      use_keras: if True builds a keras-based feature extractor, if False builds
        a slim-based one.

    Returns:
      an ssd_meta_arch.SSDMobileDetFeatureExtractor object.
    """
    min_depth = 32
    return feature_extractor_cls(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        self.conv_hyperparams_fn,
        use_explicit_padding=use_explicit_padding)

  def test_mobiledet_cpu_returns_correct_shapes(self):
    expected_feature_map_shapes = [(2, 40, 20, 72),
                                   (2, 20, 10, 144),
                                   (2, 10, 5, 512),
                                   (2, 5, 3, 256),
                                   (2, 3, 2, 256),
                                   (2, 2, 1, 128)]
    feature_extractor = self._create_feature_extractor(
        ssd_mobiledet_feature_extractor.SSDMobileDetCPUFeatureExtractor)
    image = tf.random.normal((2, 640, 320, 3))
    feature_maps = feature_extractor.extract_features(image)

    self.assertEqual(len(expected_feature_map_shapes), len(feature_maps))
    for expected_shape, x in zip(expected_feature_map_shapes, feature_maps):
      self.assertTrue(x.shape.is_compatible_with(expected_shape))

  def test_mobiledet_dsp_returns_correct_shapes(self):
    expected_feature_map_shapes = [(2, 40, 20, 144),
                                   (2, 20, 10, 240),
                                   (2, 10, 5, 512),
                                   (2, 5, 3, 256),
                                   (2, 3, 2, 256),
                                   (2, 2, 1, 128)]
    feature_extractor = self._create_feature_extractor(
        ssd_mobiledet_feature_extractor.SSDMobileDetDSPFeatureExtractor)
    image = tf.random.normal((2, 640, 320, 3))
    feature_maps = feature_extractor.extract_features(image)

    self.assertEqual(len(expected_feature_map_shapes), len(feature_maps))
    for expected_shape, x in zip(expected_feature_map_shapes, feature_maps):
      self.assertTrue(x.shape.is_compatible_with(expected_shape))

  def test_mobiledet_edgetpu_returns_correct_shapes(self):
    expected_feature_map_shapes = [(2, 40, 20, 96),
                                   (2, 20, 10, 384),
                                   (2, 10, 5, 512),
                                   (2, 5, 3, 256),
                                   (2, 3, 2, 256),
                                   (2, 2, 1, 128)]
    feature_extractor = self._create_feature_extractor(
        ssd_mobiledet_feature_extractor.SSDMobileDetEdgeTPUFeatureExtractor)
    image = tf.random.normal((2, 640, 320, 3))
    feature_maps = feature_extractor.extract_features(image)

    self.assertEqual(len(expected_feature_map_shapes), len(feature_maps))
    for expected_shape, x in zip(expected_feature_map_shapes, feature_maps):
      self.assertTrue(x.shape.is_compatible_with(expected_shape))

  def test_mobiledet_gpu_returns_correct_shapes(self):
    expected_feature_map_shapes = [(2, 40, 20, 128), (2, 20, 10, 384),
                                   (2, 10, 5, 512), (2, 5, 3, 256),
                                   (2, 3, 2, 256), (2, 2, 1, 128)]
    feature_extractor = self._create_feature_extractor(
        ssd_mobiledet_feature_extractor.SSDMobileDetGPUFeatureExtractor)
    image = tf.random.normal((2, 640, 320, 3))
    feature_maps = feature_extractor.extract_features(image)

    self.assertEqual(len(expected_feature_map_shapes), len(feature_maps))
    for expected_shape, x in zip(expected_feature_map_shapes, feature_maps):
      self.assertTrue(x.shape.is_compatible_with(expected_shape))

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

  def test_mobiledet_cpu_quantization(self):
    def model_fn(is_training):
      feature_extractor = self._create_feature_extractor(
          ssd_mobiledet_feature_extractor.SSDMobileDetCPUFeatureExtractor,
          is_training=is_training)
      image = tf.random.normal((2, 320, 320, 3))
      feature_extractor.extract_features(image)
    self._check_quantization(model_fn)

  def test_mobiledet_dsp_quantization(self):
    def model_fn(is_training):
      feature_extractor = self._create_feature_extractor(
          ssd_mobiledet_feature_extractor.SSDMobileDetDSPFeatureExtractor,
          is_training=is_training)
      image = tf.random.normal((2, 320, 320, 3))
      feature_extractor.extract_features(image)
    self._check_quantization(model_fn)

  def test_mobiledet_edgetpu_quantization(self):
    def model_fn(is_training):
      feature_extractor = self._create_feature_extractor(
          ssd_mobiledet_feature_extractor.SSDMobileDetEdgeTPUFeatureExtractor,
          is_training=is_training)
      image = tf.random.normal((2, 320, 320, 3))
      feature_extractor.extract_features(image)
    self._check_quantization(model_fn)


if __name__ == '__main__':
  tf.test.main()
