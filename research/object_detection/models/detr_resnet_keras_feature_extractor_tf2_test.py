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

"""Tests for models.detr_resnet_keras_feature_extractor."""
import unittest
import tensorflow.compat.v1 as tf

from object_detection.models import detr_resnet_keras_feature_extractor as detr_res
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class DetrResnetKerasFeatureExtractorTest(tf.test.TestCase):

  def _build_feature_extractor(self, architecture='resnet_v1_50'):
    return detr_res.DETRResnet50KerasFeatureExtractor(
        is_training=False,
        features_stride=32,
        batch_norm_trainable=False,
        weight_decay=0.0)

  def test_extract_proposal_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor()
    preprocessed_inputs = tf.random_uniform(
        [1, 448, 448, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shape = tf.shape(rpn_feature_map)
    self.assertAllEqual(features_shape.numpy(), [1, 14, 14, 2048])

  def test_extract_proposal_features_half_size_input(self):
    feature_extractor = self._build_feature_extractor()
    preprocessed_inputs = tf.random_uniform(
        [1, 224, 224, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shape = tf.shape(rpn_feature_map)
    self.assertAllEqual(features_shape.numpy(), [1, 7, 7, 2048])

  def test_extract_proposal_features_dies_with_incorrect_rank_inputs(self):
    feature_extractor = self._build_feature_extractor()
    preprocessed_inputs = tf.random_uniform(
        [448, 448, 3], maxval=255, dtype=tf.float32)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      feature_extractor.get_proposal_feature_extractor_model(
          name='TestScope')(preprocessed_inputs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
