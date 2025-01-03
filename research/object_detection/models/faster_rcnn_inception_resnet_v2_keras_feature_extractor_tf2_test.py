# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for models.faster_rcnn_inception_resnet_v2_keras_feature_extractor."""
import unittest
import tensorflow.compat.v1 as tf

from object_detection.models import faster_rcnn_inception_resnet_v2_keras_feature_extractor as frcnn_inc_res
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class FasterRcnnInceptionResnetV2KerasFeatureExtractorTest(tf.test.TestCase):

  def _build_feature_extractor(self, first_stage_features_stride):
    return frcnn_inc_res.FasterRCNNInceptionResnetV2KerasFeatureExtractor(
        is_training=False,
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_trainable=False,
        weight_decay=0.0)

  def test_extract_proposal_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    preprocessed_inputs = tf.random_uniform(
        [1, 299, 299, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shape = tf.shape(rpn_feature_map)

    self.assertAllEqual(features_shape.numpy(), [1, 19, 19, 1088])

  def test_extract_proposal_features_stride_eight(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=8)
    preprocessed_inputs = tf.random_uniform(
        [1, 224, 224, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shape = tf.shape(rpn_feature_map)

    self.assertAllEqual(features_shape.numpy(), [1, 28, 28, 1088])

  def test_extract_proposal_features_half_size_input(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    preprocessed_inputs = tf.random_uniform(
        [1, 112, 112, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shape = tf.shape(rpn_feature_map)
    self.assertAllEqual(features_shape.numpy(), [1, 7, 7, 1088])

  def test_extract_box_classifier_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    proposal_feature_maps = tf.random_uniform(
        [2, 17, 17, 1088], maxval=255, dtype=tf.float32)
    model = feature_extractor.get_box_classifier_feature_extractor_model(
        name='TestScope')
    proposal_classifier_features = (
        model(proposal_feature_maps))
    features_shape = tf.shape(proposal_classifier_features)
    self.assertAllEqual(features_shape.numpy(), [2, 9, 9, 1536])


if __name__ == '__main__':
  tf.test.main()
