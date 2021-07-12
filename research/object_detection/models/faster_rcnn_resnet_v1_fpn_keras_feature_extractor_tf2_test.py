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

"""Tests for models.faster_rcnn_resnet_v1_fpn_keras_feature_extractor."""
import unittest
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models import faster_rcnn_resnet_v1_fpn_keras_feature_extractor as frcnn_res_fpn
from object_detection.protos import hyperparams_pb2
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class FasterRCNNResnetV1FpnKerasFeatureExtractorTest(tf.test.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Parse(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _build_feature_extractor(self):
    return frcnn_res_fpn.FasterRCNNResnet50FpnKerasFeatureExtractor(
        is_training=False,
        conv_hyperparams=self._build_conv_hyperparams(),
        first_stage_features_stride=16,
        batch_norm_trainable=False,
        weight_decay=0.0)

  def test_extract_proposal_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor()
    preprocessed_inputs = tf.random_uniform(
        [2, 448, 448, 3], maxval=255, dtype=tf.float32)
    rpn_feature_maps = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shapes = [tf.shape(rpn_feature_map)
                       for rpn_feature_map in rpn_feature_maps]

    self.assertAllEqual(features_shapes[0].numpy(), [2, 112, 112, 256])
    self.assertAllEqual(features_shapes[1].numpy(), [2, 56, 56, 256])
    self.assertAllEqual(features_shapes[2].numpy(), [2, 28, 28, 256])
    self.assertAllEqual(features_shapes[3].numpy(), [2, 14, 14, 256])
    self.assertAllEqual(features_shapes[4].numpy(), [2, 7, 7, 256])

  def test_extract_proposal_features_half_size_input(self):
    feature_extractor = self._build_feature_extractor()
    preprocessed_inputs = tf.random_uniform(
        [2, 224, 224, 3], maxval=255, dtype=tf.float32)
    rpn_feature_maps = feature_extractor.get_proposal_feature_extractor_model(
        name='TestScope')(preprocessed_inputs)
    features_shapes = [tf.shape(rpn_feature_map)
                       for rpn_feature_map in rpn_feature_maps]

    self.assertAllEqual(features_shapes[0].numpy(), [2, 56, 56, 256])
    self.assertAllEqual(features_shapes[1].numpy(), [2, 28, 28, 256])
    self.assertAllEqual(features_shapes[2].numpy(), [2, 14, 14, 256])
    self.assertAllEqual(features_shapes[3].numpy(), [2, 7, 7, 256])
    self.assertAllEqual(features_shapes[4].numpy(), [2, 4, 4, 256])

  def test_extract_box_classifier_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor()
    proposal_feature_maps = tf.random_uniform(
        [3, 7, 7, 1024], maxval=255, dtype=tf.float32)
    model = feature_extractor.get_box_classifier_feature_extractor_model(
        name='TestScope')
    proposal_classifier_features = (
        model(proposal_feature_maps))
    features_shape = tf.shape(proposal_classifier_features)

    self.assertAllEqual(features_shape.numpy(), [3, 1, 1, 1024])
