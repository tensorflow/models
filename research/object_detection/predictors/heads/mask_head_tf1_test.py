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

"""Tests for object_detection.predictors.heads.mask_head."""
import unittest
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import mask_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class MaskRCNNMaskHeadTest(test_case.TestCase):

  def _build_arg_scope_with_hyperparams(self,
                                        op_type=hyperparams_pb2.Hyperparams.FC):
    hyperparams = hyperparams_pb2.Hyperparams()
    hyperparams_text_proto = """
      activation: NONE
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(hyperparams_text_proto, hyperparams)
    hyperparams.op = op_type
    return hyperparams_builder.build(hyperparams, is_training=True)

  def test_prediction_size(self):
    mask_prediction_head = mask_head.MaskRCNNMaskHead(
        num_classes=20,
        conv_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
        mask_height=14,
        mask_width=14,
        mask_prediction_num_conv_layers=2,
        mask_prediction_conv_depth=256,
        masks_are_class_agnostic=False)
    roi_pooled_features = tf.random_uniform(
        [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    prediction = mask_prediction_head.predict(
        features=roi_pooled_features, num_predictions_per_location=1)
    self.assertAllEqual([64, 1, 20, 14, 14], prediction.get_shape().as_list())

  def test_prediction_size_with_convolve_then_upsample(self):
    mask_prediction_head = mask_head.MaskRCNNMaskHead(
        num_classes=20,
        conv_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
        mask_height=28,
        mask_width=28,
        mask_prediction_num_conv_layers=2,
        mask_prediction_conv_depth=256,
        masks_are_class_agnostic=True,
        convolve_then_upsample=True)
    roi_pooled_features = tf.random_uniform(
        [64, 14, 14, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    prediction = mask_prediction_head.predict(
        features=roi_pooled_features, num_predictions_per_location=1)
    self.assertAllEqual([64, 1, 1, 28, 28], prediction.get_shape().as_list())


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ConvolutionalMaskPredictorTest(test_case.TestCase):

  def _build_arg_scope_with_hyperparams(
      self, op_type=hyperparams_pb2.Hyperparams.CONV):
    hyperparams = hyperparams_pb2.Hyperparams()
    hyperparams_text_proto = """
      activation: NONE
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(hyperparams_text_proto, hyperparams)
    hyperparams.op = op_type
    return hyperparams_builder.build(hyperparams, is_training=True)

  def test_prediction_size(self):
    mask_prediction_head = mask_head.ConvolutionalMaskHead(
        is_training=True,
        num_classes=20,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=3,
        mask_height=7,
        mask_width=7)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    mask_predictions = mask_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 20, 7, 7],
                        mask_predictions.get_shape().as_list())

  def test_class_agnostic_prediction_size(self):
    mask_prediction_head = mask_head.ConvolutionalMaskHead(
        is_training=True,
        num_classes=20,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=3,
        mask_height=7,
        mask_width=7,
        masks_are_class_agnostic=True)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    mask_predictions = mask_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 1, 7, 7],
                        mask_predictions.get_shape().as_list())


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class WeightSharedConvolutionalMaskPredictorTest(test_case.TestCase):

  def _build_arg_scope_with_hyperparams(
      self, op_type=hyperparams_pb2.Hyperparams.CONV):
    hyperparams = hyperparams_pb2.Hyperparams()
    hyperparams_text_proto = """
      activation: NONE
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(hyperparams_text_proto, hyperparams)
    hyperparams.op = op_type
    return hyperparams_builder.build(hyperparams, is_training=True)

  def test_prediction_size(self):
    mask_prediction_head = (
        mask_head.WeightSharedConvolutionalMaskHead(
            num_classes=20,
            mask_height=7,
            mask_width=7))
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    mask_predictions = mask_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 20, 7, 7],
                        mask_predictions.get_shape().as_list())

  def test_class_agnostic_prediction_size(self):
    mask_prediction_head = (
        mask_head.WeightSharedConvolutionalMaskHead(
            num_classes=20,
            mask_height=7,
            mask_width=7,
            masks_are_class_agnostic=True))
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    mask_predictions = mask_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 1, 7, 7],
                        mask_predictions.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
