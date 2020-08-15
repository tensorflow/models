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

"""Tests for object_detection.predictors.heads.box_head."""
import unittest
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import keras_box_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class ConvolutionalKerasBoxHeadTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
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
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def test_prediction_size_depthwise_false(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.ConvolutionalBoxHead(
        is_training=True,
        box_code_size=4,
        kernel_size=3,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=False,
        num_predictions_per_location=1,
        use_depthwise=False)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      box_encodings = box_prediction_head(image_feature)
      return box_encodings
    box_encodings = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 1, 4], box_encodings.shape)

  def test_prediction_size_depthwise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.ConvolutionalBoxHead(
        is_training=True,
        box_code_size=4,
        kernel_size=3,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=False,
        num_predictions_per_location=1,
        use_depthwise=True)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      box_encodings = box_prediction_head(image_feature)
      return box_encodings
    box_encodings = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 1, 4], box_encodings.shape)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class MaskRCNNKerasBoxHeadTest(test_case.TestCase):

  def _build_fc_hyperparams(
      self, op_type=hyperparams_pb2.Hyperparams.FC):
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
    return hyperparams_builder.KerasLayerHyperparams(hyperparams)

  def test_prediction_size(self):
    box_prediction_head = keras_box_head.MaskRCNNBoxHead(
        is_training=False,
        num_classes=20,
        fc_hyperparams=self._build_fc_hyperparams(),
        freeze_batchnorm=False,
        use_dropout=True,
        dropout_keep_prob=0.5,
        box_code_size=4,
        share_box_across_classes=False)
    def graph_fn():
      roi_pooled_features = tf.random_uniform(
          [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      prediction = box_prediction_head(roi_pooled_features)
      return prediction
    prediction = self.execute(graph_fn, [])
    self.assertAllEqual([64, 1, 20, 4], prediction.shape)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class WeightSharedConvolutionalKerasBoxHead(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
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
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def test_prediction_size_depthwise_false(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=4,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=False)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      box_encodings = box_prediction_head(image_feature)
      return box_encodings
    box_encodings = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 4], box_encodings.shape)

  def test_prediction_size_depthwise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=4,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=True)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      box_encodings = box_prediction_head(image_feature)
      return box_encodings
    box_encodings = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 4], box_encodings.shape)

  def test_variable_count_depth_wise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=4,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=True)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    box_prediction_head(image_feature)
    self.assertEqual(len(box_prediction_head.variables), 3)

  def test_variable_count_depth_wise_False(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=4,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=False)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    box_prediction_head(image_feature)
    self.assertEqual(len(box_prediction_head.variables), 2)


if __name__ == '__main__':
  tf.test.main()
