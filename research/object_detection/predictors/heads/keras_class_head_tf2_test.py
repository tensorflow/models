# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.predictors.heads.class_head."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import keras_class_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class ConvolutionalKerasClassPredictorTest(test_case.TestCase):

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
    class_prediction_head = keras_class_head.ConvolutionalClassHead(
        is_training=True,
        num_class_slots=20,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=3,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=False,
        num_predictions_per_location=1,
        use_depthwise=False)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_predictions = class_prediction_head(image_feature,)
      return class_predictions
    class_predictions = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 20], class_predictions.shape)

  def test_prediction_size_depthwise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    class_prediction_head = keras_class_head.ConvolutionalClassHead(
        is_training=True,
        num_class_slots=20,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=3,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=False,
        num_predictions_per_location=1,
        use_depthwise=True)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_predictions = class_prediction_head(image_feature,)
      return class_predictions
    class_predictions = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 20], class_predictions.shape)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class MaskRCNNClassHeadTest(test_case.TestCase):

  def _build_fc_hyperparams(self,
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
    return hyperparams_builder.KerasLayerHyperparams(hyperparams)

  def test_prediction_size(self):
    class_prediction_head = keras_class_head.MaskRCNNClassHead(
        is_training=False,
        num_class_slots=20,
        fc_hyperparams=self._build_fc_hyperparams(),
        freeze_batchnorm=False,
        use_dropout=True,
        dropout_keep_prob=0.5)
    def graph_fn():
      roi_pooled_features = tf.random_uniform(
          [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      prediction = class_prediction_head(roi_pooled_features)
      return prediction
    prediction = self.execute(graph_fn, [])
    self.assertAllEqual([64, 1, 20], prediction.shape)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class WeightSharedConvolutionalKerasClassPredictorTest(test_case.TestCase):

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
    class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(
        num_class_slots=20,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=False)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_predictions = class_prediction_head(image_feature)
      return class_predictions
    class_predictions = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 20], class_predictions.shape)

  def test_prediction_size_depthwise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(
        num_class_slots=20,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=1,
        use_depthwise=True)
    def graph_fn():
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_predictions = class_prediction_head(image_feature)
      return class_predictions
    class_predictions = self.execute(graph_fn, [])
    self.assertAllEqual([64, 323, 20], class_predictions.shape)

  def test_variable_count_depth_wise_true(self):
    conv_hyperparams = self._build_conv_hyperparams()
    class_prediction_head = (
        keras_class_head.WeightSharedConvolutionalClassHead(
            num_class_slots=20,
            conv_hyperparams=conv_hyperparams,
            num_predictions_per_location=1,
            use_depthwise=True))
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    class_prediction_head(image_feature)
    self.assertEqual(len(class_prediction_head.variables), 3)

  def test_variable_count_depth_wise_False(self):
    conv_hyperparams = self._build_conv_hyperparams()
    class_prediction_head = (
        keras_class_head.WeightSharedConvolutionalClassHead(
            num_class_slots=20,
            conv_hyperparams=conv_hyperparams,
            num_predictions_per_location=1,
            use_depthwise=False))
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    class_prediction_head(image_feature)
    self.assertEqual(len(class_prediction_head.variables), 2)

  def test_softmax_score_converter(self):
    num_class_slots = 10
    batch_size = 2
    height = 17
    width = 19
    num_predictions_per_location = 2
    assert num_predictions_per_location != 1

    conv_hyperparams = self._build_conv_hyperparams()
    class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(
        num_class_slots=num_class_slots,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=num_predictions_per_location,
        score_converter_fn=tf.nn.softmax)

    def graph_fn():
      image_feature = tf.random_uniform([batch_size, height, width, 1024],
                                        minval=-10.0,
                                        maxval=10.0,
                                        dtype=tf.float32)
      class_predictions = class_prediction_head(image_feature)
      return class_predictions

    class_predictions_out = self.execute(graph_fn, [])
    class_predictions_sum = np.sum(class_predictions_out, axis=-1)
    num_anchors = height * width * num_predictions_per_location
    exp_class_predictions_sum = np.ones((batch_size, num_anchors),
                                        dtype=np.float32)
    self.assertAllEqual((batch_size, num_anchors, num_class_slots),
                        class_predictions_out.shape)
    self.assertAllClose(class_predictions_sum, exp_class_predictions_sum)


if __name__ == '__main__':
  tf.test.main()
