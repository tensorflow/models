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

"""Tests for object_detection.predictors.heads.class_head."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import class_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class MaskRCNNClassHeadTest(test_case.TestCase):

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
    class_prediction_head = class_head.MaskRCNNClassHead(
        is_training=False,
        num_class_slots=20,
        fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
        use_dropout=True,
        dropout_keep_prob=0.5)
    roi_pooled_features = tf.random_uniform(
        [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    prediction = class_prediction_head.predict(
        features=roi_pooled_features, num_predictions_per_location=1)
    self.assertAllEqual([64, 1, 20], prediction.get_shape().as_list())

  def test_scope_name(self):
    expected_var_names = set([
        """ClassPredictor/weights""",
        """ClassPredictor/biases"""
    ])

    g = tf.Graph()
    with g.as_default():
      class_prediction_head = class_head.MaskRCNNClassHead(
          is_training=True,
          num_class_slots=20,
          fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
          use_dropout=True,
          dropout_keep_prob=0.5)
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_prediction_head.predict(
          features=image_feature,
          num_predictions_per_location=1)
      actual_variable_set = set([
          var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      ])
      self.assertSetEqual(expected_var_names, actual_variable_set)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ConvolutionalClassPredictorTest(test_case.TestCase):

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
    class_prediction_head = class_head.ConvolutionalClassHead(
        is_training=True,
        num_class_slots=20,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=3)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    class_predictions = class_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 20],
                        class_predictions.get_shape().as_list())

  def test_scope_name(self):
    expected_var_names = set([
        """ClassPredictor/weights""",
        """ClassPredictor/biases"""
    ])
    g = tf.Graph()
    with g.as_default():
      class_prediction_head = class_head.ConvolutionalClassHead(
          is_training=True,
          num_class_slots=20,
          use_dropout=True,
          dropout_keep_prob=0.5,
          kernel_size=3)
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_prediction_head.predict(
          features=image_feature,
          num_predictions_per_location=1)
      actual_variable_set = set([
          var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      ])
      self.assertSetEqual(expected_var_names, actual_variable_set)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class WeightSharedConvolutionalClassPredictorTest(test_case.TestCase):

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
    class_prediction_head = (
        class_head.WeightSharedConvolutionalClassHead(num_class_slots=20))
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    class_predictions = class_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

  def test_scope_name(self):
    expected_var_names = set([
        """ClassPredictor/weights""",
        """ClassPredictor/biases"""
    ])
    g = tf.Graph()
    with g.as_default():
      class_prediction_head = class_head.WeightSharedConvolutionalClassHead(
          num_class_slots=20)
      image_feature = tf.random_uniform(
          [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
      class_prediction_head.predict(
          features=image_feature,
          num_predictions_per_location=1)
      actual_variable_set = set([
          var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      ])
      self.assertSetEqual(expected_var_names, actual_variable_set)

  def test_softmax_score_converter(self):
    num_class_slots = 10
    batch_size = 2
    height = 17
    width = 19
    num_predictions_per_location = 2
    assert num_predictions_per_location != 1

    def graph_fn():
      class_prediction_head = (
          class_head.WeightSharedConvolutionalClassHead(
              num_class_slots=num_class_slots,
              score_converter_fn=tf.nn.softmax))
      image_feature = tf.random_uniform([batch_size, height, width, 1024],
                                        minval=-10.0,
                                        maxval=10.0,
                                        dtype=tf.float32)
      class_predictions = class_prediction_head.predict(
          features=image_feature,
          num_predictions_per_location=num_predictions_per_location)
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
