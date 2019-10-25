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
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import class_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


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


if __name__ == '__main__':
  tf.test.main()
