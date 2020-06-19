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
from object_detection.predictors.heads import box_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class MaskRCNNBoxHeadTest(test_case.TestCase):

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
    box_prediction_head = box_head.MaskRCNNBoxHead(
        is_training=False,
        num_classes=20,
        fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
        use_dropout=True,
        dropout_keep_prob=0.5,
        box_code_size=4,
        share_box_across_classes=False)
    roi_pooled_features = tf.random_uniform(
        [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    prediction = box_prediction_head.predict(
        features=roi_pooled_features, num_predictions_per_location=1)
    self.assertAllEqual([64, 1, 20, 4], prediction.get_shape().as_list())


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ConvolutionalBoxPredictorTest(test_case.TestCase):

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
    box_prediction_head = box_head.ConvolutionalBoxHead(
        is_training=True,
        box_code_size=4,
        kernel_size=3)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    box_encodings = box_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 1, 4], box_encodings.get_shape().as_list())


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class WeightSharedConvolutionalBoxPredictorTest(test_case.TestCase):

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
    box_prediction_head = box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=4)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    box_encodings = box_prediction_head.predict(
        features=image_feature,
        num_predictions_per_location=1)
    self.assertAllEqual([64, 323, 4], box_encodings.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
