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

"""Tests for object_detection.predictors.convolutional_keras_box_predictor."""
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import convolutional_keras_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


class ConvolutionalKerasBoxPredictorTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
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

  def test_get_boxes_for_five_aspect_ratios_per_location(self):
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5],
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4
          ))
      box_predictions = conv_box_predictor([image_features])
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      objectness_predictions = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, objectness_predictions)
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, objectness_predictions) = self.execute(graph_fn,
                                                           [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 320, 1, 4])
    self.assertAllEqual(objectness_predictions.shape, [4, 320, 1])

  def test_get_boxes_for_one_aspect_ratio_per_location(self):
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[1],
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4
          ))
      box_predictions = conv_box_predictor([image_features])
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      objectness_predictions = tf.concat(box_predictions[
          box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
      return (box_encodings, objectness_predictions)
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, objectness_predictions) = self.execute(graph_fn,
                                                           [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 64, 1, 4])
    self.assertAllEqual(objectness_predictions.shape, [4, 64, 1])

  def test_get_multi_class_predictions_for_five_aspect_ratios_per_location(
      self):
    num_classes_without_background = 6
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5],
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4
          ))
      box_predictions = conv_box_predictor([image_features])
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)
    (box_encodings,
     class_predictions_with_background) = self.execute(graph_fn,
                                                       [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 320, 1, 4])
    self.assertAllEqual(class_predictions_with_background.shape,
                        [4, 320, num_classes_without_background+1])

  def test_get_predictions_with_feature_maps_of_dynamic_shape(
      self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = (
        box_predictor_builder.build_convolutional_keras_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams=self._build_conv_hyperparams(),
            freeze_batchnorm=False,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=[5],
            min_depth=0,
            max_depth=32,
            num_layers_before_predictor=1,
            use_dropout=True,
            dropout_keep_prob=0.8,
            kernel_size=1,
            box_code_size=4
        ))
    box_predictions = conv_box_predictor([image_features])
    box_encodings = tf.concat(
        box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
    objectness_predictions = tf.concat(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)
    init_op = tf.global_variables_initializer()

    resolution = 32
    expected_num_anchors = resolution*resolution*5
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape,
       objectness_predictions_shape) = sess.run(
           [tf.shape(box_encodings), tf.shape(objectness_predictions)],
           feed_dict={image_features:
                      np.random.rand(4, resolution, resolution, 64)})
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
      self.assertAllEqual(box_encodings_shape, [4, expected_num_anchors, 1, 4])
      self.assertAllEqual(objectness_predictions_shape,
                          [4, expected_num_anchors, 1])
    expected_variable_set = set([
        'BoxPredictor/PreHeadConvolutions_0/Conv2d_0_1x1_32/bias',
        'BoxPredictor/PreHeadConvolutions_0/Conv2d_0_1x1_32/kernel',
        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/bias',
        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/kernel',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/bias',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/kernel'])
    self.assertEqual(expected_variable_set, actual_variable_set)

  # TODO(kaftan): Remove conditional after CMLE moves to TF 1.10

if __name__ == '__main__':
  tf.test.main()
