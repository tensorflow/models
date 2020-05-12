# Lint as: python2, python3
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

"""Tests for object_detection.predictors.convolutional_box_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import convolutional_box_predictor as box_predictor
from object_detection.predictors.heads import box_head
from object_detection.predictors.heads import class_head
from object_detection.predictors.heads import mask_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


class ConvolutionalBoxPredictorTest(test_case.TestCase):

  def _build_arg_scope_with_conv_hyperparams(self):
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
    return hyperparams_builder.build(conv_hyperparams, is_training=True)

  def test_get_boxes_for_five_aspect_ratios_per_location(self):
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_convolutional_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features], num_predictions_per_location=[5],
          scope='BoxPredictor')
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
          box_predictor_builder.build_convolutional_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features], num_predictions_per_location=[1],
          scope='BoxPredictor')
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
          box_predictor_builder.build_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features],
          num_predictions_per_location=[5],
          scope='BoxPredictor')
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
        box_predictor_builder.build_convolutional_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
            min_depth=0,
            max_depth=32,
            num_layers_before_predictor=1,
            use_dropout=True,
            dropout_keep_prob=0.8,
            kernel_size=1,
            box_code_size=4))
    box_predictions = conv_box_predictor.predict(
        [image_features], num_predictions_per_location=[5],
        scope='BoxPredictor')
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
        'BoxPredictor/Conv2d_0_1x1_32/biases',
        'BoxPredictor/Conv2d_0_1x1_32/weights',
        'BoxPredictor/BoxEncodingPredictor/biases',
        'BoxPredictor/BoxEncodingPredictor/weights',
        'BoxPredictor/ClassPredictor/biases',
        'BoxPredictor/ClassPredictor/weights'])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_use_depthwise_convolution(self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = (
        box_predictor_builder.build_convolutional_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
            min_depth=0,
            max_depth=32,
            num_layers_before_predictor=1,
            dropout_keep_prob=0.8,
            kernel_size=3,
            box_code_size=4,
            use_dropout=True,
            use_depthwise=True))
    box_predictions = conv_box_predictor.predict(
        [image_features], num_predictions_per_location=[5],
        scope='BoxPredictor')
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
        'BoxPredictor/Conv2d_0_1x1_32/biases',
        'BoxPredictor/Conv2d_0_1x1_32/weights',
        'BoxPredictor/BoxEncodingPredictor_depthwise/biases',
        'BoxPredictor/BoxEncodingPredictor_depthwise/depthwise_weights',
        'BoxPredictor/BoxEncodingPredictor/biases',
        'BoxPredictor/BoxEncodingPredictor/weights',
        'BoxPredictor/ClassPredictor_depthwise/biases',
        'BoxPredictor/ClassPredictor_depthwise/depthwise_weights',
        'BoxPredictor/ClassPredictor/biases',
        'BoxPredictor/ClassPredictor/weights'])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_no_dangling_outputs(self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = (
        box_predictor_builder.build_convolutional_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
            min_depth=0,
            max_depth=32,
            num_layers_before_predictor=1,
            dropout_keep_prob=0.8,
            kernel_size=3,
            box_code_size=4,
            use_dropout=True,
            use_depthwise=True))
    box_predictions = conv_box_predictor.predict(
        [image_features], num_predictions_per_location=[5],
        scope='BoxPredictor')
    tf.concat(
        box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
    tf.concat(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)

    bad_dangling_ops = []
    types_safe_to_dangle = set(['Assign', 'Mul', 'Const'])
    for op in tf.get_default_graph().get_operations():
      if (not op.outputs) or (not op.outputs[0].consumers()):
        if 'BoxPredictor' in op.name:
          if op.type not in types_safe_to_dangle:
            bad_dangling_ops.append(op)

    self.assertEqual(bad_dangling_ops, [])


class WeightSharedConvolutionalBoxPredictorTest(test_case.TestCase):

  def _build_arg_scope_with_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        random_normal_initializer {
          stddev: 0.01
          mean: 0.0
        }
      }
      batch_norm {
        train: true,
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.build(conv_hyperparams, is_training=True)

  def _build_conv_arg_scope_no_batch_norm(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        random_normal_initializer {
          stddev: 0.01
          mean: 0.0
        }
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.build(conv_hyperparams, is_training=True)

  def test_get_boxes_for_five_aspect_ratios_per_location(self):

    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features], num_predictions_per_location=[5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      objectness_predictions = tf.concat(box_predictions[
          box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
      return (box_encodings, objectness_predictions)
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, objectness_predictions) = self.execute(
        graph_fn, [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 320, 4])
    self.assertAllEqual(objectness_predictions.shape, [4, 320, 1])

  def test_bias_predictions_to_background_with_sigmoid_score_conversion(self):

    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=True,
              num_classes=2,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=1,
              class_prediction_bias_init=-4.6,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features], num_predictions_per_location=[5],
          scope='BoxPredictor')
      class_predictions = tf.concat(box_predictions[
          box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
      return (tf.nn.sigmoid(class_predictions),)
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    class_predictions = self.execute(graph_fn, [image_features])
    self.assertAlmostEqual(np.mean(class_predictions), 0.01, places=3)

  def test_get_multi_class_predictions_for_five_aspect_ratios_per_location(
      self):

    num_classes_without_background = 6
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features],
          num_predictions_per_location=[5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(box_predictions[
          box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
      return (box_encodings, class_predictions_with_background)

    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, class_predictions_with_background) = self.execute(
        graph_fn, [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 320, 4])
    self.assertAllEqual(class_predictions_with_background.shape,
                        [4, 320, num_classes_without_background+1])

  def test_get_multi_class_predictions_from_two_feature_maps(
      self):

    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    image_features1 = np.random.rand(4, 8, 8, 64).astype(np.float32)
    image_features2 = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, class_predictions_with_background) = self.execute(
        graph_fn, [image_features1, image_features2])
    self.assertAllEqual(box_encodings.shape, [4, 640, 4])
    self.assertAllEqual(class_predictions_with_background.shape,
                        [4, 640, num_classes_without_background+1])

  def test_get_multi_class_predictions_from_feature_maps_of_different_depth(
      self):

    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2, image_features3):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2, image_features3],
          num_predictions_per_location=[5, 5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    image_features1 = np.random.rand(4, 8, 8, 64).astype(np.float32)
    image_features2 = np.random.rand(4, 8, 8, 64).astype(np.float32)
    image_features3 = np.random.rand(4, 8, 8, 32).astype(np.float32)
    (box_encodings, class_predictions_with_background) = self.execute(
        graph_fn, [image_features1, image_features2, image_features3])
    self.assertAllEqual(box_encodings.shape, [4, 960, 4])
    self.assertAllEqual(class_predictions_with_background.shape,
                        [4, 960, num_classes_without_background+1])

  def test_predictions_multiple_feature_maps_share_weights_separate_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Box prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_multiple_feature_maps_share_weights_without_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Box prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/biases'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/biases'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_multiple_feature_maps_share_weights_with_depthwise(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False,
              use_depthwise=True))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Box prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/biases'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/biases'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/depthwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/pointwise_weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_no_batchnorm_params_when_batchnorm_is_not_configured(self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_conv_arg_scope_no_batch_norm(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Box prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/biases'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/biases'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_share_weights_share_tower_separate_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              share_prediction_tower=True))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Shared prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_share_weights_share_tower_without_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              share_prediction_tower=True,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor.predict(
          [image_features1, image_features2],
          num_predictions_per_location=[5, 5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    with self.test_session(graph=tf.Graph()):
      graph_fn(tf.random_uniform([4, 32, 32, 3], dtype=tf.float32),
               tf.random_uniform([4, 16, 16, 3], dtype=tf.float32))
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
    expected_variable_set = set([
        # Shared prediction tower
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/biases'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/biases'),
        # Box prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictor/biases'),
        # Class prediction head
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/weights'),
        ('BoxPredictor/WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictor/biases')])

    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_get_predictions_with_feature_maps_of_dynamic_shape(
      self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = (
        box_predictor_builder.build_weight_shared_convolutional_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
            depth=32,
            num_layers_before_predictor=1,
            box_code_size=4))
    box_predictions = conv_box_predictor.predict(
        [image_features], num_predictions_per_location=[5],
        scope='BoxPredictor')
    box_encodings = tf.concat(box_predictions[box_predictor.BOX_ENCODINGS],
                              axis=1)
    objectness_predictions = tf.concat(box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
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
      self.assertAllEqual(box_encodings_shape, [4, expected_num_anchors, 4])
      self.assertAllEqual(objectness_predictions_shape,
                          [4, expected_num_anchors, 1])

  def test_other_heads_predictions(self):
    box_code_size = 4
    num_classes_without_background = 3
    other_head_name = 'Mask'
    mask_height = 5
    mask_width = 5
    num_predictions_per_location = 5

    def graph_fn(image_features):
      box_prediction_head = box_head.WeightSharedConvolutionalBoxHead(
          box_code_size)
      class_prediction_head = class_head.WeightSharedConvolutionalClassHead(
          num_classes_without_background + 1)
      other_heads = {
          other_head_name:
              mask_head.WeightSharedConvolutionalMaskHead(
                  num_classes_without_background,
                  mask_height=mask_height,
                  mask_width=mask_width)
      }
      conv_box_predictor = box_predictor.WeightSharedConvolutionalBoxPredictor(
          is_training=False,
          num_classes=num_classes_without_background,
          box_prediction_head=box_prediction_head,
          class_prediction_head=class_prediction_head,
          other_heads=other_heads,
          conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
          depth=32,
          num_layers_before_predictor=2)
      box_predictions = conv_box_predictor.predict(
          [image_features],
          num_predictions_per_location=[num_predictions_per_location],
          scope='BoxPredictor')
      for key, value in box_predictions.items():
        box_predictions[key] = tf.concat(value, axis=1)
      assert len(box_predictions) == 3
      return (box_predictions[box_predictor.BOX_ENCODINGS],
              box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
              box_predictions[other_head_name])

    batch_size = 4
    feature_ht = 8
    feature_wt = 8
    image_features = np.random.rand(batch_size, feature_ht, feature_wt,
                                    64).astype(np.float32)
    (box_encodings, class_predictions, other_head_predictions) = self.execute(
        graph_fn, [image_features])
    num_anchors = feature_ht * feature_wt * num_predictions_per_location
    self.assertAllEqual(box_encodings.shape,
                        [batch_size, num_anchors, box_code_size])
    self.assertAllEqual(
        class_predictions.shape,
        [batch_size, num_anchors, num_classes_without_background + 1])
    self.assertAllEqual(other_head_predictions.shape, [
        batch_size, num_anchors, num_classes_without_background, mask_height,
        mask_width
    ])




if __name__ == '__main__':
  tf.test.main()
