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
from object_detection.predictors.heads import keras_box_head
from object_detection.predictors.heads import keras_class_head
from object_detection.predictors.heads import keras_mask_head
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
        'BoxPredictor/SharedConvolutions_0/Conv2d_0_1x1_32/bias',
        'BoxPredictor/SharedConvolutions_0/Conv2d_0_1x1_32/kernel',
        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/bias',
        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/kernel',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/bias',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/kernel'])
    self.assertEqual(expected_variable_set, actual_variable_set)
    self.assertEqual(conv_box_predictor._sorted_head_names,
                     ['box_encodings', 'class_predictions_with_background'])

  def test_use_depthwise_convolution(self):
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
            box_code_size=4,
            use_depthwise=True
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
        'BoxPredictor/SharedConvolutions_0/Conv2d_0_1x1_32/bias',
        'BoxPredictor/SharedConvolutions_0/Conv2d_0_1x1_32/kernel',

        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor_depthwise/'
        'bias',

        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor_depthwise/'
        'depthwise_kernel',

        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/bias',
        'BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/kernel',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor_depthwise/bias',

        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor_depthwise/'
        'depthwise_kernel',

        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/bias',
        'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/kernel'])
    self.assertEqual(expected_variable_set, actual_variable_set)
    self.assertEqual(conv_box_predictor._sorted_head_names,
                     ['box_encodings', 'class_predictions_with_background'])


class WeightSharedConvolutionalKerasBoxPredictorTest(test_case.TestCase):

  def _build_conv_hyperparams(self, add_batch_norm=True):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
          mean: 0.0
        }
      }
    """
    if add_batch_norm:
      batch_norm_proto = """
        batch_norm {
          train: true,
        }
      """
      conv_hyperparams_text_proto += batch_norm_proto
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  # pylint: disable=line-too-long
  def test_get_boxes_for_five_aspect_ratios_per_location(self):

    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5],
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor([image_features])
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
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=True,
              num_classes=2,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5],
              depth=32,
              num_layers_before_predictor=1,
              class_prediction_bias_init=-4.6,
              box_code_size=4))
      box_predictions = conv_box_predictor([image_features])
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
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5],
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor([image_features])
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
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor([image_features1, image_features2])
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
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5, 5],
              depth=32,
              num_layers_before_predictor=1,
              box_code_size=4))
      box_predictions = conv_box_predictor(
          [image_features1, image_features2, image_features3])
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
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4))
      box_predictions = conv_box_predictor([image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction tower
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_multiple_feature_maps_share_weights_without_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor([image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/bias'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction tower
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/bias'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_multiple_feature_maps_share_weights_with_depthwise(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(
                  add_batch_norm=False),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False,
              use_depthwise=True))
      box_predictions = conv_box_predictor([image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/bias'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction tower
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/bias'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/depthwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/pointwise_kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_no_batchnorm_params_when_batchnorm_is_not_configured(self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(
                  add_batch_norm=False),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor(
          [image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'BoxPredictionTower/conv2d_1/bias'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction tower
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'ClassPredictionTower/conv2d_1/bias'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_share_weights_share_tower_separate_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              share_prediction_tower=True))
      box_predictions = conv_box_predictor(
          [image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/BatchNorm/feature_1/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/BatchNorm/feature_0/beta'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/BatchNorm/feature_1/beta'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])
    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_predictions_share_weights_share_tower_without_batchnorm(
      self):
    num_classes_without_background = 6
    def graph_fn(image_features1, image_features2):
      conv_box_predictor = (
          box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
              is_training=False,
              num_classes=num_classes_without_background,
              conv_hyperparams=self._build_conv_hyperparams(
                  add_batch_norm=False),
              freeze_batchnorm=False,
              inplace_batchnorm_update=False,
              num_predictions_per_location_list=[5, 5],
              depth=32,
              num_layers_before_predictor=2,
              box_code_size=4,
              share_prediction_tower=True,
              apply_batch_norm=False))
      box_predictions = conv_box_predictor(
          [image_features1, image_features2])
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
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_0/bias'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'PredictionTower/conv2d_1/bias'),
        # Box prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalBoxHead/BoxPredictor/bias'),
        # Class prediction head
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/kernel'),
        ('WeightSharedConvolutionalBoxPredictor/'
         'WeightSharedConvolutionalClassHead/ClassPredictor/bias')])

    self.assertEqual(expected_variable_set, actual_variable_set)

  def test_get_predictions_with_feature_maps_of_dynamic_shape(
      self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = (
        box_predictor_builder.build_weight_shared_convolutional_keras_box_predictor(
            is_training=False,
            num_classes=0,
            conv_hyperparams=self._build_conv_hyperparams(),
            freeze_batchnorm=False,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=[5],
            depth=32,
            num_layers_before_predictor=1,
            box_code_size=4))
    box_predictions = conv_box_predictor([image_features])
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
      box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
          box_code_size=box_code_size,
          conv_hyperparams=self._build_conv_hyperparams(),
          num_predictions_per_location=num_predictions_per_location)
      class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(
          num_class_slots=num_classes_without_background + 1,
          conv_hyperparams=self._build_conv_hyperparams(),
          num_predictions_per_location=num_predictions_per_location)
      other_heads = {
          other_head_name:
              keras_mask_head.WeightSharedConvolutionalMaskHead(
                  num_classes=num_classes_without_background,
                  conv_hyperparams=self._build_conv_hyperparams(),
                  num_predictions_per_location=num_predictions_per_location,
                  mask_height=mask_height,
                  mask_width=mask_width)
      }

      conv_box_predictor = box_predictor.WeightSharedConvolutionalBoxPredictor(
          is_training=False,
          num_classes=num_classes_without_background,
          box_prediction_head=box_prediction_head,
          class_prediction_head=class_prediction_head,
          other_heads=other_heads,
          conv_hyperparams=self._build_conv_hyperparams(),
          freeze_batchnorm=False,
          inplace_batchnorm_update=False,
          depth=32,
          num_layers_before_predictor=2)
      box_predictions = conv_box_predictor([image_features])
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
