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

"""Tests for object_detection.core.box_predictor."""

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.core import box_predictor
from object_detection.protos import hyperparams_pb2


class MaskRCNNBoxPredictorTest(tf.test.TestCase):

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

  def test_get_boxes_with_five_classes(self):
    image_features = tf.random_uniform([2, 7, 7, 3], dtype=tf.float32)
    mask_box_predictor = box_predictor.MaskRCNNBoxPredictor(
        is_training=False,
        num_classes=5,
        fc_hyperparams=self._build_arg_scope_with_hyperparams(),
        use_dropout=False,
        dropout_keep_prob=0.5,
        box_code_size=4,
    )
    box_predictions = mask_box_predictor.predict(
        image_features, num_predictions_per_location=1, scope='BoxPredictor')
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    class_predictions_with_background = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape,
       class_predictions_with_background_shape) = sess.run(
           [tf.shape(box_encodings),
            tf.shape(class_predictions_with_background)])
      self.assertAllEqual(box_encodings_shape, [2, 1, 5, 4])
      self.assertAllEqual(class_predictions_with_background_shape, [2, 1, 6])

  def test_value_error_on_predict_instance_masks_with_no_conv_hyperparms(self):
    with self.assertRaises(ValueError):
      box_predictor.MaskRCNNBoxPredictor(
          is_training=False,
          num_classes=5,
          fc_hyperparams=self._build_arg_scope_with_hyperparams(),
          use_dropout=False,
          dropout_keep_prob=0.5,
          box_code_size=4,
          predict_instance_masks=True)

  def test_get_instance_masks(self):
    image_features = tf.random_uniform([2, 7, 7, 3], dtype=tf.float32)
    mask_box_predictor = box_predictor.MaskRCNNBoxPredictor(
        is_training=False,
        num_classes=5,
        fc_hyperparams=self._build_arg_scope_with_hyperparams(),
        use_dropout=False,
        dropout_keep_prob=0.5,
        box_code_size=4,
        conv_hyperparams=self._build_arg_scope_with_hyperparams(
            op_type=hyperparams_pb2.Hyperparams.CONV),
        predict_instance_masks=True)
    box_predictions = mask_box_predictor.predict(
        image_features, num_predictions_per_location=1, scope='BoxPredictor')
    mask_predictions = box_predictions[box_predictor.MASK_PREDICTIONS]
    self.assertListEqual([2, 1, 5, 14, 14],
                         mask_predictions.get_shape().as_list())

  def test_do_not_return_instance_masks_and_keypoints_without_request(self):
    image_features = tf.random_uniform([2, 7, 7, 3], dtype=tf.float32)
    mask_box_predictor = box_predictor.MaskRCNNBoxPredictor(
        is_training=False,
        num_classes=5,
        fc_hyperparams=self._build_arg_scope_with_hyperparams(),
        use_dropout=False,
        dropout_keep_prob=0.5,
        box_code_size=4)
    box_predictions = mask_box_predictor.predict(
        image_features, num_predictions_per_location=1, scope='BoxPredictor')
    self.assertEqual(len(box_predictions), 2)
    self.assertTrue(box_predictor.BOX_ENCODINGS in box_predictions)
    self.assertTrue(box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND
                    in box_predictions)

  def test_value_error_on_predict_keypoints(self):
    with self.assertRaises(ValueError):
      box_predictor.MaskRCNNBoxPredictor(
          is_training=False,
          num_classes=5,
          fc_hyperparams=self._build_arg_scope_with_hyperparams(),
          use_dropout=False,
          dropout_keep_prob=0.5,
          box_code_size=4,
          predict_keypoints=True)


class RfcnBoxPredictorTest(tf.test.TestCase):

  def _build_arg_scope_with_conv_hyperparams(self):
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
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.build(conv_hyperparams, is_training=True)

  def test_get_correct_box_encoding_and_class_prediction_shapes(self):
    image_features = tf.random_uniform([4, 8, 8, 64], dtype=tf.float32)
    proposal_boxes = tf.random_normal([4, 2, 4], dtype=tf.float32)
    rfcn_box_predictor = box_predictor.RfcnBoxPredictor(
        is_training=False,
        num_classes=2,
        conv_hyperparams=self._build_arg_scope_with_conv_hyperparams(),
        num_spatial_bins=[3, 3],
        depth=4,
        crop_size=[12, 12],
        box_code_size=4
    )
    box_predictions = rfcn_box_predictor.predict(
        image_features, num_predictions_per_location=1, scope='BoxPredictor',
        proposal_boxes=proposal_boxes)
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    class_predictions_with_background = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape,
       class_predictions_shape) = sess.run(
           [tf.shape(box_encodings),
            tf.shape(class_predictions_with_background)])
      self.assertAllEqual(box_encodings_shape, [8, 1, 2, 4])
      self.assertAllEqual(class_predictions_shape, [8, 1, 3])


class ConvolutionalBoxPredictorTest(tf.test.TestCase):

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
    image_features = tf.random_uniform([4, 8, 8, 64], dtype=tf.float32)
    conv_box_predictor = box_predictor.ConvolutionalBoxPredictor(
        is_training=False,
        num_classes=0,
        conv_hyperparams=self._build_arg_scope_with_conv_hyperparams(),
        min_depth=0,
        max_depth=32,
        num_layers_before_predictor=1,
        use_dropout=True,
        dropout_keep_prob=0.8,
        kernel_size=1,
        box_code_size=4
    )
    box_predictions = conv_box_predictor.predict(
        image_features, num_predictions_per_location=5, scope='BoxPredictor')
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    objectness_predictions = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape,
       objectness_predictions_shape) = sess.run(
           [tf.shape(box_encodings), tf.shape(objectness_predictions)])
      self.assertAllEqual(box_encodings_shape, [4, 320, 1, 4])
      self.assertAllEqual(objectness_predictions_shape, [4, 320, 1])

  def test_get_boxes_for_one_aspect_ratio_per_location(self):
    image_features = tf.random_uniform([4, 8, 8, 64], dtype=tf.float32)
    conv_box_predictor = box_predictor.ConvolutionalBoxPredictor(
        is_training=False,
        num_classes=0,
        conv_hyperparams=self._build_arg_scope_with_conv_hyperparams(),
        min_depth=0,
        max_depth=32,
        num_layers_before_predictor=1,
        use_dropout=True,
        dropout_keep_prob=0.8,
        kernel_size=1,
        box_code_size=4
    )
    box_predictions = conv_box_predictor.predict(
        image_features, num_predictions_per_location=1, scope='BoxPredictor')
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    objectness_predictions = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape,
       objectness_predictions_shape) = sess.run(
           [tf.shape(box_encodings), tf.shape(objectness_predictions)])
      self.assertAllEqual(box_encodings_shape, [4, 64, 1, 4])
      self.assertAllEqual(objectness_predictions_shape, [4, 64, 1])

  def test_get_multi_class_predictions_for_five_aspect_ratios_per_location(
      self):
    num_classes_without_background = 6
    image_features = tf.random_uniform([4, 8, 8, 64], dtype=tf.float32)
    conv_box_predictor = box_predictor.ConvolutionalBoxPredictor(
        is_training=False,
        num_classes=num_classes_without_background,
        conv_hyperparams=self._build_arg_scope_with_conv_hyperparams(),
        min_depth=0,
        max_depth=32,
        num_layers_before_predictor=1,
        use_dropout=True,
        dropout_keep_prob=0.8,
        kernel_size=1,
        box_code_size=4
    )
    box_predictions = conv_box_predictor.predict(
        image_features,
        num_predictions_per_location=5,
        scope='BoxPredictor')
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    class_predictions_with_background = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      (box_encodings_shape, class_predictions_with_background_shape
      ) = sess.run([
          tf.shape(box_encodings), tf.shape(class_predictions_with_background)])
      self.assertAllEqual(box_encodings_shape, [4, 320, 1, 4])
      self.assertAllEqual(class_predictions_with_background_shape,
                          [4, 320, num_classes_without_background+1])

  def test_get_boxes_for_five_aspect_ratios_per_location_fully_convolutional(
      self):
    image_features = tf.placeholder(dtype=tf.float32, shape=[4, None, None, 64])
    conv_box_predictor = box_predictor.ConvolutionalBoxPredictor(
        is_training=False,
        num_classes=0,
        conv_hyperparams=self._build_arg_scope_with_conv_hyperparams(),
        min_depth=0,
        max_depth=32,
        num_layers_before_predictor=1,
        use_dropout=True,
        dropout_keep_prob=0.8,
        kernel_size=1,
        box_code_size=4
    )
    box_predictions = conv_box_predictor.predict(
        image_features, num_predictions_per_location=5, scope='BoxPredictor')
    box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
    objectness_predictions = box_predictions[
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]
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
      self.assertAllEqual(box_encodings_shape, [4, expected_num_anchors, 1, 4])
      self.assertAllEqual(objectness_predictions_shape,
                          [4, expected_num_anchors, 1])


if __name__ == '__main__':
  tf.test.main()
