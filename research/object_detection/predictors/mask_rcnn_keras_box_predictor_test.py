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

"""Tests for object_detection.predictors.mask_rcnn_box_predictor."""
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import mask_rcnn_keras_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


class MaskRCNNKerasBoxPredictorTest(test_case.TestCase):

  def _build_hyperparams(self,
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

  def test_get_boxes_with_five_classes(self):
    def graph_fn(image_features):
      mask_box_predictor = (
          box_predictor_builder.build_mask_rcnn_keras_box_predictor(
              is_training=False,
              num_classes=5,
              fc_hyperparams=self._build_hyperparams(),
              freeze_batchnorm=False,
              use_dropout=False,
              dropout_keep_prob=0.5,
              box_code_size=4,
          ))
      box_predictions = mask_box_predictor(
          [image_features],
          prediction_stage=2)
      return (box_predictions[box_predictor.BOX_ENCODINGS],
              box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND])
    image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
    (box_encodings,
     class_predictions_with_background) = self.execute(graph_fn,
                                                       [image_features])
    self.assertAllEqual(box_encodings.shape, [2, 1, 5, 4])
    self.assertAllEqual(class_predictions_with_background.shape, [2, 1, 6])

  def test_get_boxes_with_five_classes_share_box_across_classes(self):
    def graph_fn(image_features):
      mask_box_predictor = (
          box_predictor_builder.build_mask_rcnn_keras_box_predictor(
              is_training=False,
              num_classes=5,
              fc_hyperparams=self._build_hyperparams(),
              freeze_batchnorm=False,
              use_dropout=False,
              dropout_keep_prob=0.5,
              box_code_size=4,
              share_box_across_classes=True
          ))
      box_predictions = mask_box_predictor(
          [image_features],
          prediction_stage=2)
      return (box_predictions[box_predictor.BOX_ENCODINGS],
              box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND])
    image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
    (box_encodings,
     class_predictions_with_background) = self.execute(graph_fn,
                                                       [image_features])
    self.assertAllEqual(box_encodings.shape, [2, 1, 1, 4])
    self.assertAllEqual(class_predictions_with_background.shape, [2, 1, 6])

  def test_get_instance_masks(self):
    def graph_fn(image_features):
      mask_box_predictor = (
          box_predictor_builder.build_mask_rcnn_keras_box_predictor(
              is_training=False,
              num_classes=5,
              fc_hyperparams=self._build_hyperparams(),
              freeze_batchnorm=False,
              use_dropout=False,
              dropout_keep_prob=0.5,
              box_code_size=4,
              conv_hyperparams=self._build_hyperparams(
                  op_type=hyperparams_pb2.Hyperparams.CONV),
              predict_instance_masks=True))
      box_predictions = mask_box_predictor(
          [image_features],
          prediction_stage=3)
      return (box_predictions[box_predictor.MASK_PREDICTIONS],)
    image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
    mask_predictions = self.execute(graph_fn, [image_features])
    self.assertAllEqual(mask_predictions.shape, [2, 1, 5, 14, 14])

  def test_do_not_return_instance_masks_without_request(self):
    image_features = tf.random_uniform([2, 7, 7, 3], dtype=tf.float32)
    mask_box_predictor = (
        box_predictor_builder.build_mask_rcnn_keras_box_predictor(
            is_training=False,
            num_classes=5,
            fc_hyperparams=self._build_hyperparams(),
            freeze_batchnorm=False,
            use_dropout=False,
            dropout_keep_prob=0.5,
            box_code_size=4))
    box_predictions = mask_box_predictor(
        [image_features],
        prediction_stage=2)
    self.assertEqual(len(box_predictions), 2)
    self.assertTrue(box_predictor.BOX_ENCODINGS in box_predictions)
    self.assertTrue(box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND
                    in box_predictions)


if __name__ == '__main__':
  tf.test.main()
