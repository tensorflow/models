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

"""Tests for object_detection.predictors.rfcn_box_predictor."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors import rfcn_keras_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class RfcnKerasBoxPredictorTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
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
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def test_get_correct_box_encoding_and_class_prediction_shapes(self):
    rfcn_box_predictor = box_predictor.RfcnKerasBoxPredictor(
        is_training=False,
        num_classes=2,
        conv_hyperparams=self._build_conv_hyperparams(),
        freeze_batchnorm=False,
        num_spatial_bins=[3, 3],
        depth=4,
        crop_size=[12, 12],
        box_code_size=4)
    def graph_fn(image_features, proposal_boxes):

      box_predictions = rfcn_box_predictor(
          [image_features],
          proposal_boxes=proposal_boxes)
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      class_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, class_predictions_with_background)

    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    proposal_boxes = np.random.rand(4, 2, 4).astype(np.float32)
    (box_encodings, class_predictions_with_background) = self.execute(
        graph_fn, [image_features, proposal_boxes])

    self.assertAllEqual(box_encodings.shape, [8, 1, 2, 4])
    self.assertAllEqual(class_predictions_with_background.shape, [8, 1, 3])


if __name__ == '__main__':
  tf.test.main()
