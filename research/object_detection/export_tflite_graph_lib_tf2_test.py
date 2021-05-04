# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for export_tflite_graph_lib_tf2.py."""

from __future__ import division
import os
import unittest
import six

import tensorflow.compat.v2 as tf

from object_detection import export_tflite_graph_lib_tf2
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import pipeline_pb2
from object_detection.utils import tf_version
from google.protobuf import text_format

if six.PY2:
  import mock  # pylint: disable=g-importing-member,g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-importing-member,g-import-not-at-top


class FakeModel(model.DetectionModel):

  def __init__(self):
    super(FakeModel, self).__init__(num_classes=2)
    self._conv = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        strides=(1, 1),
        padding='valid',
        kernel_initializer=tf.keras.initializers.Constant(value=1.0))

  def preprocess(self, inputs):
    true_image_shapes = []  # Doesn't matter for the fake model.
    return tf.identity(inputs), true_image_shapes

  def predict(self, preprocessed_inputs, true_image_shapes):
    prediction_tensors = {'image': self._conv(preprocessed_inputs)}
    with tf.control_dependencies([prediction_tensors['image']]):
      prediction_tensors['box_encodings'] = tf.constant(
          [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]], tf.float32)
      prediction_tensors['class_predictions_with_background'] = tf.constant(
          [[[0.7, 0.6], [0.9, 0.0]]], tf.float32)
    with tf.control_dependencies([
        tf.convert_to_tensor(
            prediction_tensors['image'].get_shape().as_list()[1:3])
    ]):
      prediction_tensors['anchors'] = tf.constant(
          [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], tf.float32)
    return prediction_tensors

  def postprocess(self, prediction_dict, true_image_shapes):
    predict_tensor_sum = tf.reduce_sum(prediction_dict['image'])
    with tf.control_dependencies(list(prediction_dict.values())):
      postprocessed_tensors = {
          'detection_boxes':
              tf.constant([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]],
                           [[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]],
                          tf.float32),
          'detection_scores':
              predict_tensor_sum +
              tf.constant([[0.7, 0.6], [0.9, 0.0]], tf.float32),
          'detection_classes':
              tf.constant([[0, 1], [1, 0]], tf.float32),
          'num_detections':
              tf.constant([2, 1], tf.float32),
          'detection_keypoints':
              tf.zeros([2, 17, 2], tf.float32),
          'detection_keypoint_scores':
              tf.zeros([2, 17], tf.float32),
      }
    return postprocessed_tensors

  def restore_map(self, checkpoint_path, from_detection_checkpoint):
    pass

  def restore_from_objects(self, fine_tune_checkpoint_type):
    pass

  def loss(self, prediction_dict, true_image_shapes):
    pass

  def regularization_losses(self):
    pass

  def updates(self):
    pass


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class ExportTfLiteGraphTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self, checkpoint_dir):
    mock_model = FakeModel()
    fake_image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    preprocessed_inputs, true_image_shapes = mock_model.preprocess(fake_image)
    predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
    mock_model.postprocess(predictions, true_image_shapes)

    ckpt = tf.train.Checkpoint(model=mock_model)
    exported_checkpoint_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)

  def _get_ssd_config(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    pipeline_config.model.ssd.post_processing.batch_non_max_suppression.iou_threshold = 0.5
    return pipeline_config

  def _get_center_net_config(self):
    pipeline_config_text = """
model {
  center_net {
    num_classes: 1
    feature_extractor {
      type: "mobilenet_v2_fpn"
    }
    image_resizer {
      fixed_shape_resizer {
        height: 10
        width: 10
      }
    }
    object_detection_task {
      localization_loss {
        l1_localization_loss {
        }
      }
    }
    object_center_params {
      classification_loss {
      }
      max_box_predictions: 20
    }
    keypoint_estimation_task {
      loss {
        localization_loss {
          l1_localization_loss {
          }
        }
        classification_loss {
          penalty_reduced_logistic_focal_loss {
          }
        }
      }
    }
  }
}
    """
    return text_format.Parse(
        pipeline_config_text, pipeline_pb2.TrainEvalPipelineConfig())

  # The tf.implements signature is important since it ensures MLIR legalization,
  # so we test it here.
  def test_postprocess_implements_signature(self):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    pipeline_config = self._get_ssd_config()

    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()

      detection_model = model_builder.build(
          pipeline_config.model, is_training=False)

      ckpt = tf.train.Checkpoint(model=detection_model)
      manager = tf.train.CheckpointManager(ckpt, tmp_dir, max_to_keep=1)
      ckpt.restore(manager.latest_checkpoint).expect_partial()

      # The module helps build a TF graph appropriate for TFLite conversion.
      detection_module = export_tflite_graph_lib_tf2.SSDModule(
          pipeline_config=pipeline_config,
          detection_model=detection_model,
          max_detections=20,
          use_regular_nms=True)

      expected_signature = ('name: "TFLite_Detection_PostProcess" attr { key: '
                            '"max_detections" value { i: 20 } } attr { key: '
                            '"max_classes_per_detection" value { i: 1 } } attr '
                            '{ key: "use_regular_nms" value { b: true } } attr '
                            '{ key: "nms_score_threshold" value { f: 0.000000 }'
                            ' } attr { key: "nms_iou_threshold" value { f: '
                            '0.500000 } } attr { key: "y_scale" value { f: '
                            '10.000000 } } attr { key: "x_scale" value { f: '
                            '10.000000 } } attr { key: "h_scale" value { f: '
                            '5.000000 } } attr { key: "w_scale" value { f: '
                            '5.000000 } } attr { key: "num_classes" value { i: '
                            '2 } }')

      self.assertEqual(expected_signature,
                       detection_module.postprocess_implements_signature())

  def test_unsupported_architecture(self):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10

    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      expected_message = 'Only ssd or center_net models are supported in tflite'
      try:
        export_tflite_graph_lib_tf2.export_tflite_model(
            pipeline_config=pipeline_config,
            trained_checkpoint_dir=tmp_dir,
            output_directory=output_directory,
            max_detections=10,
            use_regular_nms=False)
      except ValueError as e:
        if expected_message not in str(e):
          raise
      else:
        raise AssertionError('Exception not raised: %s' % expected_message)

  def test_export_yields_saved_model(self):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      export_tflite_graph_lib_tf2.export_tflite_model(
          pipeline_config=self._get_ssd_config(),
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory,
          max_detections=10,
          use_regular_nms=False)
      self.assertTrue(
          os.path.exists(
              os.path.join(output_directory, 'saved_model', 'saved_model.pb')))
      self.assertTrue(
          os.path.exists(
              os.path.join(output_directory, 'saved_model', 'variables',
                           'variables.index')))
      self.assertTrue(
          os.path.exists(
              os.path.join(output_directory, 'saved_model', 'variables',
                           'variables.data-00000-of-00001')))

  def test_exported_model_inference(self):
    tmp_dir = self.get_temp_dir()
    output_directory = os.path.join(tmp_dir, 'output')
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      export_tflite_graph_lib_tf2.export_tflite_model(
          pipeline_config=self._get_ssd_config(),
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory,
          max_detections=10,
          use_regular_nms=False)

    saved_model_path = os.path.join(output_directory, 'saved_model')
    detect_fn = tf.saved_model.load(saved_model_path)
    detect_fn_sig = detect_fn.signatures['serving_default']
    image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    detections = detect_fn_sig(image)

    # The exported graph doesn't have numerically correct outputs, but there
    # should be 4.
    self.assertEqual(4, len(detections))

  def test_center_net_inference_object_detection(self):
    tmp_dir = self.get_temp_dir()
    output_directory = os.path.join(tmp_dir, 'output')
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      export_tflite_graph_lib_tf2.export_tflite_model(
          pipeline_config=self._get_center_net_config(),
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory,
          max_detections=10,
          use_regular_nms=False)

    saved_model_path = os.path.join(output_directory, 'saved_model')
    detect_fn = tf.saved_model.load(saved_model_path)
    detect_fn_sig = detect_fn.signatures['serving_default']
    image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    detections = detect_fn_sig(image)

    # The exported graph doesn't have numerically correct outputs, but there
    # should be 4.
    self.assertEqual(4, len(detections))

  def test_center_net_inference_keypoint(self):
    tmp_dir = self.get_temp_dir()
    output_directory = os.path.join(tmp_dir, 'output')
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      export_tflite_graph_lib_tf2.export_tflite_model(
          pipeline_config=self._get_center_net_config(),
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory,
          max_detections=10,
          use_regular_nms=False,
          include_keypoints=True)

    saved_model_path = os.path.join(output_directory, 'saved_model')
    detect_fn = tf.saved_model.load(saved_model_path)
    detect_fn_sig = detect_fn.signatures['serving_default']
    image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    detections = detect_fn_sig(image)

    # The exported graph doesn't have numerically correct outputs, but there
    # should be 6 (4 for boxes, 2 for keypoints).
    self.assertEqual(6, len(detections))


if __name__ == '__main__':
  tf.test.main()
