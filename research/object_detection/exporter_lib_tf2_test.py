# Lint as: python2, python3
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

"""Test for exporter_lib_v2.py."""

from __future__ import division
import io
import os
import unittest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import six

import tensorflow.compat.v2 as tf

from object_detection import exporter_lib_v2
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.protos import pipeline_pb2
from object_detection.utils import dataset_util
from object_detection.utils import tf_version

if six.PY2:
  import mock  # pylint: disable=g-importing-member,g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-importing-member,g-import-not-at-top


class FakeModel(model.DetectionModel):

  def __init__(self, conv_weight_scalar=1.0):
    super(FakeModel, self).__init__(num_classes=2)
    self._conv = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, strides=(1, 1), padding='valid',
        kernel_initializer=tf.keras.initializers.Constant(
            value=conv_weight_scalar))

  def preprocess(self, inputs):
    true_image_shapes = []  # Doesn't matter for the fake model.
    return tf.identity(inputs), true_image_shapes

  def predict(self, preprocessed_inputs, true_image_shapes, **side_inputs):
    return_dict = {'image': self._conv(preprocessed_inputs)}
    if 'side_inp_1' in side_inputs:
      return_dict['image'] += side_inputs['side_inp_1']
    return return_dict

  def postprocess(self, prediction_dict, true_image_shapes):
    predict_tensor_sum = tf.reduce_sum(prediction_dict['image'])
    with tf.control_dependencies(list(prediction_dict.values())):
      postprocessed_tensors = {
          'detection_boxes': tf.constant([[[0.0, 0.0, 0.5, 0.5],
                                           [0.5, 0.5, 0.8, 0.8]],
                                          [[0.5, 0.5, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0]]], tf.float32),
          'detection_scores': predict_tensor_sum + tf.constant(
              [[0.7, 0.6], [0.9, 0.0]], tf.float32),
          'detection_classes': tf.constant([[0, 1],
                                            [1, 0]], tf.float32),
          'num_detections': tf.constant([2, 1], tf.float32),
      }
    return postprocessed_tensors

  def restore_map(self, checkpoint_path, fine_tune_checkpoint_type):
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
class ExportInferenceGraphTest(tf.test.TestCase, parameterized.TestCase):

  def _save_checkpoint_from_mock_model(
      self, checkpoint_dir, conv_weight_scalar=6.0):
    mock_model = FakeModel(conv_weight_scalar)
    fake_image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    preprocessed_inputs, true_image_shapes = mock_model.preprocess(fake_image)
    predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
    mock_model.postprocess(predictions, true_image_shapes)

    ckpt = tf.train.Checkpoint(model=mock_model)
    exported_checkpoint_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'encoded_image_string_tensor'},
      {'input_type': 'tf_example'},
  )
  def test_export_yields_correct_directory_structure(
      self, input_type='image_tensor'):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter_lib_v2.export_inference_graph(
          input_type=input_type,
          pipeline_config=pipeline_config,
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory)
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'saved_model.pb')))
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'variables', 'variables.index')))
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'variables',
          'variables.data-00000-of-00001')))
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'checkpoint', 'ckpt-0.index')))
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'checkpoint', 'ckpt-0.data-00000-of-00001')))
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'pipeline.config')))

  def get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return np.zeros((1, 20, 20, 3), dtype=np.uint8)
    if input_type == 'float_image_tensor':
      return np.zeros((1, 20, 20, 3), dtype=np.float32)
    elif input_type == 'encoded_image_string_tensor':
      image = Image.new('RGB', (20, 20))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((20, 20, 3), dtype=tf.uint8)
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).numpy()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                  dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                  dataset_util.bytes_feature(six.b('jpeg')),
                  'image/source_id':
                  dataset_util.bytes_feature(six.b('image_id')),
              })).SerializeToString()
      return [example]

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'encoded_image_string_tensor'},
      {'input_type': 'tf_example'},
      {'input_type': 'float_image_tensor'},
  )
  def test_export_saved_model_and_run_inference(
      self, input_type='image_tensor'):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter_lib_v2.export_inference_graph(
          input_type=input_type,
          pipeline_config=pipeline_config,
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory)

      saved_model_path = os.path.join(output_directory, 'saved_model')
      detect_fn = tf.saved_model.load(saved_model_path)
      image = self.get_dummy_input(input_type)
      detections = detect_fn(tf.constant(image))

      detection_fields = fields.DetectionResultFields
      self.assertAllClose(detections[detection_fields.detection_boxes],
                          [[[0.0, 0.0, 0.5, 0.5],
                            [0.5, 0.5, 0.8, 0.8]],
                           [[0.5, 0.5, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(detections[detection_fields.detection_scores],
                          [[0.7, 0.6], [0.9, 0.0]])
      self.assertAllClose(detections[detection_fields.detection_classes],
                          [[1, 2], [2, 1]])
      self.assertAllClose(detections[detection_fields.num_detections], [2, 1])

  @parameterized.parameters(
      {'use_default_serving': True},
      {'use_default_serving': False}
  )
  def test_export_saved_model_and_run_inference_with_side_inputs(
      self, input_type='image_tensor', use_default_serving=True):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter_lib_v2.export_inference_graph(
          input_type=input_type,
          pipeline_config=pipeline_config,
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory,
          use_side_inputs=True,
          side_input_shapes='1/2,2',
          side_input_names='side_inp_1,side_inp_2',
          side_input_types='tf.float32,tf.uint8')

      saved_model_path = os.path.join(output_directory, 'saved_model')
      detect_fn = tf.saved_model.load(saved_model_path)
      detect_fn_sig = detect_fn.signatures['serving_default']
      image = tf.constant(self.get_dummy_input(input_type))
      side_input_1 = np.ones((1,), dtype=np.float32)
      side_input_2 = np.ones((2, 2), dtype=np.uint8)
      if use_default_serving:
        detections = detect_fn_sig(input_tensor=image,
                                   side_inp_1=tf.constant(side_input_1),
                                   side_inp_2=tf.constant(side_input_2))
      else:
        detections = detect_fn(image,
                               tf.constant(side_input_1),
                               tf.constant(side_input_2))

      detection_fields = fields.DetectionResultFields
      self.assertAllClose(detections[detection_fields.detection_boxes],
                          [[[0.0, 0.0, 0.5, 0.5],
                            [0.5, 0.5, 0.8, 0.8]],
                           [[0.5, 0.5, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(detections[detection_fields.detection_scores],
                          [[400.7, 400.6], [400.9, 400.0]])
      self.assertAllClose(detections[detection_fields.detection_classes],
                          [[1, 2], [2, 1]])
      self.assertAllClose(detections[detection_fields.num_detections], [2, 1])

  def test_export_checkpoint_and_run_inference_with_image(self):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir, conv_weight_scalar=2.0)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter_lib_v2.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory)

      mock_model = FakeModel()
      ckpt = tf.compat.v2.train.Checkpoint(
          model=mock_model)
      checkpoint_dir = os.path.join(tmp_dir, 'output', 'checkpoint')
      manager = tf.compat.v2.train.CheckpointManager(
          ckpt, checkpoint_dir, max_to_keep=7)
      ckpt.restore(manager.latest_checkpoint).expect_partial()

      fake_image = tf.ones(shape=[1, 5, 5, 3], dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = mock_model.preprocess(fake_image)
      predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
      detections = mock_model.postprocess(predictions, true_image_shapes)

      # 150 = conv_weight_scalar * height * width * channels = 2 * 5 * 5 * 3.
      self.assertAllClose(detections['detection_scores'],
                          [[150 + 0.7, 150 + 0.6], [150 + 0.9, 150 + 0.0]])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
