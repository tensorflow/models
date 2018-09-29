# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.export_tflite_ssd_graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from object_detection import export_tflite_ssd_graph_lib
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import graph_rewriter_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import post_processing_pb2


if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top


class FakeModel(model.DetectionModel):

  def __init__(self, add_detection_masks=False):
    self._add_detection_masks = add_detection_masks

  def preprocess(self, inputs):
    pass

  def predict(self, preprocessed_inputs, true_image_shapes):
    features = tf.contrib.slim.conv2d(preprocessed_inputs, 3, 1)
    with tf.control_dependencies([features]):
      prediction_tensors = {
          'box_encodings':
              tf.constant([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]],
                          tf.float32),
          'class_predictions_with_background':
              tf.constant([[[0.7, 0.6], [0.9, 0.0]]], tf.float32),
      }
    with tf.control_dependencies(
        [tf.convert_to_tensor(features.get_shape().as_list()[1:3])]):
      prediction_tensors['anchors'] = tf.constant(
          [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], tf.float32)
    return prediction_tensors

  def postprocess(self, prediction_tensors, true_image_shapes):
    pass

  def restore_map(self, checkpoint_path, from_detection_checkpoint):
    pass

  def loss(self, prediction_dict, true_image_shapes):
    pass


class ExportTfliteGraphTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self,
                                       checkpoint_path,
                                       use_moving_averages,
                                       quantize=False,
                                       num_channels=3):
    g = tf.Graph()
    with g.as_default():
      mock_model = FakeModel()
      inputs = tf.placeholder(tf.float32, shape=[1, 10, 10, num_channels])
      mock_model.predict(inputs, true_image_shapes=None)
      if use_moving_averages:
        tf.train.ExponentialMovingAverage(0.0).apply()
      tf.train.get_or_create_global_step()
      if quantize:
        graph_rewriter_config = graph_rewriter_pb2.GraphRewriter()
        graph_rewriter_config.quantization.delay = 500000
        graph_rewriter_fn = graph_rewriter_builder.build(
            graph_rewriter_config, is_training=False)
        graph_rewriter_fn()

      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init)
        saver.save(sess, checkpoint_path)

  def _assert_quant_vars_exists(self, tflite_graph_file):
    with tf.gfile.Open(tflite_graph_file) as f:
      graph_string = f.read()
      print(graph_string)
      self.assertTrue('quant' in graph_string)

  def _import_graph_and_run_inference(self, tflite_graph_file, num_channels=3):
    """Imports a tflite graph, runs single inference and returns outputs."""
    graph = tf.Graph()
    with graph.as_default():
      graph_def = tf.GraphDef()
      with tf.gfile.Open(tflite_graph_file) as f:
        graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')
      input_tensor = graph.get_tensor_by_name('normalized_input_image_tensor:0')
      box_encodings = graph.get_tensor_by_name('raw_outputs/box_encodings:0')
      class_predictions = graph.get_tensor_by_name(
          'raw_outputs/class_predictions:0')
      with self.test_session(graph) as sess:
        [box_encodings_np, class_predictions_np] = sess.run(
            [box_encodings, class_predictions],
            feed_dict={input_tensor: np.random.rand(1, 10, 10, num_channels)})
    return box_encodings_np, class_predictions_np

  def _export_graph(self, pipeline_config, num_channels=3):
    """Exports a tflite graph."""
    output_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(output_dir, 'model.ckpt')
    tflite_graph_file = os.path.join(output_dir, 'tflite_graph.pb')

    quantize = pipeline_config.HasField('graph_rewriter')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix,
        use_moving_averages=pipeline_config.eval_config.use_moving_averages,
        quantize=quantize,
        num_channels=num_channels)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()

      with tf.Graph().as_default():
        export_tflite_ssd_graph_lib.export_tflite_graph(
            pipeline_config=pipeline_config,
            trained_checkpoint_prefix=trained_checkpoint_prefix,
            output_dir=output_dir,
            add_postprocessing_op=False,
            max_detections=10,
            max_classes_per_detection=1)
    return tflite_graph_file

  def _export_graph_with_postprocessing_op(self,
                                           pipeline_config,
                                           num_channels=3):
    """Exports a tflite graph with custom postprocessing op."""
    output_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(output_dir, 'model.ckpt')
    tflite_graph_file = os.path.join(output_dir, 'tflite_graph.pb')

    quantize = pipeline_config.HasField('graph_rewriter')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix,
        use_moving_averages=pipeline_config.eval_config.use_moving_averages,
        quantize=quantize,
        num_channels=num_channels)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()

      with tf.Graph().as_default():
        export_tflite_ssd_graph_lib.export_tflite_graph(
            pipeline_config=pipeline_config,
            trained_checkpoint_prefix=trained_checkpoint_prefix,
            output_dir=output_dir,
            add_postprocessing_op=True,
            max_detections=10,
            max_classes_per_detection=1)
    return tflite_graph_file

  def test_export_tflite_graph_with_moving_averages(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = True
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))

    (box_encodings_np, class_predictions_np
    ) = self._import_graph_and_run_inference(tflite_graph_file)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np, [[[0.7, 0.6], [0.9, 0.0]]])

  def test_export_tflite_graph_without_moving_averages(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))
    (box_encodings_np, class_predictions_np
    ) = self._import_graph_and_run_inference(tflite_graph_file)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np, [[[0.7, 0.6], [0.9, 0.0]]])

  def test_export_tflite_graph_grayscale(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    (pipeline_config.model.ssd.image_resizer.fixed_shape_resizer
    ).convert_to_grayscale = True
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config, num_channels=1)
    self.assertTrue(os.path.exists(tflite_graph_file))
    (box_encodings_np,
     class_predictions_np) = self._import_graph_and_run_inference(
         tflite_graph_file, num_channels=1)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np, [[[0.7, 0.6], [0.9, 0.0]]])

  def test_export_tflite_graph_with_quantization(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.graph_rewriter.quantization.delay = 500000
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))
    self._assert_quant_vars_exists(tflite_graph_file)
    (box_encodings_np, class_predictions_np
    ) = self._import_graph_and_run_inference(tflite_graph_file)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np, [[[0.7, 0.6], [0.9, 0.0]]])

  def test_export_tflite_graph_with_softmax_score_conversion(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.post_processing.score_converter = (
        post_processing_pb2.PostProcessing.SOFTMAX)
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))
    (box_encodings_np, class_predictions_np
    ) = self._import_graph_and_run_inference(tflite_graph_file)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np,
                        [[[0.524979, 0.475021], [0.710949, 0.28905]]])

  def test_export_tflite_graph_with_sigmoid_score_conversion(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.post_processing.score_converter = (
        post_processing_pb2.PostProcessing.SIGMOID)
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph(pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))
    (box_encodings_np, class_predictions_np
    ) = self._import_graph_and_run_inference(tflite_graph_file)
    self.assertAllClose(box_encodings_np,
                        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.8, 0.8]]])
    self.assertAllClose(class_predictions_np,
                        [[[0.668188, 0.645656], [0.710949, 0.5]]])

  def test_export_tflite_graph_with_postprocessing_op(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = False
    pipeline_config.model.ssd.post_processing.score_converter = (
        post_processing_pb2.PostProcessing.SIGMOID)
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = 10
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = 10
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale = 10.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale = 5.0
    pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale = 5.0
    tflite_graph_file = self._export_graph_with_postprocessing_op(
        pipeline_config)
    self.assertTrue(os.path.exists(tflite_graph_file))
    graph = tf.Graph()
    with graph.as_default():
      graph_def = tf.GraphDef()
      with tf.gfile.Open(tflite_graph_file) as f:
        graph_def.ParseFromString(f.read())
      all_op_names = [node.name for node in graph_def.node]
      self.assertTrue('TFLite_Detection_PostProcess' in all_op_names)
      for node in graph_def.node:
        if node.name == 'TFLite_Detection_PostProcess':
          self.assertTrue(node.attr['_output_quantized'].b is True)
          self.assertTrue(
              node.attr['_support_output_type_float_in_quantized_op'].b is True)
          self.assertTrue(node.attr['y_scale'].f == 10.0)
          self.assertTrue(node.attr['x_scale'].f == 10.0)
          self.assertTrue(node.attr['h_scale'].f == 5.0)
          self.assertTrue(node.attr['w_scale'].f == 5.0)
          self.assertTrue(node.attr['num_classes'].i == 2)
          self.assertTrue(
              all([
                  t == types_pb2.DT_FLOAT
                  for t in node.attr['_output_types'].list.type
              ]))


if __name__ == '__main__':
  tf.test.main()
