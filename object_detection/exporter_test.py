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

"""Tests for object_detection.export_inference_graph."""
import os
import mock
import numpy as np
import tensorflow as tf
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import pipeline_pb2


class FakeModel(model.DetectionModel):

  def preprocess(self, inputs):
    return (tf.identity(inputs) *
            tf.get_variable('dummy', shape=(),
                            initializer=tf.constant_initializer(2),
                            dtype=tf.float32))

  def predict(self, preprocessed_inputs):
    return {'image': tf.identity(preprocessed_inputs)}

  def postprocess(self, prediction_dict):
    with tf.control_dependencies(prediction_dict.values()):
      return {
          'detection_boxes': tf.constant([[0.0, 0.0, 0.5, 0.5],
                                          [0.5, 0.5, 0.8, 0.8]], tf.float32),
          'detection_scores': tf.constant([[0.7, 0.6]], tf.float32),
          'detection_classes': tf.constant([[0, 1]], tf.float32),
          'num_detections': tf.constant([2], tf.float32)
      }

  def restore_fn(self, checkpoint_path, from_detection_checkpoint):
    pass

  def loss(self, prediction_dict):
    pass


class ExportInferenceGraphTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self, checkpoint_path,
                                       use_moving_averages):
    g = tf.Graph()
    with g.as_default():
      mock_model = FakeModel(num_classes=1)
      mock_model.preprocess(tf.constant([1, 3, 4, 3], tf.float32))
      if use_moving_averages:
        tf.train.ExponentialMovingAverage(0.0).apply()
      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init)
        saver.save(sess, checkpoint_path)

  def _load_inference_graph(self, inference_graph_path):
    od_graph = tf.Graph()
    with od_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(inference_graph_path) as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return od_graph

  def _create_tf_example(self, image_array):
    with self.test_session():
      encoded_image = tf.image.encode_jpeg(tf.constant(image_array)).eval()
    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(encoded_image),
        'image/format': _bytes_feature('jpg'),
        'image/source_id': _bytes_feature('image_id')
    })).SerializeToString()
    return example

  def test_export_graph_with_image_tensor_input(self):
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      inference_graph_path = os.path.join(self.get_temp_dir(),
                                          'exported_graph.pbtxt')

      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          checkpoint_path=None,
          inference_graph_path=inference_graph_path)

  def test_export_graph_with_tf_example_input(self):
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      inference_graph_path = os.path.join(self.get_temp_dir(),
                                          'exported_graph.pbtxt')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          checkpoint_path=None,
          inference_graph_path=inference_graph_path)

  def test_export_frozen_graph(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'model-ckpt')
    self._save_checkpoint_from_mock_model(checkpoint_path,
                                          use_moving_averages=False)
    inference_graph_path = os.path.join(self.get_temp_dir(),
                                        'exported_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          checkpoint_path=checkpoint_path,
          inference_graph_path=inference_graph_path)

  def test_export_frozen_graph_with_moving_averages(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'model-ckpt')
    self._save_checkpoint_from_mock_model(checkpoint_path,
                                          use_moving_averages=True)
    inference_graph_path = os.path.join(self.get_temp_dir(),
                                        'exported_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = True
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          checkpoint_path=checkpoint_path,
          inference_graph_path=inference_graph_path)

  def test_export_and_run_inference_with_image_tensor(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'model-ckpt')
    self._save_checkpoint_from_mock_model(checkpoint_path,
                                          use_moving_averages=False)
    inference_graph_path = os.path.join(self.get_temp_dir(),
                                        'exported_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          checkpoint_path=checkpoint_path,
          inference_graph_path=inference_graph_path)

    inference_graph = self._load_inference_graph(inference_graph_path)
    with self.test_session(graph=inference_graph) as sess:
      image_tensor = inference_graph.get_tensor_by_name('image_tensor:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: np.ones((1, 4, 4, 3)).astype(np.uint8)})
      self.assertAllClose(boxes, [[0.0, 0.0, 0.5, 0.5],
                                  [0.5, 0.5, 0.8, 0.8]])
      self.assertAllClose(scores, [[0.7, 0.6]])
      self.assertAllClose(classes, [[1, 2]])
      self.assertAllClose(num_detections, [2])

  def test_export_and_run_inference_with_tf_example(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'model-ckpt')
    self._save_checkpoint_from_mock_model(checkpoint_path,
                                          use_moving_averages=False)
    inference_graph_path = os.path.join(self.get_temp_dir(),
                                        'exported_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=1)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          checkpoint_path=checkpoint_path,
          inference_graph_path=inference_graph_path)

    inference_graph = self._load_inference_graph(inference_graph_path)
    with self.test_session(graph=inference_graph) as sess:
      tf_example = inference_graph.get_tensor_by_name('tf_example:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={tf_example: self._create_tf_example(
              np.ones((4, 4, 3)).astype(np.uint8))})
      self.assertAllClose(boxes, [[0.0, 0.0, 0.5, 0.5],
                                  [0.5, 0.5, 0.8, 0.8]])
      self.assertAllClose(scores, [[0.7, 0.6]])
      self.assertAllClose(classes, [[1, 2]])
      self.assertAllClose(num_detections, [2])


if __name__ == '__main__':
  tf.test.main()
