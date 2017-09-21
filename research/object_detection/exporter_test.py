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
import numpy as np
import six
import tensorflow as tf
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import pipeline_pb2

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top


class FakeModel(model.DetectionModel):

  def __init__(self, add_detection_masks=False):
    self._add_detection_masks = add_detection_masks

  def preprocess(self, inputs):
    return tf.identity(inputs)

  def predict(self, preprocessed_inputs):
    return {'image': tf.layers.conv2d(preprocessed_inputs, 3, 1)}

  def postprocess(self, prediction_dict):
    with tf.control_dependencies(prediction_dict.values()):
      postprocessed_tensors = {
          'detection_boxes': tf.constant([[[0.0, 0.0, 0.5, 0.5],
                                           [0.5, 0.5, 0.8, 0.8]],
                                          [[0.5, 0.5, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0]]], tf.float32),
          'detection_scores': tf.constant([[0.7, 0.6],
                                           [0.9, 0.0]], tf.float32),
          'detection_classes': tf.constant([[0, 1],
                                            [1, 0]], tf.float32),
          'num_detections': tf.constant([2, 1], tf.float32)
      }
      if self._add_detection_masks:
        postprocessed_tensors['detection_masks'] = tf.constant(
            np.arange(64).reshape([2, 2, 4, 4]), tf.float32)
    return postprocessed_tensors

  def restore_map(self, checkpoint_path, from_detection_checkpoint):
    pass

  def loss(self, prediction_dict):
    pass


class ExportInferenceGraphTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self, checkpoint_path,
                                       use_moving_averages):
    g = tf.Graph()
    with g.as_default():
      mock_model = FakeModel()
      preprocessed_inputs = mock_model.preprocess(
          tf.placeholder(tf.float32, shape=[None, None, None, 3]))
      predictions = mock_model.predict(preprocessed_inputs)
      mock_model.postprocess(predictions)
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
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

  def test_export_graph_with_tf_example_input(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

  def test_export_graph_with_encoded_image_string_input(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='encoded_image_string_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

  def test_export_graph_with_moving_averages(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = True
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

  def test_export_model_with_all_output_nodes(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)
    inference_graph = self._load_inference_graph(inference_graph_path)
    with self.test_session(graph=inference_graph):
      inference_graph.get_tensor_by_name('image_tensor:0')
      inference_graph.get_tensor_by_name('detection_boxes:0')
      inference_graph.get_tensor_by_name('detection_scores:0')
      inference_graph.get_tensor_by_name('detection_classes:0')
      inference_graph.get_tensor_by_name('detection_masks:0')
      inference_graph.get_tensor_by_name('num_detections:0')

  def test_export_model_with_detection_only_nodes(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=False)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)
    inference_graph = self._load_inference_graph(inference_graph_path)
    with self.test_session(graph=inference_graph):
      inference_graph.get_tensor_by_name('image_tensor:0')
      inference_graph.get_tensor_by_name('detection_boxes:0')
      inference_graph.get_tensor_by_name('detection_scores:0')
      inference_graph.get_tensor_by_name('detection_classes:0')
      inference_graph.get_tensor_by_name('num_detections:0')
      with self.assertRaises(KeyError):
        inference_graph.get_tensor_by_name('detection_masks:0')

  def test_export_and_run_inference_with_image_tensor(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    inference_graph = self._load_inference_graph(inference_graph_path)
    with self.test_session(graph=inference_graph) as sess:
      image_tensor = inference_graph.get_tensor_by_name('image_tensor:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes_np, scores_np, classes_np, masks_np, num_detections_np) = sess.run(
          [boxes, scores, classes, masks, num_detections],
          feed_dict={image_tensor: np.ones((2, 4, 4, 3)).astype(np.uint8)})
      self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                      [0.5, 0.5, 0.8, 0.8]],
                                     [[0.5, 0.5, 1.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(scores_np, [[0.7, 0.6],
                                      [0.9, 0.0]])
      self.assertAllClose(classes_np, [[1, 2],
                                       [2, 1]])
      self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
      self.assertAllClose(num_detections_np, [2, 1])

  def _create_encoded_image_string(self, image_array_np, encoding_format):
    od_graph = tf.Graph()
    with od_graph.as_default():
      if encoding_format == 'jpg':
        encoded_string = tf.image.encode_jpeg(image_array_np)
      elif encoding_format == 'png':
        encoded_string = tf.image.encode_png(image_array_np)
      else:
        raise ValueError('Supports only the following formats: `jpg`, `png`')
    with self.test_session(graph=od_graph):
      return encoded_string.eval()

  def test_export_and_run_inference_with_encoded_image_string_tensor(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='encoded_image_string_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    inference_graph = self._load_inference_graph(inference_graph_path)
    jpg_image_str = self._create_encoded_image_string(
        np.ones((4, 4, 3)).astype(np.uint8), 'jpg')
    png_image_str = self._create_encoded_image_string(
        np.ones((4, 4, 3)).astype(np.uint8), 'png')
    with self.test_session(graph=inference_graph) as sess:
      image_str_tensor = inference_graph.get_tensor_by_name(
          'encoded_image_string_tensor:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      for image_str in [jpg_image_str, png_image_str]:
        image_str_batch_np = np.hstack([image_str]* 2)
        (boxes_np, scores_np, classes_np, masks_np,
         num_detections_np) = sess.run(
             [boxes, scores, classes, masks, num_detections],
             feed_dict={image_str_tensor: image_str_batch_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])

  def test_raise_runtime_error_on_images_with_different_sizes(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='encoded_image_string_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    inference_graph = self._load_inference_graph(inference_graph_path)
    large_image = self._create_encoded_image_string(
        np.ones((4, 4, 3)).astype(np.uint8), 'jpg')
    small_image = self._create_encoded_image_string(
        np.ones((2, 2, 3)).astype(np.uint8), 'jpg')

    image_str_batch_np = np.hstack([large_image, small_image])
    with self.test_session(graph=inference_graph) as sess:
      image_str_tensor = inference_graph.get_tensor_by_name(
          'encoded_image_string_tensor:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   '^TensorArray has inconsistent shapes.'):
        sess.run([boxes, scores, classes, masks, num_detections],
                 feed_dict={image_str_tensor: image_str_batch_np})

  def test_export_and_run_inference_with_tf_example(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    inference_graph = self._load_inference_graph(inference_graph_path)
    tf_example_np = np.expand_dims(self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8)), axis=0)
    with self.test_session(graph=inference_graph) as sess:
      tf_example = inference_graph.get_tensor_by_name('tf_example:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes_np, scores_np, classes_np, masks_np, num_detections_np) = sess.run(
          [boxes, scores, classes, masks, num_detections],
          feed_dict={tf_example: tf_example_np})
      self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                      [0.5, 0.5, 0.8, 0.8]],
                                     [[0.5, 0.5, 1.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(scores_np, [[0.7, 0.6],
                                      [0.9, 0.0]])
      self.assertAllClose(classes_np, [[1, 2],
                                       [2, 1]])
      self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
      self.assertAllClose(num_detections_np, [2, 1])

  def test_export_saved_model_and_run_inference(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    output_directory = os.path.join(tmp_dir, 'output')
    saved_model_path = os.path.join(output_directory, 'saved_model')

    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    tf_example_np = np.hstack([self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8))] * 2)
    with tf.Graph().as_default() as od_graph:
      with self.test_session(graph=od_graph) as sess:
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        tf_example = od_graph.get_tensor_by_name('tf_example:0')
        boxes = od_graph.get_tensor_by_name('detection_boxes:0')
        scores = od_graph.get_tensor_by_name('detection_scores:0')
        classes = od_graph.get_tensor_by_name('detection_classes:0')
        masks = od_graph.get_tensor_by_name('detection_masks:0')
        num_detections = od_graph.get_tensor_by_name('num_detections:0')
        (boxes_np, scores_np, classes_np, masks_np,
         num_detections_np) = sess.run(
             [boxes, scores, classes, masks, num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])

  def test_export_checkpoint_and_run_inference(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    output_directory = os.path.join(tmp_dir, 'output')
    model_path = os.path.join(output_directory, 'model.ckpt')
    meta_graph_path = model_path + '.meta'

    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      exporter.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)

    tf_example_np = np.hstack([self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8))] * 2)
    with tf.Graph().as_default() as od_graph:
      with self.test_session(graph=od_graph) as sess:
        new_saver = tf.train.import_meta_graph(meta_graph_path)
        new_saver.restore(sess, model_path)

        tf_example = od_graph.get_tensor_by_name('tf_example:0')
        boxes = od_graph.get_tensor_by_name('detection_boxes:0')
        scores = od_graph.get_tensor_by_name('detection_scores:0')
        classes = od_graph.get_tensor_by_name('detection_classes:0')
        masks = od_graph.get_tensor_by_name('detection_masks:0')
        num_detections = od_graph.get_tensor_by_name('num_detections:0')
        (boxes_np, scores_np, classes_np, masks_np,
         num_detections_np) = sess.run(
             [boxes, scores, classes, masks, num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])


if __name__ == '__main__':
  tf.test.main()
