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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
import numpy as np
import six
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.tools import strip_unused_lib
from object_detection import exporter
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import graph_rewriter_pb2
from object_detection.protos import pipeline_pb2
from object_detection.utils import ops
from object_detection.utils import tf_version
from object_detection.utils import variables_helper

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top, g-importing-member

# pylint: disable=g-import-not-at-top
try:
  import tf_slim as slim
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


class FakeModel(model.DetectionModel):

  def __init__(self, add_detection_keypoints=False, add_detection_masks=False,
               add_detection_features=False):
    self._add_detection_keypoints = add_detection_keypoints
    self._add_detection_masks = add_detection_masks
    self._add_detection_features = add_detection_features

  def preprocess(self, inputs):
    true_image_shapes = []  # Doesn't matter for the fake model.
    return tf.identity(inputs), true_image_shapes

  def predict(self, preprocessed_inputs, true_image_shapes):
    return {'image': tf.layers.conv2d(preprocessed_inputs, 3, 1)}

  def postprocess(self, prediction_dict, true_image_shapes):
    with tf.control_dependencies(list(prediction_dict.values())):
      postprocessed_tensors = {
          'detection_boxes': tf.constant([[[0.0, 0.0, 0.5, 0.5],
                                           [0.5, 0.5, 0.8, 0.8]],
                                          [[0.5, 0.5, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0]]], tf.float32),
          'detection_scores': tf.constant([[0.7, 0.6],
                                           [0.9, 0.0]], tf.float32),
          'detection_multiclass_scores': tf.constant([[[0.3, 0.7], [0.4, 0.6]],
                                                      [[0.1, 0.9], [0.0, 0.0]]],
                                                     tf.float32),
          'detection_classes': tf.constant([[0, 1],
                                            [1, 0]], tf.float32),
          'num_detections': tf.constant([2, 1], tf.float32),
          'raw_detection_boxes': tf.constant([[[0.0, 0.0, 0.5, 0.5],
                                               [0.5, 0.5, 0.8, 0.8]],
                                              [[0.5, 0.5, 1.0, 1.0],
                                               [0.0, 0.5, 0.0, 0.5]]],
                                             tf.float32),
          'raw_detection_scores': tf.constant([[0.7, 0.6],
                                               [0.9, 0.5]], tf.float32),
      }
      if self._add_detection_keypoints:
        postprocessed_tensors['detection_keypoints'] = tf.constant(
            np.arange(48).reshape([2, 2, 6, 2]), tf.float32)
      if self._add_detection_masks:
        postprocessed_tensors['detection_masks'] = tf.constant(
            np.arange(64).reshape([2, 2, 4, 4]), tf.float32)
      if self._add_detection_features:
        # let fake detection features have shape [4, 4, 10]
        postprocessed_tensors['detection_features'] = tf.constant(
            np.ones((2, 2, 4, 4, 10)), tf.float32)

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


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ExportInferenceGraphTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self,
                                       checkpoint_path,
                                       use_moving_averages,
                                       enable_quantization=False):
    g = tf.Graph()
    with g.as_default():
      mock_model = FakeModel()
      preprocessed_inputs, true_image_shapes = mock_model.preprocess(
          tf.placeholder(tf.float32, shape=[None, None, None, 3]))
      predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
      mock_model.postprocess(predictions, true_image_shapes)
      if use_moving_averages:
        tf.train.ExponentialMovingAverage(0.0).apply()
      tf.train.get_or_create_global_step()
      if enable_quantization:
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

  def _load_inference_graph(self, inference_graph_path, is_binary=True):
    od_graph = tf.Graph()
    with od_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(inference_graph_path, mode='rb') as fid:
        if is_binary:
          od_graph_def.ParseFromString(fid.read())
        else:
          text_format.Parse(fid.read(), od_graph_def)
        tf.import_graph_def(od_graph_def, name='')
    return od_graph

  def _create_tf_example(self, image_array):
    with self.test_session():
      encoded_image = tf.image.encode_jpeg(tf.constant(image_array)).eval()
    def _bytes_feature(value):
      return tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[six.ensure_binary(value)]))

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
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'saved_model.pb')))

  def test_write_inference_graph(self):
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
          output_directory=output_directory,
          write_inference_graph=True)
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'inference_graph.pbtxt')))

  def test_export_graph_with_fixed_size_image_tensor_input(self):
    input_shape = [1, 320, 320, 3]

    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix, use_moving_averages=False)
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
          output_directory=output_directory,
          input_shape=input_shape)
      saved_model_path = os.path.join(output_directory, 'saved_model')
      self.assertTrue(
          os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')))

    with tf.Graph().as_default() as od_graph:
      with self.test_session(graph=od_graph) as sess:
        meta_graph = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        signature = meta_graph.signature_def['serving_default']
        input_tensor_name = signature.inputs['inputs'].name
        image_tensor = od_graph.get_tensor_by_name(input_tensor_name)
        self.assertSequenceEqual(image_tensor.get_shape().as_list(),
                                 input_shape)

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
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'saved_model.pb')))

  def test_export_graph_with_fixed_size_tf_example_input(self):
    input_shape = [1, 320, 320, 3]

    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix, use_moving_averages=False)
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
          output_directory=output_directory,
          input_shape=input_shape)
      saved_model_path = os.path.join(output_directory, 'saved_model')
      self.assertTrue(
          os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')))

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
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'saved_model.pb')))

  def test_export_graph_with_fixed_size_encoded_image_string_input(self):
    input_shape = [1, 320, 320, 3]

    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix, use_moving_averages=False)
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
          output_directory=output_directory,
          input_shape=input_shape)
      saved_model_path = os.path.join(output_directory, 'saved_model')
      self.assertTrue(
          os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')))

  def _get_variables_in_checkpoint(self, checkpoint_file):
    return set([
        var_name
        for var_name, _ in tf.train.list_variables(checkpoint_file)])

  def test_replace_variable_values_with_moving_averages(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    new_checkpoint_prefix = os.path.join(tmp_dir, 'new.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    graph = tf.Graph()
    with graph.as_default():
      fake_model = FakeModel()
      preprocessed_inputs, true_image_shapes = fake_model.preprocess(
          tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]))
      predictions = fake_model.predict(preprocessed_inputs, true_image_shapes)
      fake_model.postprocess(predictions, true_image_shapes)
      exporter.replace_variable_values_with_moving_averages(
          graph, trained_checkpoint_prefix, new_checkpoint_prefix)

    expected_variables = set(['conv2d/bias', 'conv2d/kernel'])
    variables_in_old_ckpt = self._get_variables_in_checkpoint(
        trained_checkpoint_prefix)
    self.assertIn('conv2d/bias/ExponentialMovingAverage',
                  variables_in_old_ckpt)
    self.assertIn('conv2d/kernel/ExponentialMovingAverage',
                  variables_in_old_ckpt)
    variables_in_new_ckpt = self._get_variables_in_checkpoint(
        new_checkpoint_prefix)
    self.assertTrue(expected_variables.issubset(variables_in_new_ckpt))
    self.assertNotIn('conv2d/bias/ExponentialMovingAverage',
                     variables_in_new_ckpt)
    self.assertNotIn('conv2d/kernel/ExponentialMovingAverage',
                     variables_in_new_ckpt)

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
      self.assertTrue(os.path.exists(os.path.join(
          output_directory, 'saved_model', 'saved_model.pb')))
    expected_variables = set(['conv2d/bias', 'conv2d/kernel', 'global_step'])
    actual_variables = set(
        [var_name for var_name, _ in tf.train.list_variables(output_directory)])
    self.assertTrue(expected_variables.issubset(actual_variables))

  def test_export_model_with_quantization_nodes(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(
        trained_checkpoint_prefix,
        use_moving_averages=False,
        enable_quantization=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'inference_graph.pbtxt')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      text_format.Merge(
          """graph_rewriter {
               quantization {
                 delay: 50000
                 activation_bits: 8
                 weight_bits: 8
               }
             }""", pipeline_config)
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory,
          write_inference_graph=True)
    self._load_inference_graph(inference_graph_path, is_binary=False)
    has_quant_nodes = False
    for v in variables_helper.get_global_variables_safely():
      if six.ensure_str(v.op.name).endswith('act_quant/min'):
        has_quant_nodes = True
        break
    self.assertTrue(has_quant_nodes)

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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True,
          add_detection_features=True)
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
      inference_graph.get_tensor_by_name('detection_multiclass_scores:0')
      inference_graph.get_tensor_by_name('detection_classes:0')
      inference_graph.get_tensor_by_name('detection_keypoints:0')
      inference_graph.get_tensor_by_name('detection_masks:0')
      inference_graph.get_tensor_by_name('num_detections:0')
      inference_graph.get_tensor_by_name('detection_features:0')

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
      inference_graph.get_tensor_by_name('detection_multiclass_scores:0')
      inference_graph.get_tensor_by_name('detection_classes:0')
      inference_graph.get_tensor_by_name('num_detections:0')
      with self.assertRaises(KeyError):
        inference_graph.get_tensor_by_name('detection_keypoints:0')
        inference_graph.get_tensor_by_name('detection_masks:0')

  def test_export_model_with_detection_only_nodes_and_detection_features(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(add_detection_features=True)
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
      inference_graph.get_tensor_by_name('detection_multiclass_scores:0')
      inference_graph.get_tensor_by_name('detection_classes:0')
      inference_graph.get_tensor_by_name('num_detections:0')
      inference_graph.get_tensor_by_name('detection_features:0')
      with self.assertRaises(KeyError):
        inference_graph.get_tensor_by_name('detection_keypoints:0')
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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
      keypoints = inference_graph.get_tensor_by_name('detection_keypoints:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes_np, scores_np, classes_np, keypoints_np, masks_np,
       num_detections_np) = sess.run(
           [boxes, scores, classes, keypoints, masks, num_detections],
           feed_dict={image_tensor: np.ones((2, 4, 4, 3)).astype(np.uint8)})
      self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                      [0.5, 0.5, 0.8, 0.8]],
                                     [[0.5, 0.5, 1.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(scores_np, [[0.7, 0.6],
                                      [0.9, 0.0]])
      self.assertAllClose(classes_np, [[1, 2],
                                       [2, 1]])
      self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
      multiclass_scores = inference_graph.get_tensor_by_name(
          'detection_multiclass_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      keypoints = inference_graph.get_tensor_by_name('detection_keypoints:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      for image_str in [jpg_image_str, png_image_str]:
        image_str_batch_np = np.hstack([image_str]* 2)
        (boxes_np, scores_np, multiclass_scores_np, classes_np, keypoints_np,
         masks_np, num_detections_np) = sess.run(
             [
                 boxes, scores, multiclass_scores, classes, keypoints, masks,
                 num_detections
             ],
             feed_dict={image_str_tensor: image_str_batch_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(multiclass_scores_np, [[[0.3, 0.7], [0.4, 0.6]],
                                                   [[0.1, 0.9], [0.0, 0.0]]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
      keypoints = inference_graph.get_tensor_by_name('detection_keypoints:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'TensorArray.*shape'):
        sess.run(
            [boxes, scores, classes, keypoints, masks, num_detections],
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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
      keypoints = inference_graph.get_tensor_by_name('detection_keypoints:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes_np, scores_np, classes_np, keypoints_np, masks_np,
       num_detections_np) = sess.run(
           [boxes, scores, classes, keypoints, masks, num_detections],
           feed_dict={tf_example: tf_example_np})
      self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                      [0.5, 0.5, 0.8, 0.8]],
                                     [[0.5, 0.5, 1.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(scores_np, [[0.7, 0.6],
                                      [0.9, 0.0]])
      self.assertAllClose(classes_np, [[1, 2],
                                       [2, 1]])
      self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
      self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
      self.assertAllClose(num_detections_np, [2, 1])

  def test_write_frozen_graph(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    inference_graph_path = os.path.join(output_directory,
                                        'frozen_inference_graph.pb')
    tf.gfile.MakeDirs(output_directory)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      detection_model = model_builder.build(pipeline_config.model,
                                            is_training=False)
      outputs, _ = exporter.build_detection_graph(
          input_type='tf_example',
          detection_model=detection_model,
          input_shape=None,
          output_collection_name='inference_op',
          graph_hook_fn=None)
      output_node_names = ','.join(list(outputs.keys()))
      saver = tf.train.Saver()
      input_saver_def = saver.as_saver_def()
      exporter.freeze_graph_with_def_protos(
          input_graph_def=tf.get_default_graph().as_graph_def(),
          input_saver_def=input_saver_def,
          input_checkpoint=trained_checkpoint_prefix,
          output_node_names=output_node_names,
          restore_op_name='save/restore_all',
          filename_tensor_name='save/Const:0',
          output_graph=inference_graph_path,
          clear_devices=True,
          initializer_nodes='')

    inference_graph = self._load_inference_graph(inference_graph_path)
    tf_example_np = np.expand_dims(self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8)), axis=0)
    with self.test_session(graph=inference_graph) as sess:
      tf_example = inference_graph.get_tensor_by_name('tf_example:0')
      boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
      scores = inference_graph.get_tensor_by_name('detection_scores:0')
      classes = inference_graph.get_tensor_by_name('detection_classes:0')
      keypoints = inference_graph.get_tensor_by_name('detection_keypoints:0')
      masks = inference_graph.get_tensor_by_name('detection_masks:0')
      num_detections = inference_graph.get_tensor_by_name('num_detections:0')
      (boxes_np, scores_np, classes_np, keypoints_np, masks_np,
       num_detections_np) = sess.run(
           [boxes, scores, classes, keypoints, masks, num_detections],
           feed_dict={tf_example: tf_example_np})
      self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                      [0.5, 0.5, 0.8, 0.8]],
                                     [[0.5, 0.5, 1.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]])
      self.assertAllClose(scores_np, [[0.7, 0.6],
                                      [0.9, 0.0]])
      self.assertAllClose(classes_np, [[1, 2],
                                       [2, 1]])
      self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
      self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
      self.assertAllClose(num_detections_np, [2, 1])

  def test_export_graph_saves_pipeline_file(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=True)
    output_directory = os.path.join(tmp_dir, 'output')
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter.export_inference_graph(
          input_type='image_tensor',
          pipeline_config=pipeline_config,
          trained_checkpoint_prefix=trained_checkpoint_prefix,
          output_directory=output_directory)
      expected_pipeline_path = os.path.join(
          output_directory, 'pipeline.config')
      self.assertTrue(os.path.exists(expected_pipeline_path))

      written_pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      with tf.gfile.GFile(expected_pipeline_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, written_pipeline_config)
        self.assertProtoEquals(pipeline_config, written_pipeline_config)

  def test_export_saved_model_and_run_inference(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    output_directory = os.path.join(tmp_dir, 'output')
    saved_model_path = os.path.join(output_directory, 'saved_model')

    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
        meta_graph = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)

        signature = meta_graph.signature_def['serving_default']
        input_tensor_name = signature.inputs['inputs'].name
        tf_example = od_graph.get_tensor_by_name(input_tensor_name)

        boxes = od_graph.get_tensor_by_name(
            signature.outputs['detection_boxes'].name)
        scores = od_graph.get_tensor_by_name(
            signature.outputs['detection_scores'].name)
        multiclass_scores = od_graph.get_tensor_by_name(
            signature.outputs['detection_multiclass_scores'].name)
        classes = od_graph.get_tensor_by_name(
            signature.outputs['detection_classes'].name)
        keypoints = od_graph.get_tensor_by_name(
            signature.outputs['detection_keypoints'].name)
        masks = od_graph.get_tensor_by_name(
            signature.outputs['detection_masks'].name)
        num_detections = od_graph.get_tensor_by_name(
            signature.outputs['num_detections'].name)

        (boxes_np, scores_np, multiclass_scores_np, classes_np, keypoints_np,
         masks_np, num_detections_np) = sess.run(
             [boxes, scores, multiclass_scores, classes, keypoints, masks,
              num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(multiclass_scores_np, [[[0.3, 0.7], [0.4, 0.6]],
                                                   [[0.1, 0.9], [0.0, 0.0]]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])

  def test_write_saved_model(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    output_directory = os.path.join(tmp_dir, 'output')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    tf.gfile.MakeDirs(output_directory)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      detection_model = model_builder.build(pipeline_config.model,
                                            is_training=False)
      outputs, placeholder_tensor = exporter.build_detection_graph(
          input_type='tf_example',
          detection_model=detection_model,
          input_shape=None,
          output_collection_name='inference_op',
          graph_hook_fn=None)
      output_node_names = ','.join(list(outputs.keys()))
      saver = tf.train.Saver()
      input_saver_def = saver.as_saver_def()
      frozen_graph_def = exporter.freeze_graph_with_def_protos(
          input_graph_def=tf.get_default_graph().as_graph_def(),
          input_saver_def=input_saver_def,
          input_checkpoint=trained_checkpoint_prefix,
          output_node_names=output_node_names,
          restore_op_name='save/restore_all',
          filename_tensor_name='save/Const:0',
          output_graph='',
          clear_devices=True,
          initializer_nodes='')
      exporter.write_saved_model(
          saved_model_path=saved_model_path,
          frozen_graph_def=frozen_graph_def,
          inputs=placeholder_tensor,
          outputs=outputs)

    tf_example_np = np.hstack([self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8))] * 2)
    with tf.Graph().as_default() as od_graph:
      with self.test_session(graph=od_graph) as sess:
        meta_graph = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)

        signature = meta_graph.signature_def['serving_default']
        input_tensor_name = signature.inputs['inputs'].name
        tf_example = od_graph.get_tensor_by_name(input_tensor_name)

        boxes = od_graph.get_tensor_by_name(
            signature.outputs['detection_boxes'].name)
        scores = od_graph.get_tensor_by_name(
            signature.outputs['detection_scores'].name)
        classes = od_graph.get_tensor_by_name(
            signature.outputs['detection_classes'].name)
        keypoints = od_graph.get_tensor_by_name(
            signature.outputs['detection_keypoints'].name)
        masks = od_graph.get_tensor_by_name(
            signature.outputs['detection_masks'].name)
        num_detections = od_graph.get_tensor_by_name(
            signature.outputs['num_detections'].name)

        (boxes_np, scores_np, classes_np, keypoints_np, masks_np,
         num_detections_np) = sess.run(
             [boxes, scores, classes, keypoints, masks, num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
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
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
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
        keypoints = od_graph.get_tensor_by_name('detection_keypoints:0')
        masks = od_graph.get_tensor_by_name('detection_masks:0')
        num_detections = od_graph.get_tensor_by_name('num_detections:0')
        (boxes_np, scores_np, classes_np, keypoints_np, masks_np,
         num_detections_np) = sess.run(
             [boxes, scores, classes, keypoints, masks, num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])

  def test_write_graph_and_checkpoint(self):
    tmp_dir = self.get_temp_dir()
    trained_checkpoint_prefix = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(trained_checkpoint_prefix,
                                          use_moving_averages=False)
    output_directory = os.path.join(tmp_dir, 'output')
    model_path = os.path.join(output_directory, 'model.ckpt')
    meta_graph_path = model_path + '.meta'
    tf.gfile.MakeDirs(output_directory)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(
          add_detection_keypoints=True, add_detection_masks=True)
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      pipeline_config.eval_config.use_moving_averages = False
      detection_model = model_builder.build(pipeline_config.model,
                                            is_training=False)
      exporter.build_detection_graph(
          input_type='tf_example',
          detection_model=detection_model,
          input_shape=None,
          output_collection_name='inference_op',
          graph_hook_fn=None)
      saver = tf.train.Saver()
      input_saver_def = saver.as_saver_def()
      exporter.write_graph_and_checkpoint(
          inference_graph_def=tf.get_default_graph().as_graph_def(),
          model_path=model_path,
          input_saver_def=input_saver_def,
          trained_checkpoint_prefix=trained_checkpoint_prefix)

    tf_example_np = np.hstack([self._create_tf_example(
        np.ones((4, 4, 3)).astype(np.uint8))] * 2)
    with tf.Graph().as_default() as od_graph:
      with self.test_session(graph=od_graph) as sess:
        new_saver = tf.train.import_meta_graph(meta_graph_path)
        new_saver.restore(sess, model_path)

        tf_example = od_graph.get_tensor_by_name('tf_example:0')
        boxes = od_graph.get_tensor_by_name('detection_boxes:0')
        scores = od_graph.get_tensor_by_name('detection_scores:0')
        raw_boxes = od_graph.get_tensor_by_name('raw_detection_boxes:0')
        raw_scores = od_graph.get_tensor_by_name('raw_detection_scores:0')
        classes = od_graph.get_tensor_by_name('detection_classes:0')
        keypoints = od_graph.get_tensor_by_name('detection_keypoints:0')
        masks = od_graph.get_tensor_by_name('detection_masks:0')
        num_detections = od_graph.get_tensor_by_name('num_detections:0')
        (boxes_np, scores_np, raw_boxes_np, raw_scores_np, classes_np,
         keypoints_np, masks_np, num_detections_np) = sess.run(
             [boxes, scores, raw_boxes, raw_scores, classes, keypoints, masks,
              num_detections],
             feed_dict={tf_example: tf_example_np})
        self.assertAllClose(boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.8, 0.8]],
                                       [[0.5, 0.5, 1.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0]]])
        self.assertAllClose(scores_np, [[0.7, 0.6],
                                        [0.9, 0.0]])
        self.assertAllClose(raw_boxes_np, [[[0.0, 0.0, 0.5, 0.5],
                                            [0.5, 0.5, 0.8, 0.8]],
                                           [[0.5, 0.5, 1.0, 1.0],
                                            [0.0, 0.5, 0.0, 0.5]]])
        self.assertAllClose(raw_scores_np, [[0.7, 0.6],
                                            [0.9, 0.5]])
        self.assertAllClose(classes_np, [[1, 2],
                                         [2, 1]])
        self.assertAllClose(keypoints_np, np.arange(48).reshape([2, 2, 6, 2]))
        self.assertAllClose(masks_np, np.arange(64).reshape([2, 2, 4, 4]))
        self.assertAllClose(num_detections_np, [2, 1])

  def test_rewrite_nn_resize_op(self):
    g = tf.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
      y = array_ops.placeholder(dtypes.float32, shape=(8, 20, 20, 8))
      s = ops.nearest_neighbor_upsampling(x, 2)
      t = s + y
      exporter.rewrite_nn_resize_op()

    resize_op_found = False
    for op in g.get_operations():
      if op.type == 'ResizeNearestNeighbor':
        resize_op_found = True
        self.assertEqual(op.inputs[0], x)
        self.assertEqual(op.outputs[0].consumers()[0], t.op)
        break

    self.assertTrue(resize_op_found)

  def test_rewrite_nn_resize_op_quantized(self):
    g = tf.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
      x_conv = slim.conv2d(x, 8, 1)
      y = array_ops.placeholder(dtypes.float32, shape=(8, 20, 20, 8))
      s = ops.nearest_neighbor_upsampling(x_conv, 2)
      t = s + y

      graph_rewriter_config = graph_rewriter_pb2.GraphRewriter()
      graph_rewriter_config.quantization.delay = 500000
      graph_rewriter_fn = graph_rewriter_builder.build(
          graph_rewriter_config, is_training=False)
      graph_rewriter_fn()

      exporter.rewrite_nn_resize_op(is_quantized=True)

    resize_op_found = False
    for op in g.get_operations():
      if op.type == 'ResizeNearestNeighbor':
        resize_op_found = True
        self.assertEqual(op.inputs[0].op.type, 'FakeQuantWithMinMaxVars')
        self.assertEqual(op.outputs[0].consumers()[0], t.op)
        break

    self.assertTrue(resize_op_found)

  def test_rewrite_nn_resize_op_odd_size(self):
    g = tf.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
      s = ops.nearest_neighbor_upsampling(x, 2)
      t = s[:, :19, :19, :]
      exporter.rewrite_nn_resize_op()

    resize_op_found = False
    for op in g.get_operations():
      if op.type == 'ResizeNearestNeighbor':
        resize_op_found = True
        self.assertEqual(op.inputs[0], x)
        self.assertEqual(op.outputs[0].consumers()[0], t.op)
        break

    self.assertTrue(resize_op_found)

  def test_rewrite_nn_resize_op_quantized_odd_size(self):
    g = tf.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
      x_conv = slim.conv2d(x, 8, 1)
      s = ops.nearest_neighbor_upsampling(x_conv, 2)
      t = s[:, :19, :19, :]

      graph_rewriter_config = graph_rewriter_pb2.GraphRewriter()
      graph_rewriter_config.quantization.delay = 500000
      graph_rewriter_fn = graph_rewriter_builder.build(
          graph_rewriter_config, is_training=False)
      graph_rewriter_fn()

      exporter.rewrite_nn_resize_op(is_quantized=True)

    resize_op_found = False
    for op in g.get_operations():
      if op.type == 'ResizeNearestNeighbor':
        resize_op_found = True
        self.assertEqual(op.inputs[0].op.type, 'FakeQuantWithMinMaxVars')
        self.assertEqual(op.outputs[0].consumers()[0], t.op)
        break

    self.assertTrue(resize_op_found)

  def test_rewrite_nn_resize_op_multiple_path(self):
    g = tf.Graph()
    with g.as_default():
      with tf.name_scope('nearest_upsampling'):
        x_1 = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
        x_1_stack_1 = tf.stack([x_1] * 2, axis=3)
        x_1_reshape_1 = tf.reshape(x_1_stack_1, [8, 10, 20, 8])
        x_1_stack_2 = tf.stack([x_1_reshape_1] * 2, axis=2)
        x_1_reshape_2 = tf.reshape(x_1_stack_2, [8, 20, 20, 8])

      with tf.name_scope('nearest_upsampling'):
        x_2 = array_ops.placeholder(dtypes.float32, shape=(8, 10, 10, 8))
        x_2_stack_1 = tf.stack([x_2] * 2, axis=3)
        x_2_reshape_1 = tf.reshape(x_2_stack_1, [8, 10, 20, 8])
        x_2_stack_2 = tf.stack([x_2_reshape_1] * 2, axis=2)
        x_2_reshape_2 = tf.reshape(x_2_stack_2, [8, 20, 20, 8])

      t = x_1_reshape_2 + x_2_reshape_2

      exporter.rewrite_nn_resize_op()

    graph_def = g.as_graph_def()
    graph_def = strip_unused_lib.strip_unused(
        graph_def,
        input_node_names=[
            'nearest_upsampling/Placeholder', 'nearest_upsampling_1/Placeholder'
        ],
        output_node_names=['add'],
        placeholder_type_enum=dtypes.float32.as_datatype_enum)

    counter_resize_op = 0
    t_input_ops = [op.name for op in t.op.inputs]
    for node in graph_def.node:
      # Make sure Stacks are replaced.
      self.assertNotEqual(node.op, 'Pack')
      if node.op == 'ResizeNearestNeighbor':
        counter_resize_op += 1
        self.assertIn(six.ensure_str(node.name) + ':0', t_input_ops)
    self.assertEqual(counter_resize_op, 2)


if __name__ == '__main__':
  tf.test.main()
