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
"""Tests for generate_detection_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import tempfile
import unittest
import numpy as np
import six
import tensorflow.compat.v1 as tf

from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.dataset_tools.context_rcnn import generate_detection_data
from object_detection.protos import pipeline_pb2
from object_detection.utils import tf_version
from apache_beam import runners

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  mock = unittest.mock


class FakeModel(model.DetectionModel):
  """A Fake Detection model with expected output nodes from post-processing."""

  def preprocess(self, inputs):
    true_image_shapes = []  # Doesn't matter for the fake model.
    return tf.identity(inputs), true_image_shapes

  def predict(self, preprocessed_inputs, true_image_shapes):
    return {'image': tf.layers.conv2d(preprocessed_inputs, 3, 1)}

  def postprocess(self, prediction_dict, true_image_shapes):
    with tf.control_dependencies(prediction_dict.values()):
      postprocessed_tensors = {
          'detection_boxes': tf.constant([[[0.0, 0.1, 0.5, 0.6],
                                           [0.5, 0.5, 0.8, 0.8]]], tf.float32),
          'detection_scores': tf.constant([[0.95, 0.6]], tf.float32),
          'detection_multiclass_scores': tf.constant([[[0.1, 0.7, 0.2],
                                                       [0.3, 0.1, 0.6]]],
                                                     tf.float32),
          'detection_classes': tf.constant([[0, 1]], tf.float32),
          'num_detections': tf.constant([2], tf.float32)
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


@contextlib.contextmanager
def InMemoryTFRecord(entries):
  temp = tempfile.NamedTemporaryFile(delete=False)
  filename = temp.name
  try:
    with tf.python_io.TFRecordWriter(filename) as writer:
      for value in entries:
        writer.write(value)
    yield filename
  finally:
    os.unlink(filename)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class GenerateDetectionDataTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self, checkpoint_path):
    """A function to save checkpoint from a fake Detection Model.

    Args:
      checkpoint_path: Path to save checkpoint from Fake model.
    """
    g = tf.Graph()
    with g.as_default():
      mock_model = FakeModel(num_classes=5)
      preprocessed_inputs, true_image_shapes = mock_model.preprocess(
          tf.placeholder(tf.float32, shape=[None, None, None, 3]))
      predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
      mock_model.postprocess(predictions, true_image_shapes)
      tf.train.get_or_create_global_step()
      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(init)
        saver.save(sess, checkpoint_path)

  def _export_saved_model(self):
    tmp_dir = self.get_temp_dir()
    checkpoint_path = os.path.join(tmp_dir, 'model.ckpt')
    self._save_checkpoint_from_mock_model(checkpoint_path)
    output_directory = os.path.join(tmp_dir, 'output')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    tf.io.gfile.makedirs(output_directory)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel(num_classes=5)
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
      output_node_names = ','.join(outputs.keys())
      saver = tf.train.Saver()
      input_saver_def = saver.as_saver_def()
      frozen_graph_def = exporter.freeze_graph_with_def_protos(
          input_graph_def=tf.get_default_graph().as_graph_def(),
          input_saver_def=input_saver_def,
          input_checkpoint=checkpoint_path,
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
      return saved_model_path

  def _create_tf_example(self):
    with self.test_session():
      encoded_image = tf.image.encode_jpeg(
          tf.constant(np.ones((4, 6, 3)).astype(np.uint8))).eval()

    def BytesFeature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def Int64Feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': BytesFeature(encoded_image),
        'image/source_id': BytesFeature(b'image_id'),
        'image/height': Int64Feature(4),
        'image/width': Int64Feature(6),
        'image/object/class/label': Int64Feature(5),
        'image/object/class/text': BytesFeature(b'hyena'),
        'image/class/label': Int64Feature(5),
        'image/class/text': BytesFeature(b'hyena'),
    }))

    return example.SerializeToString()

  def assert_expected_example(self, example):
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0.0])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0.1])
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [0.5])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [0.6])
    self.assertAllClose(
        example.features.feature['image/object/class/score']
        .float_list.value, [0.95])
    self.assertAllClose(
        example.features.feature['image/object/class/label']
        .int64_list.value, [5])
    self.assertAllEqual(
        example.features.feature['image/object/class/text']
        .bytes_list.value, [b'hyena'])
    self.assertAllClose(
        example.features.feature['image/class/label']
        .int64_list.value, [5])
    self.assertAllEqual(
        example.features.feature['image/class/text']
        .bytes_list.value, [b'hyena'])

    # Check other essential attributes.
    self.assertAllEqual(
        example.features.feature['image/height'].int64_list.value, [4])
    self.assertAllEqual(
        example.features.feature['image/width'].int64_list.value, [6])
    self.assertAllEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [b'image_id'])
    self.assertTrue(
        example.features.feature['image/encoded'].bytes_list.value)

  def test_generate_detection_data_fn(self):
    saved_model_path = self._export_saved_model()
    confidence_threshold = 0.8
    inference_fn = generate_detection_data.GenerateDetectionDataFn(
        saved_model_path, confidence_threshold)
    inference_fn.start_bundle()
    generated_example = self._create_tf_example()
    self.assertAllEqual(tf.train.Example.FromString(
        generated_example).features.feature['image/object/class/label']
                        .int64_list.value, [5])
    self.assertAllEqual(tf.train.Example.FromString(
        generated_example).features.feature['image/object/class/text']
                        .bytes_list.value, [b'hyena'])
    output = inference_fn.process(generated_example)
    output_example = output[0]

    self.assertAllEqual(
        output_example.features.feature['image/object/class/label']
        .int64_list.value, [5])
    self.assertAllEqual(output_example.features.feature['image/width']
                        .int64_list.value, [6])

    self.assert_expected_example(output_example)

  def test_beam_pipeline(self):
    with InMemoryTFRecord([self._create_tf_example()]) as input_tfrecord:
      runner = runners.DirectRunner()
      temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
      output_tfrecord = os.path.join(temp_dir, 'output_tfrecord')
      saved_model_path = self._export_saved_model()
      confidence_threshold = 0.8
      num_shards = 1
      pipeline = generate_detection_data.construct_pipeline(
          input_tfrecord, output_tfrecord, saved_model_path,
          confidence_threshold, num_shards)
      runner.run(pipeline)
      filenames = tf.io.gfile.glob(output_tfrecord + '-?????-of-?????')
      actual_output = []
      record_iterator = tf.python_io.tf_record_iterator(path=filenames[0])
      for record in record_iterator:
        actual_output.append(record)
      self.assertEqual(len(actual_output), 1)
      self.assert_expected_example(tf.train.Example.FromString(
          actual_output[0]))


if __name__ == '__main__':
  tf.test.main()
