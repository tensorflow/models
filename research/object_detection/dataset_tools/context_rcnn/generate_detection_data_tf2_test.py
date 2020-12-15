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
import tensorflow as tf

from object_detection import exporter_lib_v2
from object_detection.builders import model_builder
from object_detection.core import model
from object_detection.protos import pipeline_pb2
from object_detection.utils import tf_version

if tf_version.is_tf2():
  from object_detection.dataset_tools.context_rcnn import generate_detection_data  # pylint:disable=g-import-not-at-top

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  mock = unittest.mock

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


class FakeModel(model.DetectionModel):

  def __init__(self, conv_weight_scalar=1.0):
    super(FakeModel, self).__init__(num_classes=5)
    self._conv = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, strides=(1, 1), padding='valid',
        kernel_initializer=tf.keras.initializers.Constant(
            value=conv_weight_scalar))

  def preprocess(self, inputs):
    true_image_shapes = []  # Doesn't matter for the fake model.
    return tf.identity(inputs), true_image_shapes

  def predict(self, preprocessed_inputs, true_image_shapes):
    return {'image': self._conv(preprocessed_inputs)}

  def postprocess(self, prediction_dict, true_image_shapes):
    with tf.control_dependencies(list(prediction_dict.values())):
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
    with tf.io.TFRecordWriter(filename) as writer:
      for value in entries:
        writer.write(value)
    yield filename
  finally:
    os.unlink(filename)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class GenerateDetectionDataTest(tf.test.TestCase):

  def _save_checkpoint_from_mock_model(self, checkpoint_path):
    """A function to save checkpoint from a fake Detection Model.

    Args:
      checkpoint_path: Path to save checkpoint from Fake model.
    """
    mock_model = FakeModel()
    fake_image = tf.zeros(shape=[1, 10, 10, 3], dtype=tf.float32)
    preprocessed_inputs, true_image_shapes = mock_model.preprocess(fake_image)
    predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
    mock_model.postprocess(predictions, true_image_shapes)
    ckpt = tf.train.Checkpoint(model=mock_model)
    exported_checkpoint_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)

  def _export_saved_model(self):
    tmp_dir = self.get_temp_dir()
    self._save_checkpoint_from_mock_model(tmp_dir)
    output_directory = os.path.join(tmp_dir, 'output')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    tf.io.gfile.makedirs(output_directory)
    with mock.patch.object(
        model_builder, 'build', autospec=True) as mock_builder:
      mock_builder.return_value = FakeModel()
      exporter_lib_v2.INPUT_BUILDER_UTIL_MAP['model_build'] = mock_builder
      output_directory = os.path.join(tmp_dir, 'output')
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      exporter_lib_v2.export_inference_graph(
          input_type='tf_example',
          pipeline_config=pipeline_config,
          trained_checkpoint_dir=tmp_dir,
          output_directory=output_directory)
      saved_model_path = os.path.join(output_directory, 'saved_model')
    return saved_model_path

  def _create_tf_example(self):
    with self.test_session():
      encoded_image = tf.io.encode_jpeg(
          tf.constant(np.ones((4, 6, 3)).astype(np.uint8))).numpy()

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
    inference_fn.setup()
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
      temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
      output_tfrecord = os.path.join(temp_dir, 'output_tfrecord')
      saved_model_path = self._export_saved_model()
      confidence_threshold = 0.8
      num_shards = 1
      pipeline_options = beam.options.pipeline_options.PipelineOptions(
          runner='DirectRunner')
      p = beam.Pipeline(options=pipeline_options)
      generate_detection_data.construct_pipeline(
          p, input_tfrecord, output_tfrecord, saved_model_path,
          confidence_threshold, num_shards)
      p.run()
      filenames = tf.io.gfile.glob(output_tfrecord + '-?????-of-?????')
      actual_output = []
      record_iterator = tf.data.TFRecordDataset(
          tf.convert_to_tensor(filenames)).as_numpy_iterator()
      for record in record_iterator:
        actual_output.append(record)
      self.assertEqual(len(actual_output), 1)
      self.assert_expected_example(tf.train.Example.FromString(
          actual_output[0]))


if __name__ == '__main__':
  tf.test.main()
