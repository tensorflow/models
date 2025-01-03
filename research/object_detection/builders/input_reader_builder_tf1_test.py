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

"""Tests for input_reader_builder."""

import os
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import input_reader_builder
from object_detection.core import standard_fields as fields
from object_detection.dataset_tools import seq_example_util
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util
from object_detection.utils import tf_version


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  parent_path = os.path.dirname(tf.resource_loader.get_data_files_path())
  return os.path.join(parent_path, 'data',
                      'pet_label_map.pbtxt')


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class InputReaderBuilderTest(tf.test.TestCase):

  def create_tf_record(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    flat_mask = (4 * 5) * [1.0]
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/height': dataset_util.int64_feature(4),
        'image/width': dataset_util.int64_feature(5),
        'image/object/bbox/xmin': dataset_util.float_list_feature([0.0]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([1.0]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([0.0]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([1.0]),
        'image/object/class/label': dataset_util.int64_list_feature([2]),
        'image/object/mask': dataset_util.float_list_feature(flat_mask),
    }))
    writer.write(example.SerializeToString())
    writer.close()

    return path

  def _make_random_serialized_jpeg_images(self, num_frames, image_height,
                                          image_width):
    images = tf.cast(tf.random.uniform(
        [num_frames, image_height, image_width, 3],
        maxval=256,
        dtype=tf.int32), dtype=tf.uint8)
    images_list = tf.unstack(images, axis=0)
    encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
    with tf.Session() as sess:
      encoded_images = sess.run(encoded_images_list)
    return encoded_images

  def create_tf_record_sequence_example(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)
    num_frames = 4
    image_height = 20
    image_width = 30
    image_source_ids = [str(i) for i in range(num_frames)]
    with self.test_session():
      encoded_images = self._make_random_serialized_jpeg_images(
          num_frames, image_height, image_width)
      sequence_example_serialized = seq_example_util.make_sequence_example(
          dataset_name='video_dataset',
          video_id='video',
          encoded_images=encoded_images,
          image_height=image_height,
          image_width=image_width,
          image_source_ids=image_source_ids,
          image_format='JPEG',
          is_annotated=[[1], [1], [1], [1]],
          bboxes=[
              [[]],  # Frame 0.
              [[0., 0., 1., 1.]],  # Frame 1.
              [[0., 0., 1., 1.],
               [0.1, 0.1, 0.2, 0.2]],  # Frame 2.
              [[]],  # Frame 3.
          ],
          label_strings=[
              [],  # Frame 0.
              ['Abyssinian'],  # Frame 1.
              ['Abyssinian', 'american_bulldog'],  # Frame 2.
              [],  # Frame 3
          ]).SerializeToString()

    writer.write(sequence_example_serialized)
    writer.close()

    return path

  def create_tf_record_with_context(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    flat_mask = (4 * 5) * [1.0]
    context_features = (10 * 3) * [1.0]
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    dataset_util.bytes_feature(encoded_jpeg),
                'image/format':
                    dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/height':
                    dataset_util.int64_feature(4),
                'image/width':
                    dataset_util.int64_feature(5),
                'image/object/bbox/xmin':
                    dataset_util.float_list_feature([0.0]),
                'image/object/bbox/xmax':
                    dataset_util.float_list_feature([1.0]),
                'image/object/bbox/ymin':
                    dataset_util.float_list_feature([0.0]),
                'image/object/bbox/ymax':
                    dataset_util.float_list_feature([1.0]),
                'image/object/class/label':
                    dataset_util.int64_list_feature([2]),
                'image/object/mask':
                    dataset_util.float_list_feature(flat_mask),
                'image/context_features':
                    dataset_util.float_list_feature(context_features),
                'image/context_feature_length':
                    dataset_util.int64_list_feature([10]),
            }))
    writer.write(example.SerializeToString())
    writer.close()

    return path

  def test_build_tf_record_input_reader(self):
    tf_record_path = self.create_tf_record()

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    tensor_dict = input_reader_builder.build(input_reader_proto)

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    self.assertNotIn(fields.InputDataFields.groundtruth_instance_masks,
                     output_dict)
    self.assertEqual((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
    self.assertEqual([2],
                     output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])

  def test_build_tf_record_input_reader_sequence_example(self):
    tf_record_path = self.create_tf_record_sequence_example()

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      input_type: TF_SEQUENCE_EXAMPLE
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    input_reader_proto.label_map_path = _get_labelmap_path()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    tensor_dict = input_reader_builder.build(input_reader_proto)

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    expected_groundtruth_classes = [[-1, -1], [1, -1], [1, 2], [-1, -1]]
    expected_groundtruth_boxes = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                                  [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
                                  [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    expected_num_groundtruth_boxes = [0, 1, 2, 0]

    self.assertNotIn(
        fields.InputDataFields.groundtruth_instance_masks, output_dict)
    # sequence example images are encoded
    self.assertEqual((4,), output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual(expected_groundtruth_classes,
                        output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (4, 2, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllClose(expected_groundtruth_boxes,
                        output_dict[fields.InputDataFields.groundtruth_boxes])
    self.assertAllClose(
        expected_num_groundtruth_boxes,
        output_dict[fields.InputDataFields.num_groundtruth_boxes])

  def test_build_tf_record_input_reader_with_context(self):
    tf_record_path = self.create_tf_record_with_context()

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    input_reader_proto.load_context_features = True
    tensor_dict = input_reader_builder.build(input_reader_proto)

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    self.assertNotIn(fields.InputDataFields.groundtruth_instance_masks,
                     output_dict)
    self.assertEqual((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
    self.assertEqual([2],
                     output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])
    self.assertAllEqual(
        (3, 10), output_dict[fields.InputDataFields.context_features].shape)
    self.assertAllEqual(
        (10), output_dict[fields.InputDataFields.context_feature_length])

  def test_build_tf_record_input_reader_and_load_instance_masks(self):
    tf_record_path = self.create_tf_record()

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      load_instance_masks: true
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    tensor_dict = input_reader_builder.build(input_reader_proto)

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    self.assertEqual((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
    self.assertEqual([2],
                     output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])
    self.assertAllEqual(
        (1, 4, 5),
        output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)

  def test_raises_error_with_no_input_paths(self):
    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      load_instance_masks: true
    """
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    with self.assertRaises(ValueError):
      input_reader_builder.build(input_reader_proto)

if __name__ == '__main__':
  tf.test.main()
