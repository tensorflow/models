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
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from object_detection.builders import input_reader_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


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
