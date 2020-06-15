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
"""Tests for decoder_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import decoder_builder
from object_detection.core import standard_fields as fields
from object_detection.dataset_tools import seq_example_util
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  parent_path = os.path.dirname(tf.resource_loader.get_data_files_path())
  return os.path.join(parent_path, 'data',
                      'pet_label_map.pbtxt')


class DecoderBuilderTest(tf.test.TestCase):

  def _make_serialized_tf_example(self, has_additional_channels=False):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    additional_channels_tensor = np.random.randint(
        255, size=(4, 5, 1)).astype(np.uint8)
    flat_mask = (4 * 5) * [1.0]
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
      encoded_additional_channels_jpeg = tf.image.encode_jpeg(
          tf.constant(additional_channels_tensor)).eval()
    features = {
        'image/source_id': dataset_util.bytes_feature('0'.encode()),
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
    }
    if has_additional_channels:
      additional_channels_key = 'image/additional_channels/encoded'
      features[additional_channels_key] = dataset_util.bytes_list_feature(
          [encoded_additional_channels_jpeg] * 2)
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

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

  def _make_serialized_tf_sequence_example(self):
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
    return sequence_example_serialized

  def test_build_tf_record_input_reader(self):
    input_reader_text_proto = 'tf_record_input_reader {}'
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Parse(input_reader_text_proto, input_reader_proto)

    decoder = decoder_builder.build(input_reader_proto)
    tensor_dict = decoder.decode(self._make_serialized_tf_example())

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    self.assertNotIn(
        fields.InputDataFields.groundtruth_instance_masks, output_dict)
    self.assertEqual((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual([2],
                        output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])

  def test_build_tf_record_input_reader_sequence_example(self):
    label_map_path = _get_labelmap_path()
    input_reader_text_proto = """
      input_type: TF_SEQUENCE_EXAMPLE
      tf_record_input_reader {}
    """
    input_reader_proto = input_reader_pb2.InputReader()
    input_reader_proto.label_map_path = label_map_path
    text_format.Parse(input_reader_text_proto, input_reader_proto)

    decoder = decoder_builder.build(input_reader_proto)
    tensor_dict = decoder.decode(self._make_serialized_tf_sequence_example())

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
    # Sequence example images are encoded.
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

  def test_build_tf_record_input_reader_and_load_instance_masks(self):
    input_reader_text_proto = """
      load_instance_masks: true
      tf_record_input_reader {}
    """
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Parse(input_reader_text_proto, input_reader_proto)

    decoder = decoder_builder.build(input_reader_proto)
    tensor_dict = decoder.decode(self._make_serialized_tf_example())

    with tf.train.MonitoredSession() as sess:
      output_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        (1, 4, 5),
        output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)


if __name__ == '__main__':
  tf.test.main()
