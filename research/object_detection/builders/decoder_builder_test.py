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

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import decoder_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


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
