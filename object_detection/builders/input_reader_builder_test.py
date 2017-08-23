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

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from object_detection.builders import input_reader_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2


class InputReaderBuilderTest(tf.test.TestCase):

  def create_tf_record(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/encoded': feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(value=[encoded_jpeg])),
        'image/format': feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(value=['jpeg'.encode('utf-8')])),
        'image/object/bbox/xmin': feature_pb2.Feature(
            float_list=feature_pb2.FloatList(value=[0.0])),
        'image/object/bbox/xmax': feature_pb2.Feature(
            float_list=feature_pb2.FloatList(value=[1.0])),
        'image/object/bbox/ymin': feature_pb2.Feature(
            float_list=feature_pb2.FloatList(value=[0.0])),
        'image/object/bbox/ymax': feature_pb2.Feature(
            float_list=feature_pb2.FloatList(value=[1.0])),
        'image/object/class/label': feature_pb2.Feature(
            int64_list=feature_pb2.Int64List(value=[2])),
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

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)

    self.assertEquals(
        (4, 5, 3), output_dict[fields.InputDataFields.image].shape)
    self.assertEquals(
        [2], output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEquals(
        (1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0])


if __name__ == '__main__':
  tf.test.main()
