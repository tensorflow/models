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
"""Tests for tf_sequence_example_decoder.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_sequence_example_decoder
from object_detection.dataset_tools import seq_example_util
from object_detection.utils import test_case


class TfSequenceExampleDecoderTest(test_case.TestCase):

  def _create_label_map(self, path):
    label_map_text = """
      item {
        name: "dog"
        id: 1
      }
      item {
        name: "cat"
        id: 2
      }
      item {
        name: "panda"
        id: 4
      }
    """
    with tf.gfile.Open(path, 'wb') as f:
      f.write(label_map_text)

  def _make_random_serialized_jpeg_images(self, num_frames, image_height,
                                          image_width):
    def graph_fn():
      images = tf.cast(tf.random.uniform(
          [num_frames, image_height, image_width, 3],
          maxval=256,
          dtype=tf.int32), dtype=tf.uint8)
      images_list = tf.unstack(images, axis=0)
      return [tf.io.encode_jpeg(image) for image in images_list]
    encoded_images = self.execute(graph_fn, [])
    return encoded_images

  def test_decode_sequence_example(self):
    num_frames = 4
    image_height = 20
    image_width = 30

    expected_groundtruth_boxes = [
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.2, 0.2, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ]
    expected_groundtruth_classes = [
        [-1, -1],
        [-1, 1],
        [1, 2],
        [-1, -1]
    ]

    flds = fields.InputDataFields
    encoded_images = self._make_random_serialized_jpeg_images(
        num_frames, image_height, image_width)

    def graph_fn():
      label_map_proto_file = os.path.join(self.get_temp_dir(), 'labelmap.pbtxt')
      self._create_label_map(label_map_proto_file)
      decoder = tf_sequence_example_decoder.TfSequenceExampleDecoder(
          label_map_proto_file=label_map_proto_file)
      sequence_example_serialized = seq_example_util.make_sequence_example(
          dataset_name='video_dataset',
          video_id='video',
          encoded_images=encoded_images,
          image_height=image_height,
          image_width=image_width,
          image_format='JPEG',
          image_source_ids=[str(i) for i in range(num_frames)],
          is_annotated=[[1], [1], [1], [1]],
          bboxes=[
              [[0., 0., 1., 1.]],  # Frame 0.
              [[0.2, 0.2, 1., 1.],
               [0., 0., 1., 1.]],  # Frame 1.
              [[0., 0., 1., 1.],  # Frame 2.
               [0.1, 0.1, 0.2, 0.2]],
              [[]],  # Frame 3.
          ],
          label_strings=[
              ['fox'],  # Frame 0. Fox will be filtered out.
              ['fox', 'dog'],  # Frame 1. Fox will be filtered out.
              ['dog', 'cat'],  # Frame 2.
              [],  # Frame 3
          ]).SerializeToString()

      example_string_tensor = tf.convert_to_tensor(sequence_example_serialized)
      return decoder.decode(example_string_tensor)

    tensor_dict_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_groundtruth_boxes,
                        tensor_dict_out[flds.groundtruth_boxes])
    self.assertAllEqual(expected_groundtruth_classes,
                        tensor_dict_out[flds.groundtruth_classes])

  def test_decode_sequence_example_context(self):
    num_frames = 4
    image_height = 20
    image_width = 30

    expected_groundtruth_boxes = [
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.2, 0.2, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ]
    expected_groundtruth_classes = [
        [-1, -1],
        [-1, 1],
        [1, 2],
        [-1, -1]
    ]

    expected_context_features = np.array(
        [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32)

    flds = fields.InputDataFields
    encoded_images = self._make_random_serialized_jpeg_images(
        num_frames, image_height, image_width)

    def graph_fn():
      label_map_proto_file = os.path.join(self.get_temp_dir(), 'labelmap.pbtxt')
      self._create_label_map(label_map_proto_file)
      decoder = tf_sequence_example_decoder.TfSequenceExampleDecoder(
          label_map_proto_file=label_map_proto_file,
          load_context_features=True)
      sequence_example_serialized = seq_example_util.make_sequence_example(
          dataset_name='video_dataset',
          video_id='video',
          encoded_images=encoded_images,
          image_height=image_height,
          image_width=image_width,
          image_format='JPEG',
          image_source_ids=[str(i) for i in range(num_frames)],
          is_annotated=[[1], [1], [1], [1]],
          bboxes=[
              [[0., 0., 1., 1.]],  # Frame 0.
              [[0.2, 0.2, 1., 1.],
               [0., 0., 1., 1.]],  # Frame 1.
              [[0., 0., 1., 1.],  # Frame 2.
               [0.1, 0.1, 0.2, 0.2]],
              [[]],  # Frame 3.
          ],
          label_strings=[
              ['fox'],  # Frame 0. Fox will be filtered out.
              ['fox', 'dog'],  # Frame 1. Fox will be filtered out.
              ['dog', 'cat'],  # Frame 2.
              [],  # Frame 3
          ],
          context_features=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
          context_feature_length=[3],
          context_features_image_id_list=[b'im_1', b'im_2']
          ).SerializeToString()

      example_string_tensor = tf.convert_to_tensor(sequence_example_serialized)
      return decoder.decode(example_string_tensor)

    tensor_dict_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_groundtruth_boxes,
                        tensor_dict_out[flds.groundtruth_boxes])
    self.assertAllEqual(expected_groundtruth_classes,
                        tensor_dict_out[flds.groundtruth_classes])
    self.assertAllClose(expected_context_features,
                        tensor_dict_out[flds.context_features])

  def test_decode_sequence_example_context_image_id_list(self):
    num_frames = 4
    image_height = 20
    image_width = 30

    expected_groundtruth_boxes = [
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.2, 0.2, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ]
    expected_groundtruth_classes = [
        [-1, -1],
        [-1, 1],
        [1, 2],
        [-1, -1]
    ]

    expected_context_image_ids = [b'im_1', b'im_2']

    flds = fields.InputDataFields
    encoded_images = self._make_random_serialized_jpeg_images(
        num_frames, image_height, image_width)

    def graph_fn():
      label_map_proto_file = os.path.join(self.get_temp_dir(), 'labelmap.pbtxt')
      self._create_label_map(label_map_proto_file)
      decoder = tf_sequence_example_decoder.TfSequenceExampleDecoder(
          label_map_proto_file=label_map_proto_file,
          load_context_image_ids=True)
      sequence_example_serialized = seq_example_util.make_sequence_example(
          dataset_name='video_dataset',
          video_id='video',
          encoded_images=encoded_images,
          image_height=image_height,
          image_width=image_width,
          image_format='JPEG',
          image_source_ids=[str(i) for i in range(num_frames)],
          is_annotated=[[1], [1], [1], [1]],
          bboxes=[
              [[0., 0., 1., 1.]],  # Frame 0.
              [[0.2, 0.2, 1., 1.],
               [0., 0., 1., 1.]],  # Frame 1.
              [[0., 0., 1., 1.],  # Frame 2.
               [0.1, 0.1, 0.2, 0.2]],
              [[]],  # Frame 3.
          ],
          label_strings=[
              ['fox'],  # Frame 0. Fox will be filtered out.
              ['fox', 'dog'],  # Frame 1. Fox will be filtered out.
              ['dog', 'cat'],  # Frame 2.
              [],  # Frame 3
          ],
          context_features=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
          context_feature_length=[3],
          context_features_image_id_list=[b'im_1', b'im_2']
          ).SerializeToString()

      example_string_tensor = tf.convert_to_tensor(sequence_example_serialized)
      return decoder.decode(example_string_tensor)

    tensor_dict_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_groundtruth_boxes,
                        tensor_dict_out[flds.groundtruth_boxes])
    self.assertAllEqual(expected_groundtruth_classes,
                        tensor_dict_out[flds.groundtruth_classes])
    self.assertAllEqual(expected_context_image_ids,
                        tensor_dict_out[flds.context_features_image_id_list])

  def test_decode_sequence_example_negative_clip(self):
    num_frames = 4
    image_height = 20
    image_width = 30

    expected_groundtruth_boxes = -1 * np.ones((4, 0, 4))
    expected_groundtruth_classes = -1 * np.ones((4, 0))

    flds = fields.InputDataFields

    encoded_images = self._make_random_serialized_jpeg_images(
        num_frames, image_height, image_width)

    def graph_fn():
      sequence_example_serialized = seq_example_util.make_sequence_example(
          dataset_name='video_dataset',
          video_id='video',
          encoded_images=encoded_images,
          image_height=image_height,
          image_width=image_width,
          image_format='JPEG',
          image_source_ids=[str(i) for i in range(num_frames)],
          bboxes=[
              [[]],
              [[]],
              [[]],
              [[]]
          ],
          label_strings=[
              [],
              [],
              [],
              []
          ]).SerializeToString()
      example_string_tensor = tf.convert_to_tensor(sequence_example_serialized)

      label_map_proto_file = os.path.join(self.get_temp_dir(), 'labelmap.pbtxt')
      self._create_label_map(label_map_proto_file)
      decoder = tf_sequence_example_decoder.TfSequenceExampleDecoder(
          label_map_proto_file=label_map_proto_file)
      return decoder.decode(example_string_tensor)

    tensor_dict_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_groundtruth_boxes,
                        tensor_dict_out[flds.groundtruth_boxes])
    self.assertAllEqual(expected_groundtruth_classes,
                        tensor_dict_out[flds.groundtruth_classes])


if __name__ == '__main__':
  tf.test.main()
