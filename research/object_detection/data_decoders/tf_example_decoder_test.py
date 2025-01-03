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
"""Tests for object_detection.data_decoders.tf_example_decoder."""

import os
import numpy as np
import six
import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util
from object_detection.utils import test_case


class TfExampleDecoderTest(test_case.TestCase):

  def _create_encoded_and_decoded_data(self, data, encoding_type):
    if encoding_type == 'jpeg':
      encode_fn = tf.image.encode_jpeg
      decode_fn = tf.image.decode_jpeg
    elif encoding_type == 'png':
      encode_fn = tf.image.encode_png
      decode_fn = tf.image.decode_png
    else:
      raise ValueError('Invalid encoding type.')

    def prepare_data_fn():
      encoded_data = encode_fn(data)
      decoded_data = decode_fn(encoded_data)
      return encoded_data, decoded_data

    return self.execute_cpu(prepare_data_fn, [])

  def testDecodeAdditionalChannels(self):
    image = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(image, 'jpeg')

    additional_channel = np.random.randint(256, size=(4, 5, 1)).astype(np.uint8)
    (encoded_additional_channel,
     decoded_additional_channel) = self._create_encoded_and_decoded_data(
         additional_channel, 'jpeg')

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/additional_channels/encoded':
                      dataset_util.bytes_list_feature(
                          [encoded_additional_channel] * 2),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/source_id':
                      dataset_util.bytes_feature(six.b('image_id')),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          num_additional_channels=2)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        np.concatenate([decoded_additional_channel] * 2, axis=2),
        tensor_dict[fields.InputDataFields.image_additional_channels])

  def testDecodeJpegImage(self):
    image = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, decoded_jpeg = self._create_encoded_and_decoded_data(
        image, 'jpeg')

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/source_id':
                      dataset_util.bytes_feature(six.b('image_id')),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))
      self.assertAllEqual(
          (output[fields.InputDataFields.image].get_shape().as_list()),
          [None, None, 3])
      self.assertAllEqual(
          (output[fields.InputDataFields.original_image_spatial_shape]
           .get_shape().as_list()), [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(decoded_jpeg, tensor_dict[fields.InputDataFields.image])
    self.assertAllEqual([4, 5], tensor_dict[fields.InputDataFields.
                                            original_image_spatial_shape])
    self.assertEqual(
        six.b('image_id'), tensor_dict[fields.InputDataFields.source_id])

  def testDecodeImageKeyAndFilename(self):
    image = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(image, 'jpeg')

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/key/sha256':
                      dataset_util.bytes_feature(six.b('abc')),
                  'image/filename':
                      dataset_util.bytes_feature(six.b('filename'))
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertEqual(six.b('abc'), tensor_dict[fields.InputDataFields.key])
    self.assertEqual(
        six.b('filename'), tensor_dict[fields.InputDataFields.filename])

  def testDecodePngImage(self):
    image = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_png, decoded_png = self._create_encoded_and_decoded_data(
        image, 'png')

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_png),
                  'image/format':
                      dataset_util.bytes_feature(six.b('png')),
                  'image/source_id':
                      dataset_util.bytes_feature(six.b('image_id'))
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))
      self.assertAllEqual(
          (output[fields.InputDataFields.image].get_shape().as_list()),
          [None, None, 3])
      self.assertAllEqual(
          (output[fields.InputDataFields.original_image_spatial_shape]
           .get_shape().as_list()), [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(decoded_png, tensor_dict[fields.InputDataFields.image])
    self.assertAllEqual([4, 5], tensor_dict[fields.InputDataFields.
                                            original_image_spatial_shape])
    self.assertEqual(
        six.b('image_id'), tensor_dict[fields.InputDataFields.source_id])

  def testDecodePngInstanceMasks(self):
    image = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_png, _ = self._create_encoded_and_decoded_data(image, 'png')
    mask_1 = np.random.randint(0, 2, size=(10, 10, 1)).astype(np.uint8)
    mask_2 = np.random.randint(0, 2, size=(10, 10, 1)).astype(np.uint8)
    encoded_png_1, _ = self._create_encoded_and_decoded_data(mask_1, 'png')
    decoded_png_1 = np.squeeze(mask_1.astype(np.float32))
    encoded_png_2, _ = self._create_encoded_and_decoded_data(mask_2, 'png')
    decoded_png_2 = np.squeeze(mask_2.astype(np.float32))
    encoded_masks = [encoded_png_1, encoded_png_2]
    decoded_masks = np.stack([decoded_png_1, decoded_png_2])

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_png),
                  'image/format':
                      dataset_util.bytes_feature(six.b('png')),
                  'image/object/mask':
                      dataset_util.bytes_list_feature(encoded_masks)
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_instance_masks=True,
          instance_mask_type=input_reader_pb2.PNG_MASKS)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        decoded_masks,
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])

  def testDecodeEmptyPngInstanceMasks(self):
    image_tensor = np.random.randint(256, size=(10, 10, 3)).astype(np.uint8)
    encoded_png, _ = self._create_encoded_and_decoded_data(image_tensor, 'png')
    encoded_masks = []

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_png),
                  'image/format':
                      dataset_util.bytes_feature(six.b('png')),
                  'image/object/mask':
                      dataset_util.bytes_list_feature(encoded_masks),
                  'image/height':
                      dataset_util.int64_feature(10),
                  'image/width':
                      dataset_util.int64_feature(10),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_instance_masks=True,
          instance_mask_type=input_reader_pb2.PNG_MASKS)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks].shape,
        [0, 10, 10])

  def testDecodeBoundingBox(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

  def testDecodeKeypointDepth(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    keypoint_visibility = [1, 2, 0, 1, 0, 2]
    keypoint_depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    keypoint_depth_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/z':
                      dataset_util.float_list_feature(keypoint_depths),
                  'image/object/keypoint/z/weights':
                      dataset_util.float_list_feature(keypoint_depth_weights),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          num_keypoints=3, load_keypoint_depth_features=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual(
          (output[fields.InputDataFields.groundtruth_keypoint_depths].get_shape(
          ).as_list()), [2, 3])
      self.assertAllEqual(
          (output[fields.InputDataFields.groundtruth_keypoint_depth_weights]
           .get_shape().as_list()), [2, 3])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    expected_keypoint_depths = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    self.assertAllClose(
        expected_keypoint_depths,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_depths])

    expected_keypoint_depth_weights = [[1.0, 0.9, 0.8], [0.7, 0.6, 0.5]]
    self.assertAllClose(
        expected_keypoint_depth_weights,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_depth_weights])

  def testDecodeKeypointDepthNoDepth(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    keypoint_visibility = [1, 2, 0, 1, 0, 2]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          num_keypoints=3, load_keypoint_depth_features=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    expected_keypoints_depth_default = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    self.assertAllClose(
        expected_keypoints_depth_default,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_depths])
    self.assertAllClose(
        expected_keypoints_depth_default,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_depth_weights])

  def testDecodeKeypoint(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    keypoint_visibility = [1, 2, 0, 1, 0, 2]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(num_keypoints=3)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_keypoints].get_shape().as_list()),
                          [2, 3, 2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

    expected_keypoints = [
        [[0.0, 1.0], [1.0, 2.0], [np.nan, np.nan]],
        [[3.0, 4.0], [np.nan, np.nan], [5.0, 6.0]]]
    self.assertAllClose(
        expected_keypoints,
        tensor_dict[fields.InputDataFields.groundtruth_keypoints])

    expected_visibility = (
        (np.array(keypoint_visibility) > 0).reshape((2, 3)))
    self.assertAllEqual(
        expected_visibility,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_visibilities])

  def testDecodeKeypointNoInstance(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = []
    bbox_xmins = []
    bbox_ymaxs = []
    bbox_xmaxs = []
    keypoint_ys = []
    keypoint_xs = []
    keypoint_visibility = []

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(num_keypoints=3)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_keypoints].get_shape().as_list()),
                          [0, 3, 2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        [0, 4], tensor_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0, 3, 2],
        tensor_dict[fields.InputDataFields.groundtruth_keypoints].shape)

  def testDecodeKeypointWithText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes = [0, 1]
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    keypoint_visibility = [1, 2, 0, 1, 0, 2]
    keypoint_texts = [
        six.b('nose'), six.b('left_eye'), six.b('right_eye'), six.b('nose'),
        six.b('left_eye'), six.b('right_eye')
    ]

    label_map_string = """
      item: {
        id: 1
        name: 'face'
        display_name: 'face'
        keypoints {
         id: 0
         label: "nose"
        }
        keypoints {
         id: 2
         label: "right_eye"
        }
      }
      item: {
        id: 2
        name: 'person'
        display_name: 'person'
        keypoints {
         id: 1
         label: "left_eye"
        }
      }
    """
    label_map_proto_file = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.gfile.Open(label_map_proto_file, 'wb') as f:
      f.write(label_map_string)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
                  'image/object/keypoint/text':
                      dataset_util.bytes_list_feature(keypoint_texts),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(bbox_classes),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_proto_file, num_keypoints=5,
          use_keypoint_label_map=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_keypoints].get_shape().as_list()),
                          [None, 5, 2])
      return output

    output = self.execute_cpu(graph_fn, [])
    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        output[fields.InputDataFields.groundtruth_boxes])

    expected_keypoints = [[[0.0, 1.0], [1.0, 2.0], [np.nan, np.nan],
                           [np.nan, np.nan], [np.nan, np.nan]],
                          [[3.0, 4.0], [np.nan, np.nan], [5.0, 6.0],
                           [np.nan, np.nan], [np.nan, np.nan]]]
    self.assertAllClose(expected_keypoints,
                        output[fields.InputDataFields.groundtruth_keypoints])

    expected_visibility = (
        (np.array(keypoint_visibility) > 0).reshape((2, 3)))
    gt_kpts_vis_fld = fields.InputDataFields.groundtruth_keypoint_visibilities
    self.assertAllEqual(expected_visibility, output[gt_kpts_vis_fld][:, 0:3])
    # The additional keypoints should all have False visibility.
    self.assertAllEqual(
        np.zeros([2, 2], dtype=bool), output[gt_kpts_vis_fld][:, 3:])

  def testDecodeKeypointWithKptsLabelsNotInText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes = [0, 1]
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    keypoint_visibility = [1, 2, 0, 1, 0, 2]
    keypoint_texts = [
        six.b('nose'), six.b('left_eye'), six.b('right_eye'), six.b('nose'),
        six.b('left_eye'), six.b('right_eye')
    ]

    label_map_string = """
      item: {
        id: 1
        name: 'face'
        display_name: 'face'
        keypoints {
         id: 0
         label: "missing_part"
        }
        keypoints {
         id: 2
         label: "right_eye"
        }
        keypoints {
         id: 3
         label: "nose"
        }
      }
      item: {
        id: 2
        name: 'person'
        display_name: 'person'
        keypoints {
         id: 1
         label: "left_eye"
        }
      }
    """
    label_map_proto_file = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.gfile.Open(label_map_proto_file, 'wb') as f:
      f.write(label_map_string)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
                  'image/object/keypoint/visibility':
                      dataset_util.int64_list_feature(keypoint_visibility),
                  'image/object/keypoint/text':
                      dataset_util.bytes_list_feature(keypoint_texts),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(bbox_classes),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_proto_file, num_keypoints=5,
          use_keypoint_label_map=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_keypoints].get_shape().as_list()),
                          [None, 5, 2])
      return output

    output = self.execute_cpu(graph_fn, [])
    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        output[fields.InputDataFields.groundtruth_boxes])

    expected_keypoints = [[[np.nan, np.nan], [1., 2.], [np.nan, np.nan],
                           [0., 1.], [np.nan, np.nan]],
                          [[np.nan, np.nan], [np.nan, np.nan], [5., 6.],
                           [3., 4.], [np.nan, np.nan]]]

    gt_kpts_vis_fld = fields.InputDataFields.groundtruth_keypoint_visibilities
    self.assertAllClose(expected_keypoints,
                        output[fields.InputDataFields.groundtruth_keypoints])

    expected_visibility = [[False, True, False, True, False],
                           [False, False, True, True, False]]
    gt_kpts_vis_fld = fields.InputDataFields.groundtruth_keypoint_visibilities
    self.assertAllEqual(expected_visibility, output[gt_kpts_vis_fld])

  def testDecodeKeypointNoVisibilities(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/keypoint/y':
                      dataset_util.float_list_feature(keypoint_ys),
                  'image/object/keypoint/x':
                      dataset_util.float_list_feature(keypoint_xs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(num_keypoints=3)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_keypoints].get_shape().as_list()),
                          [2, 3, 2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

    expected_keypoints = (
        np.vstack([keypoint_ys, keypoint_xs]).transpose().reshape((2, 3, 2)))
    self.assertAllEqual(
        expected_keypoints,
        tensor_dict[fields.InputDataFields.groundtruth_keypoints])

    expected_visibility = np.ones((2, 3))
    self.assertAllEqual(
        expected_visibility,
        tensor_dict[fields.InputDataFields.groundtruth_keypoint_visibilities])

  def testDecodeDefaultGroundtruthWeights(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_boxes].get_shape().as_list()),
                          [None, 4])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllClose(tensor_dict[fields.InputDataFields.groundtruth_weights],
                        np.ones(2, dtype=np.float32))

  def testDecodeObjectLabel(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes = [0, 1]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(bbox_classes),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeMultiClassScores(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    flattened_multiclass_scores = [100., 50.] + [20., 30.]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/multiclass_scores':
                      dataset_util.float_list_feature(
                          flattened_multiclass_scores),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_multiclass_scores=True)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(flattened_multiclass_scores,
                        tensor_dict[fields.InputDataFields.multiclass_scores])

  def testDecodeEmptyMultiClassScores(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_multiclass_scores=True)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertEqual(
        (0,), tensor_dict[fields.InputDataFields.multiclass_scores].shape)

  def testDecodeObjectLabelNoText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes = [1, 2]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(bbox_classes),
              })).SerializeToString()
      label_map_string = """
        item {
          id:1
          name:'cat'
        }
        item {
          id:2
          name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)

      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [None])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes_text = [six.b('cat'), six.b('dog')]
    # Annotation label gets overridden by labelmap id.
    annotated_bbox_classes = [3, 4]
    expected_bbox_classes = [1, 2]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(annotated_bbox_classes),
              })).SerializeToString()
      label_map_string = """
        item {
          id:1
          name:'cat'
          keypoints {
            id: 0
            label: "nose"
          }
          keypoints {
            id: 1
            label: "left_eye"
          }
          keypoints {
            id: 2
            label: "right_eye"
          }
        }
        item {
          id:2
          name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)

      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(expected_bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelUnrecognizedName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes_text = [six.b('cat'), six.b('cheetah')]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
              })).SerializeToString()

      label_map_string = """
        item {
          id:2
          name:'cat'
        }
        item {
          id:1
          name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)
      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      output = example_decoder.decode(tf.convert_to_tensor(example))
      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [None])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual([2, -1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithMappingWithDisplayName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes_text = [six.b('cat'), six.b('dog')]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
              })).SerializeToString()

      label_map_string = """
        item {
          id:3
          display_name:'cat'
        }
        item {
          id:1
          display_name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)
      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [None])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual([3, 1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelUnrecognizedNameWithMappingWithDisplayName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes_text = [six.b('cat'), six.b('cheetah')]
    bbox_classes_id = [5, 6]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(bbox_classes_id),
              })).SerializeToString()

      label_map_string = """
        item {
          name:'/m/cat'
          id:3
          display_name:'cat'
        }
        item {
          name:'/m/dog'
          id:1
          display_name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)
      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual([3, -1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithMappingWithName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_classes_text = [six.b('cat'), six.b('dog')]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
              })).SerializeToString()

      label_map_string = """
        item {
          id:3
          name:'cat'
        }
        item {
          id:1
          name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)
      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [None])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual([3, 1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectArea(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    object_area = [100., 174.]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/area':
                      dataset_util.float_list_feature(object_area),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_area].get_shape().as_list()), [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(object_area,
                        tensor_dict[fields.InputDataFields.groundtruth_area])

  def testDecodeVerifiedNegClasses(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    neg_category_ids = [0, 5, 8]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/neg_category_ids':
                      dataset_util.int64_list_feature(neg_category_ids),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        neg_category_ids,
        tensor_dict[fields.InputDataFields.groundtruth_verified_neg_classes])

  def testDecodeNotExhaustiveClasses(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    not_exhaustive_category_ids = [0, 5, 8]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/not_exhaustive_category_ids':
                      dataset_util.int64_list_feature(
                          not_exhaustive_category_ids),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        not_exhaustive_category_ids,
        tensor_dict[fields.InputDataFields.groundtruth_not_exhaustive_classes])

  def testDecodeObjectIsCrowd(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    object_is_crowd = [0, 1]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/is_crowd':
                      dataset_util.int64_list_feature(object_is_crowd),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_is_crowd].get_shape().as_list()),
                          [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        [bool(item) for item in object_is_crowd],
        tensor_dict[fields.InputDataFields.groundtruth_is_crowd])

  def testDecodeObjectDifficult(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    object_difficult = [0, 1]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/difficult':
                      dataset_util.int64_list_feature(object_difficult),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_difficult].get_shape().as_list()),
                          [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        [bool(item) for item in object_difficult],
        tensor_dict[fields.InputDataFields.groundtruth_difficult])

  def testDecodeObjectGroupOf(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    object_group_of = [0, 1]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/group_of':
                      dataset_util.int64_list_feature(object_group_of),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_group_of].get_shape().as_list()),
                          [2])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        [bool(item) for item in object_group_of],
        tensor_dict[fields.InputDataFields.groundtruth_group_of])

  def testDecodeObjectWeight(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    object_weights = [0.75, 1.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/weight':
                      dataset_util.float_list_feature(object_weights),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_weights].get_shape().as_list()),
                          [None])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(object_weights,
                        tensor_dict[fields.InputDataFields.groundtruth_weights])

  def testDecodeClassConfidence(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    class_confidence = [0.0, 1.0, 0.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/class/confidence':
                      dataset_util.float_list_feature(class_confidence),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual(
          (output[fields.InputDataFields.groundtruth_image_confidences]
           .get_shape().as_list()), [3])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(
        class_confidence,
        tensor_dict[fields.InputDataFields.groundtruth_image_confidences])

  def testDecodeInstanceSegmentation(self):
    num_instances = 4
    image_height = 5
    image_width = 3

    # Randomly generate image.
    image_tensor = np.random.randint(
        256, size=(image_height, image_width, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances, image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/height':
                      dataset_util.int64_feature(image_height),
                  'image/width':
                      dataset_util.int64_feature(image_width),
                  'image/object/mask':
                      dataset_util.float_list_feature(instance_masks_flattened),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(object_classes)
              })).SerializeToString()
      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_instance_masks=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual(
          (output[fields.InputDataFields.groundtruth_instance_masks].get_shape(
          ).as_list()), [4, 5, 3])

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [4])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(
        instance_masks.astype(np.float32),
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_instance_mask_weights],
        [1, 1, 1, 1])
    self.assertAllEqual(object_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testInstancesNotAvailableByDefault(self):
    num_instances = 4
    image_height = 5
    image_width = 3
    # Randomly generate image.
    image_tensor = np.random.randint(
        256, size=(image_height, image_width, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances, image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/height':
                      dataset_util.int64_feature(image_height),
                  'image/width':
                      dataset_util.int64_feature(image_width),
                  'image/object/mask':
                      dataset_util.float_list_feature(instance_masks_flattened),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(object_classes)
              })).SerializeToString()
      example_decoder = tf_example_decoder.TfExampleDecoder()
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertNotIn(fields.InputDataFields.groundtruth_instance_masks,
                     tensor_dict)

  def testDecodeInstanceSegmentationWithWeights(self):
    num_instances = 4
    image_height = 5
    image_width = 3

    # Randomly generate image.
    image_tensor = np.random.randint(
        256, size=(image_height, image_width, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances, image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])
    instance_mask_weights = np.array([1, 1, 0, 1], dtype=np.float32)

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/height':
                      dataset_util.int64_feature(image_height),
                  'image/width':
                      dataset_util.int64_feature(image_width),
                  'image/object/mask':
                      dataset_util.float_list_feature(instance_masks_flattened),
                  'image/object/mask/weight':
                      dataset_util.float_list_feature(instance_mask_weights),
                  'image/object/class/label':
                      dataset_util.int64_list_feature(object_classes)
              })).SerializeToString()
      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_instance_masks=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))

      self.assertAllEqual(
          (output[fields.InputDataFields.groundtruth_instance_masks].get_shape(
          ).as_list()), [4, 5, 3])
      self.assertAllEqual(
          output[fields.InputDataFields.groundtruth_instance_mask_weights],
          [1, 1, 0, 1])

      self.assertAllEqual((output[
          fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                          [4])
      return output

    tensor_dict = self.execute_cpu(graph_fn, [])

    self.assertAllEqual(
        instance_masks.astype(np.float32),
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])
    self.assertAllEqual(object_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeImageLabels(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')

    def graph_fn_1():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
                  'image/format': dataset_util.bytes_feature(six.b('jpeg')),
                  'image/class/label': dataset_util.int64_list_feature([1, 2]),
              })).SerializeToString()
      example_decoder = tf_example_decoder.TfExampleDecoder()
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn_1, [])
    self.assertIn(fields.InputDataFields.groundtruth_image_classes, tensor_dict)
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_image_classes],
        np.array([1, 2]))

    def graph_fn_2():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/class/text':
                      dataset_util.bytes_list_feature(
                          [six.b('dog'), six.b('cat')]),
              })).SerializeToString()
      label_map_string = """
        item {
          id:3
          name:'cat'
        }
        item {
          id:1
          name:'dog'
        }
      """
      label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
      with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)
      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn_2, [])
    self.assertIn(fields.InputDataFields.groundtruth_image_classes, tensor_dict)
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_image_classes],
        np.array([1, 3]))

  def testDecodeContextFeatures(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    num_features = 8
    context_feature_length = 10
    context_features = np.random.random(num_features*context_feature_length)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/context_features':
                      dataset_util.float_list_feature(context_features),
                  'image/context_feature_length':
                      dataset_util.int64_feature(context_feature_length),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_context_features=True)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertAllClose(
        context_features.reshape(num_features, context_feature_length),
        tensor_dict[fields.InputDataFields.context_features])
    self.assertAllEqual(
        context_feature_length,
        tensor_dict[fields.InputDataFields.context_feature_length])

  def testContextFeaturesNotAvailableByDefault(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    num_features = 10
    context_feature_length = 10
    context_features = np.random.random(num_features*context_feature_length)

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/context_features':
                      dataset_util.float_list_feature(context_features),
                  'image/context_feature_length':
                      dataset_util.int64_feature(context_feature_length),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder()
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])
    self.assertNotIn(fields.InputDataFields.context_features,
                     tensor_dict)

  def testExpandLabels(self):
    label_map_string = """
      item {
        id:1
        name:'cat'
        ancestor_ids: 2
      }
      item {
        id:2
        name:'animal'
        descendant_ids: 1
      }
      item {
        id:3
        name:'man'
        ancestor_ids: 5
      }
      item {
        id:4
        name:'woman'
        display_name:'woman'
        ancestor_ids: 5
      }
      item {
        id:5
        name:'person'
        descendant_ids: 3
        descendant_ids: 4
      }
    """

    label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.gfile.Open(label_map_path, 'wb') as f:
      f.write(label_map_string)
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    bbox_classes_text = [six.b('cat'), six.b('cat')]
    bbox_group_of = [0, 1]
    image_class_text = [six.b('cat'), six.b('person')]
    image_confidence = [1.0, 0.0]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/class/text':
                      dataset_util.bytes_list_feature(bbox_classes_text),
                  'image/object/group_of':
                      dataset_util.int64_list_feature(bbox_group_of),
                  'image/class/text':
                      dataset_util.bytes_list_feature(image_class_text),
                  'image/class/confidence':
                      dataset_util.float_list_feature(image_confidence),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          label_map_proto_file=label_map_path, expand_hierarchy_labels=True)
      return example_decoder.decode(tf.convert_to_tensor(example))

    tensor_dict = self.execute_cpu(graph_fn, [])

    boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                       bbox_xmaxs]).transpose()
    expected_boxes = np.stack(
        [boxes[0, :], boxes[0, :], boxes[1, :], boxes[1, :]], axis=0)
    expected_boxes_class = np.array([1, 2, 1, 2])
    expected_boxes_group_of = np.array([0, 0, 1, 1])
    expected_image_class = np.array([1, 2, 3, 4, 5])
    expected_image_confidence = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])
    self.assertAllEqual(expected_boxes_class,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])
    self.assertAllEqual(
        expected_boxes_group_of,
        tensor_dict[fields.InputDataFields.groundtruth_group_of])
    self.assertAllEqual(
        expected_image_class,
        tensor_dict[fields.InputDataFields.groundtruth_image_classes])
    self.assertAllEqual(
        expected_image_confidence,
        tensor_dict[fields.InputDataFields.groundtruth_image_confidences])

  def testDecodeDensePose(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0, 2.0]
    bbox_xmins = [1.0, 5.0, 8.0]
    bbox_ymaxs = [2.0, 6.0, 1.0]
    bbox_xmaxs = [3.0, 7.0, 3.3]
    densepose_num = [0, 4, 2]
    densepose_part_index = [2, 2, 3, 4, 2, 9]
    densepose_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    densepose_y = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    densepose_u = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    densepose_v = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/densepose/num':
                      dataset_util.int64_list_feature(densepose_num),
                  'image/object/densepose/part_index':
                      dataset_util.int64_list_feature(densepose_part_index),
                  'image/object/densepose/x':
                      dataset_util.float_list_feature(densepose_x),
                  'image/object/densepose/y':
                      dataset_util.float_list_feature(densepose_y),
                  'image/object/densepose/u':
                      dataset_util.float_list_feature(densepose_u),
                  'image/object/densepose/v':
                      dataset_util.float_list_feature(densepose_v),

              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_dense_pose=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))
      dp_num_points = output[fields.InputDataFields.groundtruth_dp_num_points]
      dp_part_ids = output[fields.InputDataFields.groundtruth_dp_part_ids]
      dp_surface_coords = output[
          fields.InputDataFields.groundtruth_dp_surface_coords]
      return dp_num_points, dp_part_ids, dp_surface_coords

    dp_num_points, dp_part_ids, dp_surface_coords = self.execute_cpu(
        graph_fn, [])

    expected_dp_num_points = [0, 4, 2]
    expected_dp_part_ids = [
        [0, 0, 0, 0],
        [2, 2, 3, 4],
        [2, 9, 0, 0]
    ]
    expected_dp_surface_coords = np.array(
        [
            # Instance 0 (no points).
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
            # Instance 1 (4 points).
            [[0.9, 0.1, 0.99, 0.01],
             [0.8, 0.2, 0.98, 0.02],
             [0.7, 0.3, 0.97, 0.03],
             [0.6, 0.4, 0.96, 0.04]],
            # Instance 2 (2 points).
            [[0.5, 0.5, 0.95, 0.05],
             [0.4, 0.6, 0.94, 0.06],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
        ], dtype=np.float32)

    self.assertAllEqual(dp_num_points, expected_dp_num_points)
    self.assertAllEqual(dp_part_ids, expected_dp_part_ids)
    self.assertAllClose(dp_surface_coords, expected_dp_surface_coords)

  def testDecodeTrack(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg, _ = self._create_encoded_and_decoded_data(
        image_tensor, 'jpeg')
    bbox_ymins = [0.0, 4.0, 2.0]
    bbox_xmins = [1.0, 5.0, 8.0]
    bbox_ymaxs = [2.0, 6.0, 1.0]
    bbox_xmaxs = [3.0, 7.0, 3.3]
    track_labels = [0, 1, 2]

    def graph_fn():
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      dataset_util.bytes_feature(encoded_jpeg),
                  'image/format':
                      dataset_util.bytes_feature(six.b('jpeg')),
                  'image/object/bbox/ymin':
                      dataset_util.float_list_feature(bbox_ymins),
                  'image/object/bbox/xmin':
                      dataset_util.float_list_feature(bbox_xmins),
                  'image/object/bbox/ymax':
                      dataset_util.float_list_feature(bbox_ymaxs),
                  'image/object/bbox/xmax':
                      dataset_util.float_list_feature(bbox_xmaxs),
                  'image/object/track/label':
                      dataset_util.int64_list_feature(track_labels),
              })).SerializeToString()

      example_decoder = tf_example_decoder.TfExampleDecoder(
          load_track_id=True)
      output = example_decoder.decode(tf.convert_to_tensor(example))
      track_ids = output[fields.InputDataFields.groundtruth_track_ids]
      return track_ids

    track_ids = self.execute_cpu(graph_fn, [])

    expected_track_labels = [0, 1, 2]

    self.assertAllEqual(track_ids, expected_track_labels)


if __name__ == '__main__':
  tf.test.main()
