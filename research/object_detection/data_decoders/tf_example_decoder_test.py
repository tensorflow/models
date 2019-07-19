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
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class TfExampleDecoderTest(tf.test.TestCase):

  def _EncodeImage(self, image_tensor, encoding_type='jpeg'):
    with self.test_session():
      if encoding_type == 'jpeg':
        image_encoded = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
      elif encoding_type == 'png':
        image_encoded = tf.image.encode_png(tf.constant(image_tensor)).eval()
      else:
        raise ValueError('Invalid encoding type.')
    return image_encoded

  def _DecodeImage(self, image_encoded, encoding_type='jpeg'):
    with self.test_session():
      if encoding_type == 'jpeg':
        image_decoded = tf.image.decode_jpeg(tf.constant(image_encoded)).eval()
      elif encoding_type == 'png':
        image_decoded = tf.image.decode_png(tf.constant(image_encoded)).eval()
      else:
        raise ValueError('Invalid encoding type.')
    return image_decoded

  def testDecodeAdditionalChannels(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    additional_channel_tensor = np.random.randint(
        256, size=(4, 5, 1)).astype(np.uint8)
    encoded_additional_channel = self._EncodeImage(additional_channel_tensor)
    decoded_additional_channel = self._DecodeImage(encoded_additional_channel)

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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
      self.assertAllEqual(
          np.concatenate([decoded_additional_channel] * 2, axis=2),
          tensor_dict[fields.InputDataFields.image_additional_channels])

  def testDecodeJpegImage(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    decoded_jpeg = self._DecodeImage(encoded_jpeg)
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.image].
                         get_shape().as_list()), [None, None, 3])
    self.assertAllEqual((tensor_dict[fields.InputDataFields.
                                     original_image_spatial_shape].
                         get_shape().as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(decoded_jpeg, tensor_dict[fields.InputDataFields.image])
    self.assertAllEqual([4, 5], tensor_dict[fields.InputDataFields.
                                            original_image_spatial_shape])
    self.assertEqual(
        six.b('image_id'), tensor_dict[fields.InputDataFields.source_id])

  def testDecodeImageKeyAndFilename(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
                'image/key/sha256': dataset_util.bytes_feature(six.b('abc')),
                'image/filename': dataset_util.bytes_feature(six.b('filename'))
            })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertEqual(six.b('abc'), tensor_dict[fields.InputDataFields.key])
    self.assertEqual(
        six.b('filename'), tensor_dict[fields.InputDataFields.filename])

  def testDecodePngImage(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_png = self._EncodeImage(image_tensor, encoding_type='png')
    decoded_png = self._DecodeImage(encoded_png, encoding_type='png')
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': dataset_util.bytes_feature(encoded_png),
                'image/format': dataset_util.bytes_feature(six.b('png')),
                'image/source_id': dataset_util.bytes_feature(
                    six.b('image_id'))
            })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.image].
                         get_shape().as_list()), [None, None, 3])
    self.assertAllEqual((tensor_dict[fields.InputDataFields.
                                     original_image_spatial_shape].
                         get_shape().as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(decoded_png, tensor_dict[fields.InputDataFields.image])
    self.assertAllEqual([4, 5], tensor_dict[fields.InputDataFields.
                                            original_image_spatial_shape])
    self.assertEqual(
        six.b('image_id'), tensor_dict[fields.InputDataFields.source_id])

  def testDecodePngInstanceMasks(self):
    image_tensor = np.random.randint(256, size=(10, 10, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    mask_1 = np.random.randint(0, 2, size=(10, 10, 1)).astype(np.uint8)
    mask_2 = np.random.randint(0, 2, size=(10, 10, 1)).astype(np.uint8)
    encoded_png_1 = self._EncodeImage(mask_1, encoding_type='png')
    decoded_png_1 = np.squeeze(mask_1.astype(np.float32))
    encoded_png_2 = self._EncodeImage(mask_2, encoding_type='png')
    decoded_png_2 = np.squeeze(mask_2.astype(np.float32))
    encoded_masks = [encoded_png_1, encoded_png_2]
    decoded_masks = np.stack([decoded_png_1, decoded_png_2])
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    dataset_util.bytes_feature(encoded_jpeg),
                'image/format':
                    dataset_util.bytes_feature(six.b('jpeg')),
                'image/object/mask':
                    dataset_util.bytes_list_feature(encoded_masks)
            })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=True, instance_mask_type=input_reader_pb2.PNG_MASKS)
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        decoded_masks,
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])

  def testDecodeEmptyPngInstanceMasks(self):
    image_tensor = np.random.randint(256, size=(10, 10, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    encoded_masks = []
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    dataset_util.bytes_feature(encoded_jpeg),
                'image/format':
                    dataset_util.bytes_feature(six.b('jpeg')),
                'image/object/mask':
                    dataset_util.bytes_list_feature(encoded_masks),
                'image/height':
                    dataset_util.int64_feature(10),
                'image/width':
                    dataset_util.int64_feature(10),
            })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=True, instance_mask_type=input_reader_pb2.PNG_MASKS)
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
      self.assertAllEqual(
          tensor_dict[fields.InputDataFields.groundtruth_instance_masks].shape,
          [0, 10, 10])

  def testDecodeBoundingBox(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_boxes]
                         .get_shape().as_list()), [None, 4])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

  def testDecodeKeypoint(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_boxes]
                         .get_shape().as_list()), [None, 4])
    self.assertAllEqual(
        (tensor_dict[fields.InputDataFields.groundtruth_keypoints].get_shape()
         .as_list()), [2, 3, 2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    expected_boxes = np.vstack([bbox_ymins, bbox_xmins, bbox_ymaxs,
                                bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

    expected_keypoints = (
        np.vstack([keypoint_ys, keypoint_xs]).transpose().reshape((2, 3, 2)))
    self.assertAllEqual(
        expected_keypoints,
        tensor_dict[fields.InputDataFields.groundtruth_keypoints])

  def testDecodeDefaultGroundtruthWeights(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_boxes]
                         .get_shape().as_list()), [None, 4])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllClose(tensor_dict[fields.InputDataFields.groundtruth_weights],
                        np.ones(2, dtype=np.float32))

  def testDecodeObjectLabel(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes = [0, 1]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [2])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeMultiClassScores(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    flattened_multiclass_scores = [100., 50.] + [20., 30.]
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    dataset_util.bytes_feature(encoded_jpeg),
                'image/format':
                    dataset_util.bytes_feature(six.b('jpeg')),
                'image/object/class/multiclass_scores':
                    dataset_util.float_list_feature(flattened_multiclass_scores
                                                   ),
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
    self.assertAllEqual(flattened_multiclass_scores,
                        tensor_dict[fields.InputDataFields.multiclass_scores])

  def testDecodeEmptyMultiClassScores(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
    self.assertEqual(0,
                     tensor_dict[fields.InputDataFields.multiclass_scores].size)

  def testDecodeObjectLabelNoText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes = [1, 2]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [None])

    init = tf.tables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = [six.b('cat'), six.b('dog')]
    # Annotation label gets overridden by labelmap id.
    annotated_bbox_classes = [3, 4]
    expected_bbox_classes = [1, 2]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    init = tf.tables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(expected_bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelUnrecognizedName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = [six.b('cat'), six.b('cheetah')]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [None])

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([2, -1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithMappingWithDisplayName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = [six.b('cat'), six.b('dog')]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [None])

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([3, 1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelUnrecognizedNameWithMappingWithDisplayName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = [six.b('cat'), six.b('cheetah')]
    bbox_classes_id = [5, 6]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([3, -1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelWithMappingWithName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = [six.b('cat'), six.b('dog')]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [None])

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([3, 1],
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectArea(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_area = [100., 174.]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_area]
                         .get_shape().as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(object_area,
                        tensor_dict[fields.InputDataFields.groundtruth_area])

  def testDecodeObjectIsCrowd(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_is_crowd = [0, 1]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual(
        (tensor_dict[fields.InputDataFields.groundtruth_is_crowd].get_shape()
         .as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        [bool(item) for item in object_is_crowd],
        tensor_dict[fields.InputDataFields.groundtruth_is_crowd])

  def testDecodeObjectDifficult(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_difficult = [0, 1]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual(
        (tensor_dict[fields.InputDataFields.groundtruth_difficult].get_shape()
         .as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        [bool(item) for item in object_difficult],
        tensor_dict[fields.InputDataFields.groundtruth_difficult])

  def testDecodeObjectGroupOf(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_group_of = [0, 1]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual(
        (tensor_dict[fields.InputDataFields.groundtruth_group_of].get_shape()
         .as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        [bool(item) for item in object_group_of],
        tensor_dict[fields.InputDataFields.groundtruth_group_of])

  def testDecodeObjectWeight(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_weights = [0.75, 1.0]
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_weights]
                         .get_shape().as_list()), [None])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(object_weights,
                        tensor_dict[fields.InputDataFields.groundtruth_weights])

  def testDecodeInstanceSegmentation(self):
    num_instances = 4
    image_height = 5
    image_width = 3

    # Randomly generate image.
    image_tensor = np.random.randint(
        256, size=(image_height, image_width, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances, image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual(
        (tensor_dict[fields.InputDataFields.groundtruth_instance_masks]
         .get_shape().as_list()), [4, 5, 3])

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_classes]
                         .get_shape().as_list()), [4])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        instance_masks.astype(np.float32),
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])
    self.assertAllEqual(object_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testInstancesNotAvailableByDefault(self):
    num_instances = 4
    image_height = 5
    image_width = 3
    # Randomly generate image.
    image_tensor = np.random.randint(
        256, size=(image_height, image_width, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances, image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    self.assertTrue(
        fields.InputDataFields.groundtruth_instance_masks not in tensor_dict)

  def testDecodeImageLabels(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
                'image/format': dataset_util.bytes_feature(six.b('jpeg')),
                'image/class/label': dataset_util.int64_list_feature([1, 2]),
            })).SerializeToString()
    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
    self.assertTrue(
        fields.InputDataFields.groundtruth_image_classes in tensor_dict)
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_image_classes],
        np.array([1, 2]))
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
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      tensor_dict = sess.run(tensor_dict)
    self.assertTrue(
        fields.InputDataFields.groundtruth_image_classes in tensor_dict)
    self.assertAllEqual(
        tensor_dict[fields.InputDataFields.groundtruth_image_classes],
        np.array([1, 3]))


if __name__ == '__main__':
  tf.test.main()
