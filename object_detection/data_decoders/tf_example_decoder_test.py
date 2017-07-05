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

import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder


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

  def _Int64Feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _FloatFeature(self, value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _BytesFeature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def testDecodeJpegImage(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    decoded_jpeg = self._DecodeImage(encoded_jpeg)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/source_id': self._BytesFeature('image_id'),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.image].
                         get_shape().as_list()), [None, None, 3])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(decoded_jpeg, tensor_dict[fields.InputDataFields.image])
    self.assertEqual('image_id', tensor_dict[fields.InputDataFields.source_id])

  def testDecodeImageKeyAndFilename(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/key/sha256': self._BytesFeature('abc'),
        'image/filename': self._BytesFeature('filename')
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertEqual('abc', tensor_dict[fields.InputDataFields.key])
    self.assertEqual('filename', tensor_dict[fields.InputDataFields.filename])

  def testDecodePngImage(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_png = self._EncodeImage(image_tensor, encoding_type='png')
    decoded_png = self._DecodeImage(encoded_png, encoding_type='png')
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_png),
        'image/format': self._BytesFeature('png'),
        'image/source_id': self._BytesFeature('image_id')
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.image].
                         get_shape().as_list()), [None, None, 3])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(decoded_png, tensor_dict[fields.InputDataFields.image])
    self.assertEqual('image_id', tensor_dict[fields.InputDataFields.source_id])

  def testDecodeBoundingBox(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/bbox/ymin': self._FloatFeature(bbox_ymins),
        'image/object/bbox/xmin': self._FloatFeature(bbox_xmins),
        'image/object/bbox/ymax': self._FloatFeature(bbox_ymaxs),
        'image/object/bbox/xmax': self._FloatFeature(bbox_xmaxs),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_boxes].
                         get_shape().as_list()), [None, 4])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    expected_boxes = np.vstack([bbox_ymins, bbox_xmins,
                                bbox_ymaxs, bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])

  def testDecodeObjectLabel(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes = [0, 1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/class/label': self._Int64Feature(bbox_classes),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                        [None])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectArea(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_area = [100., 174.]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/area': self._FloatFeature(object_area),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_area].
                         get_shape().as_list()), [None])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(object_area,
                        tensor_dict[fields.InputDataFields.groundtruth_area])

  def testDecodeObjectIsCrowd(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_is_crowd = [0, 1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/is_crowd': self._Int64Feature(object_is_crowd),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_is_crowd].get_shape().as_list()),
                        [None])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([bool(item) for item in object_is_crowd],
                        tensor_dict[
                            fields.InputDataFields.groundtruth_is_crowd])

  def testDecodeObjectDifficult(self):
    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_difficult = [0, 1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/difficult': self._Int64Feature(object_difficult),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_difficult].get_shape().as_list()),
                        [None])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([bool(item) for item in object_difficult],
                        tensor_dict[
                            fields.InputDataFields.groundtruth_difficult])

  def testDecodeInstanceSegmentation(self):
    num_instances = 4
    image_height = 5
    image_width = 3

    # Randomly generate image.
    image_tensor = np.random.randint(255, size=(image_height,
                                                image_width,
                                                3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    # Randomly generate instance segmentation masks.
    instance_segmentation = (
        np.random.randint(2, size=(num_instances,
                                   image_height,
                                   image_width)).astype(np.int64))

    # Randomly generate class labels for each instance.
    instance_segmentation_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/height': self._Int64Feature([image_height]),
        'image/width': self._Int64Feature([image_width]),
        'image/segmentation/object': self._Int64Feature(
            instance_segmentation.flatten()),
        'image/segmentation/object/class': self._Int64Feature(
            instance_segmentation_classes)})).SerializeToString()
    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks].
        get_shape().as_list()), [None, None, None])

    self.assertAllEqual((
        tensor_dict[fields.InputDataFields.groundtruth_instance_classes].
        get_shape().as_list()), [None])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        instance_segmentation.astype(np.bool),
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])
    self.assertAllEqual(
        instance_segmentation_classes,
        tensor_dict[fields.InputDataFields.groundtruth_instance_classes])


if __name__ == '__main__':
  tf.test.main()
