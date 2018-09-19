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
import tensorflow as tf

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2

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

  def _Int64Feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _FloatFeature(self, value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _BytesFeature(self, value):
    if isinstance(value, list):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _Int64FeatureFromList(self, ndarray):
    return feature_pb2.Feature(
        int64_list=feature_pb2.Int64List(value=ndarray.flatten().tolist()))

  def _BytesFeatureFromList(self, ndarray):
    values = ndarray.flatten().tolist()
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=values))

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
                    self._BytesFeature(encoded_jpeg),
                'image/additional_channels/encoded':
                    self._BytesFeatureFromList(
                        np.array([encoded_additional_channel] * 2)),
                'image/format':
                    self._BytesFeature('jpeg'),
                'image/source_id':
                    self._BytesFeature('image_id'),
            })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder(
        num_additional_channels=2)
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)
      self.assertAllEqual(
          np.concatenate([decoded_additional_channel] * 2, axis=2),
          tensor_dict[fields.InputDataFields.image_additional_channels])

  def testDecodeExampleWithBranchedBackupHandler(self):
    example1 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/text':
                    self._BytesFeatureFromList(
                        np.array(['cat', 'dog', 'guinea pig'])),
                'image/object/class/label':
                    self._Int64FeatureFromList(np.array([42, 10, 900]))
            }))
    example2 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/text':
                    self._BytesFeatureFromList(
                        np.array(['cat', 'dog', 'guinea pig'])),
            }))
    example3 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/label':
                    self._Int64FeatureFromList(np.array([42, 10, 901]))
            }))
    # 'dog' -> 0, 'guinea pig' -> 1, 'cat' -> 2
    table = lookup_ops.index_table_from_tensor(
        constant_op.constant(['dog', 'guinea pig', 'cat']))
    keys_to_features = {
        'image/object/class/text': parsing_ops.VarLenFeature(dtypes.string),
        'image/object/class/label': parsing_ops.VarLenFeature(dtypes.int64),
    }
    backup_handler = tf_example_decoder.BackupHandler(
        handler=slim_example_decoder.Tensor('image/object/class/label'),
        backup=tf_example_decoder.LookupTensor('image/object/class/text',
                                               table))
    items_to_handlers = {
        'labels': backup_handler,
    }
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    obtained_class_ids_each_example = []
    with self.test_session() as sess:
      sess.run(lookup_ops.tables_initializer())
      for example in [example1, example2, example3]:
        serialized_example = array_ops.reshape(
            example.SerializeToString(), shape=[])
        obtained_class_ids_each_example.append(
            decoder.decode(serialized_example)[0].eval())

    self.assertAllClose([42, 10, 900], obtained_class_ids_each_example[0])
    self.assertAllClose([2, 0, 1], obtained_class_ids_each_example[1])
    self.assertAllClose([42, 10, 901], obtained_class_ids_each_example[2])

  def testDecodeExampleWithBranchedLookup(self):

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/object/class/text': self._BytesFeatureFromList(
            np.array(['cat', 'dog', 'guinea pig'])),
    }))
    serialized_example = example.SerializeToString()
    # 'dog' -> 0, 'guinea pig' -> 1, 'cat' -> 2
    table = lookup_ops.index_table_from_tensor(
        constant_op.constant(['dog', 'guinea pig', 'cat']))

    with self.test_session() as sess:
      sess.run(lookup_ops.tables_initializer())

      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/object/class/text': parsing_ops.VarLenFeature(dtypes.string),
      }

      items_to_handlers = {
          'labels':
              tf_example_decoder.LookupTensor('image/object/class/text', table),
      }

      decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)
      obtained_class_ids = decoder.decode(serialized_example)[0].eval()

    self.assertAllClose([2, 0, 1], obtained_class_ids)

  def testDecodeJpegImage(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
                'image/encoded': self._BytesFeature(encoded_jpeg),
                'image/format': self._BytesFeature('jpeg'),
                'image/object/mask': self._BytesFeature(encoded_masks)
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
                'image/encoded': self._BytesFeature(encoded_jpeg),
                'image/format': self._BytesFeature('jpeg'),
                'image/object/mask': self._BytesFeature(encoded_masks),
                'image/height': self._Int64Feature([10]),
                'image/width': self._Int64Feature([10]),
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
    self.assertAllEqual(
        2, tensor_dict[fields.InputDataFields.num_groundtruth_boxes])

  @test_util.enable_c_shapes
  def testDecodeKeypoint(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_ymins = [0.0, 4.0]
    bbox_xmins = [1.0, 5.0]
    bbox_ymaxs = [2.0, 6.0]
    bbox_xmaxs = [3.0, 7.0]
    keypoint_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    keypoint_xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/bbox/ymin': self._FloatFeature(bbox_ymins),
        'image/object/bbox/xmin': self._FloatFeature(bbox_xmins),
        'image/object/bbox/ymax': self._FloatFeature(bbox_ymaxs),
        'image/object/bbox/xmax': self._FloatFeature(bbox_xmaxs),
        'image/object/keypoint/y': self._FloatFeature(keypoint_ys),
        'image/object/keypoint/x': self._FloatFeature(keypoint_xs),
    })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder(num_keypoints=3)
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[fields.InputDataFields.groundtruth_boxes].
                         get_shape().as_list()), [None, 4])
    self.assertAllEqual((tensor_dict[fields.InputDataFields.
                                     groundtruth_keypoints].
                         get_shape().as_list()), [2, 3, 2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    expected_boxes = np.vstack([bbox_ymins, bbox_xmins,
                                bbox_ymaxs, bbox_xmaxs]).transpose()
    self.assertAllEqual(expected_boxes,
                        tensor_dict[fields.InputDataFields.groundtruth_boxes])
    self.assertAllEqual(
        2, tensor_dict[fields.InputDataFields.num_groundtruth_boxes])

    expected_keypoints = (
        np.vstack([keypoint_ys, keypoint_xs]).transpose().reshape((2, 3, 2)))
    self.assertAllEqual(expected_keypoints,
                        tensor_dict[
                            fields.InputDataFields.groundtruth_keypoints])

  def testDecodeDefaultGroundtruthWeights(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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

    self.assertAllClose(tensor_dict[fields.InputDataFields.groundtruth_weights],
                        np.ones(2, dtype=np.float32))

  @test_util.enable_c_shapes
  def testDecodeObjectLabel(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
                        [2])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelNoText(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes = [1, 2]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/object/class/label': self._Int64Feature(bbox_classes),
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

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_classes].get_shape().as_list()),
                        [None])

    init = tf.tables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(bbox_classes,
                        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testDecodeObjectLabelUnrecognizedName(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = ['cat', 'cheetah']
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    self._BytesFeature(encoded_jpeg),
                'image/format':
                    self._BytesFeature('jpeg'),
                'image/object/class/text':
                    self._BytesFeature(bbox_classes_text),
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

  def testDecodeObjectLabelWithMapping(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    bbox_classes_text = ['cat', 'dog']
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded':
                    self._BytesFeature(encoded_jpeg),
                'image/format':
                    self._BytesFeature('jpeg'),
                'image/object/class/text':
                    self._BytesFeature(bbox_classes_text),
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

  @test_util.enable_c_shapes
  def testDecodeObjectArea(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
                         get_shape().as_list()), [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(object_area,
                        tensor_dict[fields.InputDataFields.groundtruth_area])

  @test_util.enable_c_shapes
  def testDecodeObjectIsCrowd(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
                        [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([bool(item) for item in object_is_crowd],
                        tensor_dict[
                            fields.InputDataFields.groundtruth_is_crowd])

  @test_util.enable_c_shapes
  def testDecodeObjectDifficult(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
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
                        [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual([bool(item) for item in object_difficult],
                        tensor_dict[
                            fields.InputDataFields.groundtruth_difficult])

  @test_util.enable_c_shapes
  def testDecodeObjectGroupOf(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_group_of = [0, 1]
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': self._BytesFeature(encoded_jpeg),
            'image/format': self._BytesFeature('jpeg'),
            'image/object/group_of': self._Int64Feature(object_group_of),
        })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_group_of].get_shape().as_list()),
                        [2])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        [bool(item) for item in object_group_of],
        tensor_dict[fields.InputDataFields.groundtruth_group_of])

  def testDecodeObjectWeight(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    object_weights = [0.75, 1.0]
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': self._BytesFeature(encoded_jpeg),
            'image/format': self._BytesFeature('jpeg'),
            'image/object/weight': self._FloatFeature(object_weights),
        })).SerializeToString()

    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((tensor_dict[
        fields.InputDataFields.groundtruth_weights].get_shape().as_list()),
                        [None])
    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        object_weights,
        tensor_dict[fields.InputDataFields.groundtruth_weights])

  @test_util.enable_c_shapes
  def testDecodeInstanceSegmentation(self):
    num_instances = 4
    image_height = 5
    image_width = 3

    # Randomly generate image.
    image_tensor = np.random.randint(256, size=(image_height,
                                                image_width,
                                                3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances,
                                   image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/height': self._Int64Feature([image_height]),
        'image/width': self._Int64Feature([image_width]),
        'image/object/mask': self._FloatFeature(instance_masks_flattened),
        'image/object/class/label': self._Int64Feature(
            object_classes)})).SerializeToString()
    example_decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=True)
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

    self.assertAllEqual((
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks].
        get_shape().as_list()), [4, 5, 3])

    self.assertAllEqual((
        tensor_dict[fields.InputDataFields.groundtruth_classes].
        get_shape().as_list()), [4])

    with self.test_session() as sess:
      tensor_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        instance_masks.astype(np.float32),
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks])
    self.assertAllEqual(
        object_classes,
        tensor_dict[fields.InputDataFields.groundtruth_classes])

  def testInstancesNotAvailableByDefault(self):
    num_instances = 4
    image_height = 5
    image_width = 3
    # Randomly generate image.
    image_tensor = np.random.randint(256, size=(image_height,
                                                image_width,
                                                3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)

    # Randomly generate instance segmentation masks.
    instance_masks = (
        np.random.randint(2, size=(num_instances,
                                   image_height,
                                   image_width)).astype(np.float32))
    instance_masks_flattened = np.reshape(instance_masks, [-1])

    # Randomly generate class labels for each instance.
    object_classes = np.random.randint(
        100, size=(num_instances)).astype(np.int64)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._BytesFeature(encoded_jpeg),
        'image/format': self._BytesFeature('jpeg'),
        'image/height': self._Int64Feature([image_height]),
        'image/width': self._Int64Feature([image_width]),
        'image/object/mask': self._FloatFeature(instance_masks_flattened),
        'image/object/class/label': self._Int64Feature(
            object_classes)})).SerializeToString()
    example_decoder = tf_example_decoder.TfExampleDecoder()
    tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))
    self.assertTrue(fields.InputDataFields.groundtruth_instance_masks
                    not in tensor_dict)

  def testDecodeImageLabels(self):
    image_tensor = np.random.randint(256, size=(4, 5, 3)).astype(np.uint8)
    encoded_jpeg = self._EncodeImage(image_tensor)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': self._BytesFeature(encoded_jpeg),
                'image/format': self._BytesFeature('jpeg'),
                'image/class/label': self._Int64Feature([1, 2]),
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
                'image/encoded': self._BytesFeature(encoded_jpeg),
                'image/format': self._BytesFeature('jpeg'),
                'image/class/text': self._BytesFeature(['dog', 'cat']),
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
