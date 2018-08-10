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
"""Tests for dataset_builder."""

import os
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from object_detection.builders import dataset_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2


class DatasetBuilderTest(tf.test.TestCase):

  def create_tf_record(self, has_additional_channels=False):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    additional_channels_tensor = np.random.randint(
        255, size=(4, 5, 1)).astype(np.uint8)
    flat_mask = (4 * 5) * [1.0]
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
      encoded_additional_channels_jpeg = tf.image.encode_jpeg(
          tf.constant(additional_channels_tensor)).eval()
    features = {
        'image/encoded':
            feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(value=[encoded_jpeg])),
        'image/format':
            feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(value=['jpeg'.encode('utf-8')])
            ),
        'image/height':
            feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[4])),
        'image/width':
            feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[5])),
        'image/object/bbox/xmin':
            feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[0.0])),
        'image/object/bbox/xmax':
            feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[1.0])),
        'image/object/bbox/ymin':
            feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[0.0])),
        'image/object/bbox/ymax':
            feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[1.0])),
        'image/object/class/label':
            feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[2])),
        'image/object/mask':
            feature_pb2.Feature(
                float_list=feature_pb2.FloatList(value=flat_mask)),
    }
    if has_additional_channels:
      features['image/additional_channels/encoded'] = feature_pb2.Feature(
          bytes_list=feature_pb2.BytesList(
              value=[encoded_additional_channels_jpeg] * 2))
    example = example_pb2.Example(
        features=feature_pb2.Features(feature=features))
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
    tensor_dict = dataset_builder.make_initializable_iterator(
        dataset_builder.build(input_reader_proto, batch_size=1)).get_next()

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)

    self.assertTrue(
        fields.InputDataFields.groundtruth_instance_masks not in output_dict)
    self.assertEquals((1, 4, 5, 3),
                      output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual([[2]],
                        output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEquals(
        (1, 1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0][0])

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
    tensor_dict = dataset_builder.make_initializable_iterator(
        dataset_builder.build(input_reader_proto, batch_size=1)).get_next()

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)
    self.assertAllEqual(
        (1, 1, 4, 5),
        output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)

  def test_build_tf_record_input_reader_with_batch_size_two(self):
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

    def one_hot_class_encoding_fn(tensor_dict):
      tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(
          tensor_dict[fields.InputDataFields.groundtruth_classes] - 1, depth=3)
      return tensor_dict

    tensor_dict = dataset_builder.make_initializable_iterator(
        dataset_builder.build(
            input_reader_proto,
            transform_input_data_fn=one_hot_class_encoding_fn,
            batch_size=2)).get_next()

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)

    self.assertAllEqual([2, 4, 5, 3],
                        output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual([2, 1, 3],
                        output_dict[fields.InputDataFields.groundtruth_classes].
                        shape)
    self.assertAllEqual([2, 1, 4],
                        output_dict[fields.InputDataFields.groundtruth_boxes].
                        shape)
    self.assertAllEqual(
        [[[0.0, 0.0, 1.0, 1.0]],
         [[0.0, 0.0, 1.0, 1.0]]],
        output_dict[fields.InputDataFields.groundtruth_boxes])

  def test_build_tf_record_input_reader_with_batch_size_two_and_masks(self):
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

    def one_hot_class_encoding_fn(tensor_dict):
      tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(
          tensor_dict[fields.InputDataFields.groundtruth_classes] - 1, depth=3)
      return tensor_dict

    tensor_dict = dataset_builder.make_initializable_iterator(
        dataset_builder.build(
            input_reader_proto,
            transform_input_data_fn=one_hot_class_encoding_fn,
            batch_size=2)).get_next()

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)

    self.assertAllEqual(
        [2, 1, 4, 5],
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
      dataset_builder.build(input_reader_proto, batch_size=1)


class ReadDatasetTest(tf.test.TestCase):

  def setUp(self):
    self._path_template = os.path.join(self.get_temp_dir(), 'examples_%s.txt')

    for i in range(5):
      path = self._path_template % i
      with tf.gfile.Open(path, 'wb') as f:
        f.write('\n'.join([str(i + 1), str((i + 1) * 10)]))

    self._shuffle_path_template = os.path.join(self.get_temp_dir(),
                                               'shuffle_%s.txt')
    for i in range(2):
      path = self._shuffle_path_template % i
      with tf.gfile.Open(path, 'wb') as f:
        f.write('\n'.join([str(i)] * 5))

  def _get_dataset_next(self, files, config, batch_size):
    def decode_func(value):
      return [tf.string_to_number(value, out_type=tf.int32)]

    dataset = dataset_builder.read_dataset(
        tf.data.TextLineDataset, files, config)
    dataset = dataset.map(decode_func)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

  def test_make_initializable_iterator_with_hashTable(self):
    keys = [1, 0, -1]
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2, -1, 5]])
    table = tf.contrib.lookup.HashTable(
        initializer=tf.contrib.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=list(reversed(keys))),
        default_value=100)
    dataset = dataset.map(table.lookup)
    data = dataset_builder.make_initializable_iterator(dataset).get_next()
    init = tf.tables_initializer()

    with self.test_session() as sess:
      sess.run(init)
      self.assertAllEqual(sess.run(data), [-1, 100, 1, 100])

  def test_read_dataset(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '*'], config,
                                  batch_size=20)
    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data),
                          [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3,
                            30, 4, 40, 5, 50]])

  def test_reduce_num_reader(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 10
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '*'], config,
                                  batch_size=20)
    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data),
                          [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3,
                            30, 4, 40, 5, 50]])

  def test_enable_shuffle(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = True

    tf.set_random_seed(1)  # Set graph level seed.
    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    with self.test_session() as sess:
      self.assertTrue(
          np.any(np.not_equal(sess.run(data), expected_non_shuffle_output)))

  def test_disable_shuffle_(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data), [expected_non_shuffle_output])

  def test_read_dataset_single_epoch(self):
    config = input_reader_pb2.InputReader()
    config.num_epochs = 1
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '0'], config,
                                  batch_size=30)
    with self.test_session() as sess:
      # First batch will retrieve as much as it can, second batch will fail.
      self.assertAllEqual(sess.run(data), [[1, 10]])
      self.assertRaises(tf.errors.OutOfRangeError, sess.run, data)


if __name__ == '__main__':
  tf.test.main()
