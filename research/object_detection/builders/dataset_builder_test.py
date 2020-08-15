# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import dataset_builder
from object_detection.core import standard_fields as fields
from object_detection.dataset_tools import seq_example_util
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util
from object_detection.utils import test_case

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import lookup as contrib_lookup
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


def get_iterator_next_for_testing(dataset, is_tf2):
  iterator = dataset.make_initializable_iterator()
  if not is_tf2:
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator.get_next()


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  parent_path = os.path.dirname(tf.resource_loader.get_data_files_path())
  return os.path.join(parent_path, 'data',
                      'pet_label_map.pbtxt')


class DatasetBuilderTest(test_case.TestCase):

  def create_tf_record(self, has_additional_channels=False, num_shards=1,
                       num_examples_per_shard=1):

    def dummy_jpeg_fn():
      image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
      additional_channels_tensor = np.random.randint(
          255, size=(4, 5, 1)).astype(np.uint8)
      encoded_jpeg = tf.image.encode_jpeg(image_tensor)
      encoded_additional_channels_jpeg = tf.image.encode_jpeg(
          additional_channels_tensor)

      return encoded_jpeg, encoded_additional_channels_jpeg

    encoded_jpeg, encoded_additional_channels_jpeg = self.execute(
        dummy_jpeg_fn, [])

    tmp_dir = self.get_temp_dir()
    flat_mask = (4 * 5) * [1.0]

    for i in range(num_shards):
      path = os.path.join(tmp_dir, '%05d.tfrecord' % i)
      writer = tf.python_io.TFRecordWriter(path)

      for j in range(num_examples_per_shard):
        if num_shards > 1:
          source_id = (str(i) + '_' + str(j)).encode()
        else:
          source_id = str(j).encode()

        features = {
            'image/source_id': dataset_util.bytes_feature(source_id),
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
        writer.write(example.SerializeToString())

      writer.close()

    return os.path.join(self.get_temp_dir(), '?????.tfrecord')

  def _make_random_serialized_jpeg_images(self, num_frames, image_height,
                                          image_width):
    def graph_fn():
      images = tf.cast(tf.random.uniform(
          [num_frames, image_height, image_width, 3],
          maxval=256,
          dtype=tf.int32), dtype=tf.uint8)
      images_list = tf.unstack(images, axis=0)
      encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
      return encoded_images_list

    encoded_images = self.execute(graph_fn, [])
    return encoded_images

  def create_tf_record_sequence_example(self):
    path = os.path.join(self.get_temp_dir(), 'seq_tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    num_frames = 4
    image_height = 4
    image_width = 5
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
      writer.write(sequence_example_serialized)
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

    def graph_fn():
      return get_iterator_next_for_testing(
          dataset_builder.build(input_reader_proto, batch_size=1),
          self.is_tf2())

    output_dict = self.execute(graph_fn, [])

    self.assertNotIn(
        fields.InputDataFields.groundtruth_instance_masks, output_dict)
    self.assertEqual((1, 4, 5, 3),
                     output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual([[2]],
                        output_dict[fields.InputDataFields.groundtruth_classes])
    self.assertEqual(
        (1, 1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual(
        [0.0, 0.0, 1.0, 1.0],
        output_dict[fields.InputDataFields.groundtruth_boxes][0][0])

  def get_mock_reduce_to_frame_fn(self):
    def mock_reduce_to_frame_fn(dataset, dataset_map_fn, batch_size, config):
      def get_frame(tensor_dict):
        out_tensor_dict = {}
        out_tensor_dict[fields.InputDataFields.source_id] = (
            tensor_dict[fields.InputDataFields.source_id][0])
        return out_tensor_dict
      return dataset_map_fn(dataset, get_frame, batch_size, config)
    return mock_reduce_to_frame_fn

  def test_build_tf_record_input_reader_sequence_example_train(self):
    tf_record_path = self.create_tf_record_sequence_example()
    label_map_path = _get_labelmap_path()
    input_type = 'TF_SEQUENCE_EXAMPLE'
    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      input_type: {1}
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path, input_type)
    input_reader_proto = input_reader_pb2.InputReader()
    input_reader_proto.label_map_path = label_map_path
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    reduce_to_frame_fn = self.get_mock_reduce_to_frame_fn()

    def graph_fn():
      return get_iterator_next_for_testing(
          dataset_builder.build(input_reader_proto, batch_size=1,
                                reduce_to_frame_fn=reduce_to_frame_fn),
          self.is_tf2())

    output_dict = self.execute(graph_fn, [])

    self.assertEqual((1,),
                     output_dict[fields.InputDataFields.source_id].shape)

  def test_build_tf_record_input_reader_sequence_example_test(self):
    tf_record_path = self.create_tf_record_sequence_example()
    input_type = 'TF_SEQUENCE_EXAMPLE'
    label_map_path = _get_labelmap_path()
    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      input_type: {1}
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path, input_type)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    input_reader_proto.label_map_path = label_map_path
    reduce_to_frame_fn = self.get_mock_reduce_to_frame_fn()
    def graph_fn():
      return get_iterator_next_for_testing(
          dataset_builder.build(input_reader_proto, batch_size=1,
                                reduce_to_frame_fn=reduce_to_frame_fn),
          self.is_tf2())

    output_dict = self.execute(graph_fn, [])

    self.assertEqual((1,),
                     output_dict[fields.InputDataFields.source_id].shape)

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

    def graph_fn():
      return get_iterator_next_for_testing(
          dataset_builder.build(input_reader_proto, batch_size=1),
          self.is_tf2()
      )

    output_dict = self.execute(graph_fn, [])
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

    def graph_fn():
      return dataset_builder.make_initializable_iterator(
          dataset_builder.build(
              input_reader_proto,
              transform_input_data_fn=one_hot_class_encoding_fn,
              batch_size=2)).get_next()

    output_dict = self.execute(graph_fn, [])

    self.assertAllEqual([2, 4, 5, 3],
                        output_dict[fields.InputDataFields.image].shape)
    self.assertAllEqual(
        [2, 1, 3],
        output_dict[fields.InputDataFields.groundtruth_classes].shape)
    self.assertAllEqual(
        [2, 1, 4], output_dict[fields.InputDataFields.groundtruth_boxes].shape)
    self.assertAllEqual([[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]],
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

    def graph_fn():
      return dataset_builder.make_initializable_iterator(
          dataset_builder.build(
              input_reader_proto,
              transform_input_data_fn=one_hot_class_encoding_fn,
              batch_size=2)).get_next()

    output_dict = self.execute(graph_fn, [])

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

  def test_sample_all_data(self):
    tf_record_path = self.create_tf_record(num_examples_per_shard=2)

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      sample_1_of_n_examples: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)

    def graph_fn():
      dataset = dataset_builder.build(input_reader_proto, batch_size=1)
      sample1_ds = dataset.take(1)
      sample2_ds = dataset.skip(1)
      iter1 = dataset_builder.make_initializable_iterator(sample1_ds)
      iter2 = dataset_builder.make_initializable_iterator(sample2_ds)

      return iter1.get_next(), iter2.get_next()

    output_dict1, output_dict2 = self.execute(graph_fn, [])
    self.assertAllEqual([b'0'], output_dict1[fields.InputDataFields.source_id])
    self.assertEqual([b'1'], output_dict2[fields.InputDataFields.source_id])

  def test_sample_one_of_n_shards(self):
    tf_record_path = self.create_tf_record(num_examples_per_shard=4)

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      sample_1_of_n_examples: 2
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)

    def graph_fn():
      dataset = dataset_builder.build(input_reader_proto, batch_size=1)
      sample1_ds = dataset.take(1)
      sample2_ds = dataset.skip(1)
      iter1 = dataset_builder.make_initializable_iterator(sample1_ds)
      iter2 = dataset_builder.make_initializable_iterator(sample2_ds)

      return iter1.get_next(), iter2.get_next()

    output_dict1, output_dict2 = self.execute(graph_fn, [])
    self.assertAllEqual([b'0'], output_dict1[fields.InputDataFields.source_id])
    self.assertEqual([b'2'], output_dict2[fields.InputDataFields.source_id])

  def test_no_input_context(self):
    """Test that all samples are read with no input context given."""
    tf_record_path = self.create_tf_record(num_examples_per_shard=16,
                                           num_shards=2)

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      num_epochs: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)

    for i in range(4):

      # pylint:disable=cell-var-from-loop
      def graph_fn():
        dataset = dataset_builder.build(input_reader_proto, batch_size=8)
        dataset = dataset.skip(i)
        return get_iterator_next_for_testing(dataset, self.is_tf2())

      batch = self.execute(graph_fn, [])
      self.assertEqual(batch['image'].shape, (8, 4, 5, 3))

    def graph_fn_last_batch():
      dataset = dataset_builder.build(input_reader_proto, batch_size=8)
      dataset = dataset.skip(4)
      return get_iterator_next_for_testing(dataset, self.is_tf2())

    self.assertRaises(tf.errors.OutOfRangeError, self.execute,
                      compute_fn=graph_fn_last_batch, inputs=[])

  def test_with_input_context(self):
    """Test that a subset is read with input context given."""
    tf_record_path = self.create_tf_record(num_examples_per_shard=16,
                                           num_shards=2)

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      num_epochs: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)

    input_context = tf.distribute.InputContext(
        num_input_pipelines=2, input_pipeline_id=0, num_replicas_in_sync=4
    )

    for i in range(8):

      # pylint:disable=cell-var-from-loop
      def graph_fn():

        dataset = dataset_builder.build(input_reader_proto, batch_size=8,
                                        input_context=input_context)
        dataset = dataset.skip(i)
        return get_iterator_next_for_testing(dataset, self.is_tf2())

      batch = self.execute(graph_fn, [])
      self.assertEqual(batch['image'].shape, (2, 4, 5, 3))

    def graph_fn_last_batch():
      dataset = dataset_builder.build(input_reader_proto, batch_size=8,
                                      input_context=input_context)
      dataset = dataset.skip(8)
      return get_iterator_next_for_testing(dataset, self.is_tf2())

    self.assertRaises(tf.errors.OutOfRangeError, self.execute,
                      compute_fn=graph_fn_last_batch, inputs=[])


class ReadDatasetTest(test_case.TestCase):

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

    super(ReadDatasetTest, self).setUp()

  def _get_dataset_next(self, files, config, batch_size, num_batches_skip=0):

    def decode_func(value):
      return [tf.string_to_number(value, out_type=tf.int32)]

    dataset = dataset_builder.read_dataset(tf.data.TextLineDataset, files,
                                           config)
    dataset = dataset.map(decode_func)
    dataset = dataset.batch(batch_size)

    if num_batches_skip > 0:
      dataset = dataset.skip(num_batches_skip)

    return get_iterator_next_for_testing(dataset, self.is_tf2())

  def test_make_initializable_iterator_with_hashTable(self):

    def graph_fn():
      keys = [1, 0, -1]
      dataset = tf.data.Dataset.from_tensor_slices([[1, 2, -1, 5]])
      try:
        # Dynamically try to load the tf v2 lookup, falling back to contrib
        lookup = tf.compat.v2.lookup
        hash_table_class = tf.compat.v2.lookup.StaticHashTable
      except AttributeError:
        lookup = contrib_lookup
        hash_table_class = contrib_lookup.HashTable
      table = hash_table_class(
          initializer=lookup.KeyValueTensorInitializer(
              keys=keys, values=list(reversed(keys))),
          default_value=100)
      dataset = dataset.map(table.lookup)
      return dataset_builder.make_initializable_iterator(dataset).get_next()

    result = self.execute(graph_fn, [])
    self.assertAllEqual(result, [-1, 100, 1, 100])

  def test_read_dataset(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    def graph_fn():
      return self._get_dataset_next(
          [self._path_template % '*'], config, batch_size=20)

    data = self.execute(graph_fn, [])
    # Note that the execute function extracts single outputs if the return
    # value is of size 1.
    self.assertCountEqual(
        data, [
            1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3, 30, 4, 40, 5,
            50
        ])

  def test_reduce_num_reader(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 10
    config.shuffle = False

    def graph_fn():
      return self._get_dataset_next(
          [self._path_template % '*'], config, batch_size=20)

    data = self.execute(graph_fn, [])
    # Note that the execute function extracts single outputs if the return
    # value is of size 1.
    self.assertCountEqual(
        data, [
            1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3, 30, 4, 40, 5,
            50
        ])

  def test_enable_shuffle(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = True

    tf.set_random_seed(1)  # Set graph level seed.

    def graph_fn():
      return self._get_dataset_next(
          [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    data = self.execute(graph_fn, [])

    self.assertTrue(
        np.any(np.not_equal(data, expected_non_shuffle_output)))

  def test_disable_shuffle_(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    def graph_fn():
      return self._get_dataset_next(
          [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    expected_non_shuffle_output2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # Note that the execute function extracts single outputs if the return
    # value is of size 1.
    data = self.execute(graph_fn, [])
    self.assertTrue(all(data == expected_non_shuffle_output1) or
                    all(data == expected_non_shuffle_output2))

  def test_read_dataset_single_epoch(self):
    config = input_reader_pb2.InputReader()
    config.num_epochs = 1
    config.num_readers = 1
    config.shuffle = False

    def graph_fn():
      return self._get_dataset_next(
          [self._path_template % '0'], config, batch_size=30)

    data = self.execute(graph_fn, [])

    # Note that the execute function extracts single outputs if the return
    # value is of size 1.
    self.assertAllEqual(data, [1, 10])

    # First batch will retrieve as much as it can, second batch will fail.
    def graph_fn_second_batch():
      return self._get_dataset_next(
          [self._path_template % '0'], config, batch_size=30,
          num_batches_skip=1)

    self.assertRaises(tf.errors.OutOfRangeError, self.execute,
                      compute_fn=graph_fn_second_batch, inputs=[])


if __name__ == '__main__':
  tf.test.main()
