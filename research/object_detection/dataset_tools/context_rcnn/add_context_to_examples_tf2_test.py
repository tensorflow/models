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
"""Tests for add_context_to_examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import datetime
import os
import tempfile
import unittest

import numpy as np
import six
import tensorflow as tf

from object_detection.utils import tf_version

if tf_version.is_tf2():
  from object_detection.dataset_tools.context_rcnn import add_context_to_examples  # pylint:disable=g-import-not-at-top

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


@contextlib.contextmanager
def InMemoryTFRecord(entries):
  temp = tempfile.NamedTemporaryFile(delete=False)
  filename = temp.name
  try:
    with tf.io.TFRecordWriter(filename) as writer:
      for value in entries:
        writer.write(value)
    yield filename
  finally:
    os.unlink(temp.name)


def BytesFeature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def BytesListFeature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def Int64Feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def Int64ListFeature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def FloatListFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class GenerateContextDataTest(tf.test.TestCase):

  def _create_first_tf_example(self):
    encoded_image = tf.io.encode_jpeg(
        tf.constant(np.ones((4, 4, 3)).astype(np.uint8))).numpy()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': BytesFeature(encoded_image),
        'image/source_id': BytesFeature(six.ensure_binary('image_id_1')),
        'image/height': Int64Feature(4),
        'image/width': Int64Feature(4),
        'image/object/class/label': Int64ListFeature([5, 5]),
        'image/object/class/text': BytesListFeature([six.ensure_binary('hyena'),
                                                     six.ensure_binary('hyena')
                                                    ]),
        'image/object/bbox/xmin': FloatListFeature([0.0, 0.1]),
        'image/object/bbox/xmax': FloatListFeature([0.2, 0.3]),
        'image/object/bbox/ymin': FloatListFeature([0.4, 0.5]),
        'image/object/bbox/ymax': FloatListFeature([0.6, 0.7]),
        'image/seq_id': BytesFeature(six.ensure_binary('01')),
        'image/seq_num_frames': Int64Feature(2),
        'image/seq_frame_num': Int64Feature(0),
        'image/date_captured': BytesFeature(
            six.ensure_binary(str(datetime.datetime(2020, 1, 1, 1, 0, 0)))),
        'image/embedding': FloatListFeature([0.1, 0.2, 0.3]),
        'image/embedding_score': FloatListFeature([0.9]),
        'image/embedding_length': Int64Feature(3)

    }))

    return example.SerializeToString()

  def _create_second_tf_example(self):
    encoded_image = tf.io.encode_jpeg(
        tf.constant(np.ones((4, 4, 3)).astype(np.uint8))).numpy()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': BytesFeature(encoded_image),
        'image/source_id': BytesFeature(six.ensure_binary('image_id_2')),
        'image/height': Int64Feature(4),
        'image/width': Int64Feature(4),
        'image/object/class/label': Int64ListFeature([5]),
        'image/object/class/text': BytesListFeature([six.ensure_binary('hyena')
                                                    ]),
        'image/object/bbox/xmin': FloatListFeature([0.0]),
        'image/object/bbox/xmax': FloatListFeature([0.1]),
        'image/object/bbox/ymin': FloatListFeature([0.2]),
        'image/object/bbox/ymax': FloatListFeature([0.3]),
        'image/seq_id': BytesFeature(six.ensure_binary('01')),
        'image/seq_num_frames': Int64Feature(2),
        'image/seq_frame_num': Int64Feature(1),
        'image/date_captured': BytesFeature(
            six.ensure_binary(str(datetime.datetime(2020, 1, 1, 1, 1, 0)))),
        'image/embedding': FloatListFeature([0.4, 0.5, 0.6]),
        'image/embedding_score': FloatListFeature([0.9]),
        'image/embedding_length': Int64Feature(3)
    }))

    return example.SerializeToString()

  def assert_expected_examples(self, tf_example_list):
    self.assertAllEqual(
        {tf_example.features.feature['image/source_id'].bytes_list.value[0]
         for tf_example in tf_example_list},
        {six.ensure_binary('image_id_1'), six.ensure_binary('image_id_2')})
    self.assertAllClose(
        tf_example_list[0].features.feature[
            'image/context_features'].float_list.value,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    self.assertAllClose(
        tf_example_list[1].features.feature[
            'image/context_features'].float_list.value,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

  def assert_expected_sequence_example(self, tf_sequence_example_list):
    tf_sequence_example = tf_sequence_example_list[0]
    num_frames = 2

    self.assertAllEqual(
        tf_sequence_example.context.feature[
            'clip/media_id'].bytes_list.value[0], six.ensure_binary(
                '01_0'))
    self.assertAllClose(
        tf_sequence_example.context.feature[
            'image/context_features'].float_list.value,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    seq_feature_dict = tf_sequence_example.feature_lists.feature_list

    self.assertLen(
        seq_feature_dict['image/encoded'].feature[:],
        num_frames)
    actual_timestamps = [
        feature.int64_list.value[0] for feature
        in seq_feature_dict['image/timestamp'].feature]
    timestamps = [0, 1]
    self.assertAllEqual(timestamps, actual_timestamps)

    # First image.
    self.assertAllClose(
        [0.4, 0.5],
        seq_feature_dict['region/bbox/ymin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.0, 0.1],
        seq_feature_dict['region/bbox/xmin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.6, 0.7],
        seq_feature_dict['region/bbox/ymax'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.2, 0.3],
        seq_feature_dict['region/bbox/xmax'].feature[0].float_list.value[:])
    self.assertAllEqual(
        [six.ensure_binary('hyena'), six.ensure_binary('hyena')],
        seq_feature_dict['region/label/string'].feature[0].bytes_list.value[:])

    # Second example.
    self.assertAllClose(
        [0.2],
        seq_feature_dict['region/bbox/ymin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.0],
        seq_feature_dict['region/bbox/xmin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.3],
        seq_feature_dict['region/bbox/ymax'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.1],
        seq_feature_dict['region/bbox/xmax'].feature[1].float_list.value[:])
    self.assertAllEqual(
        [six.ensure_binary('hyena')],
        seq_feature_dict['region/label/string'].feature[1].bytes_list.value[:])

  def assert_expected_key(self, key):
    self.assertAllEqual(key, b'01')

  def assert_sorted(self, example_collection):
    example_list = list(example_collection)
    counter = 0
    for example in example_list:
      frame_num = example.features.feature[
          'image/seq_frame_num'].int64_list.value[0]
      self.assertGreaterEqual(frame_num, counter)
      counter = frame_num

  def assert_context(self, example_collection):
    example_list = list(example_collection)
    for example in example_list:
      context = example.features.feature[
          'image/context_features'].float_list.value
      self.assertAllClose([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], context)

  def assert_resized(self, example):
    width = example.features.feature['image/width'].int64_list.value[0]
    self.assertAllEqual(width, 2)
    height = example.features.feature['image/height'].int64_list.value[0]
    self.assertAllEqual(height, 2)

  def assert_size(self, example):
    width = example.features.feature['image/width'].int64_list.value[0]
    self.assertAllEqual(width, 4)
    height = example.features.feature['image/height'].int64_list.value[0]
    self.assertAllEqual(height, 4)

  def test_sliding_window(self):
    example_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    max_clip_length = 3
    stride_length = 3
    out_list = [list(i) for i in add_context_to_examples.get_sliding_window(
        example_list, max_clip_length, stride_length)]
    self.assertAllEqual(out_list, [['a', 'b', 'c'],
                                   ['d', 'e', 'f'],
                                   ['g']])

  def test_rekey_data_fn(self):
    sequence_key = 'image/seq_id'
    time_horizon = None
    reduce_image_size = False
    max_dim = None

    rekey_fn = add_context_to_examples.ReKeyDataFn(
        sequence_key, time_horizon,
        reduce_image_size, max_dim)
    output = rekey_fn.process(self._create_first_tf_example())

    self.assert_expected_key(output[0][0])
    self.assert_size(output[0][1])

  def test_rekey_data_fn_w_resize(self):
    sequence_key = 'image/seq_id'
    time_horizon = None
    reduce_image_size = True
    max_dim = 2

    rekey_fn = add_context_to_examples.ReKeyDataFn(
        sequence_key, time_horizon,
        reduce_image_size, max_dim)
    output = rekey_fn.process(self._create_first_tf_example())

    self.assert_expected_key(output[0][0])
    self.assert_resized(output[0][1])

  def test_sort_fn(self):
    sequence_key = 'image/seq_id'
    sorted_image_ids = False
    max_num_elements_in_context_features = 10
    sort_fn = add_context_to_examples.SortGroupedDataFn(
        sequence_key, sorted_image_ids, max_num_elements_in_context_features)
    output = sort_fn.process(
        ('dummy_key', [tf.train.Example.FromString(
            self._create_second_tf_example()),
                       tf.train.Example.FromString(
                           self._create_first_tf_example())]))

    self.assert_sorted(output[0][1])

  def test_add_context_fn(self):
    sequence_key = 'image/seq_id'
    add_context_features = True
    image_ids_to_keep = 'All'
    context_fn = add_context_to_examples.GenerateContextFn(
        sequence_key, add_context_features, image_ids_to_keep)
    output = context_fn.process(
        ('dummy_key', [tf.train.Example.FromString(
            self._create_first_tf_example()),
                       tf.train.Example.FromString(
                           self._create_second_tf_example())]))

    self.assertEqual(len(output), 2)
    self.assert_context(output)

  def test_add_context_fn_output_sequence_example(self):
    sequence_key = 'image/seq_id'
    add_context_features = True
    image_ids_to_keep = 'All'
    context_fn = add_context_to_examples.GenerateContextFn(
        sequence_key, add_context_features, image_ids_to_keep,
        output_type='tf_sequence_example')
    output = context_fn.process(
        ('01',
         [tf.train.Example.FromString(self._create_first_tf_example()),
          tf.train.Example.FromString(self._create_second_tf_example())]))

    self.assertEqual(len(output), 1)
    self.assert_expected_sequence_example(output)

  def test_add_context_fn_output_sequence_example_cliplen(self):
    sequence_key = 'image/seq_id'
    add_context_features = True
    image_ids_to_keep = 'All'
    context_fn = add_context_to_examples.GenerateContextFn(
        sequence_key, add_context_features, image_ids_to_keep,
        output_type='tf_sequence_example', max_clip_length=1)
    output = context_fn.process(
        ('01',
         [tf.train.Example.FromString(self._create_first_tf_example()),
          tf.train.Example.FromString(self._create_second_tf_example())]))
    self.assertEqual(len(output), 2)

  def test_beam_pipeline(self):
    with InMemoryTFRecord(
        [self._create_first_tf_example(),
         self._create_second_tf_example()]) as input_tfrecord:
      temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
      output_tfrecord = os.path.join(temp_dir, 'output_tfrecord')
      sequence_key = six.ensure_binary('image/seq_id')
      max_num_elements = 10
      num_shards = 1
      pipeline_options = beam.options.pipeline_options.PipelineOptions(
          runner='DirectRunner')
      p = beam.Pipeline(options=pipeline_options)
      add_context_to_examples.construct_pipeline(
          p,
          input_tfrecord,
          output_tfrecord,
          sequence_key,
          max_num_elements_in_context_features=max_num_elements,
          num_shards=num_shards)
      p.run()
      filenames = tf.io.gfile.glob(output_tfrecord + '-?????-of-?????')
      actual_output = []
      record_iterator = tf.data.TFRecordDataset(
          tf.convert_to_tensor(filenames)).as_numpy_iterator()
      for record in record_iterator:
        actual_output.append(record)
      self.assertEqual(len(actual_output), 2)
      self.assert_expected_examples([tf.train.Example.FromString(
          tf_example) for tf_example in actual_output])

  def test_beam_pipeline_sequence_example(self):
    with InMemoryTFRecord(
        [self._create_first_tf_example(),
         self._create_second_tf_example()]) as input_tfrecord:
      temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
      output_tfrecord = os.path.join(temp_dir, 'output_tfrecord')
      sequence_key = six.ensure_binary('image/seq_id')
      max_num_elements = 10
      num_shards = 1
      pipeline_options = beam.options.pipeline_options.PipelineOptions(
          runner='DirectRunner')
      p = beam.Pipeline(options=pipeline_options)
      add_context_to_examples.construct_pipeline(
          p,
          input_tfrecord,
          output_tfrecord,
          sequence_key,
          max_num_elements_in_context_features=max_num_elements,
          num_shards=num_shards,
          output_type='tf_sequence_example')
      p.run()
      filenames = tf.io.gfile.glob(output_tfrecord + '-?????-of-?????')
      actual_output = []
      record_iterator = tf.data.TFRecordDataset(
          tf.convert_to_tensor(filenames)).as_numpy_iterator()
      for record in record_iterator:
        actual_output.append(record)
      self.assertEqual(len(actual_output), 1)
      self.assert_expected_sequence_example(
          [tf.train.SequenceExample.FromString(
              tf_example) for tf_example in actual_output])

if __name__ == '__main__':
  tf.test.main()
