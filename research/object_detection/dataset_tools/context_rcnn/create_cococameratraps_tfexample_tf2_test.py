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
"""Tests for create_cococameratraps_tfexample_main."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import json
import os
import tempfile
import unittest

import numpy as np

from PIL import Image
import tensorflow as tf
from object_detection.utils import tf_version

if tf_version.is_tf2():
  from object_detection.dataset_tools.context_rcnn import create_cococameratraps_tfexample_main  # pylint:disable=g-import-not-at-top

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CreateCOCOCameraTrapsTfexampleTest(tf.test.TestCase):

  IMAGE_HEIGHT = 360
  IMAGE_WIDTH = 480

  def _write_random_images_to_directory(self, directory, num_frames):
    for frame_num in range(num_frames):
      img = np.random.randint(0, high=256,
                              size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
                              dtype=np.uint8)
      pil_image = Image.fromarray(img)
      fname = 'im_' + str(frame_num) + '.jpg'
      pil_image.save(os.path.join(directory, fname), 'JPEG')

  def _create_json_file(self, directory, num_frames, keep_bboxes=False):
    json_dict = {'images': [], 'annotations': []}
    json_dict['categories'] = [{'id': 0, 'name': 'empty'},
                               {'id': 1, 'name': 'animal'}]
    for idx in range(num_frames):
      im = {'id': 'im_' + str(idx),
            'file_name': 'im_' + str(idx) + '.jpg',
            'height': self.IMAGE_HEIGHT,
            'width': self.IMAGE_WIDTH,
            'seq_id': 'seq_1',
            'seq_num_frames': num_frames,
            'frame_num': idx,
            'location': 'loc_' + str(idx),
            'date_captured': str(datetime.datetime.now())
           }
      json_dict['images'].append(im)
      ann = {'id': 'ann' + str(idx),
             'image_id': 'im_' + str(idx),
             'category_id': 1,
            }
      if keep_bboxes:
        ann['bbox'] = [0.0 * self.IMAGE_WIDTH,
                       0.1 * self.IMAGE_HEIGHT,
                       0.5 * self.IMAGE_WIDTH,
                       0.5 * self.IMAGE_HEIGHT]
      json_dict['annotations'].append(ann)

    json_path = os.path.join(directory, 'test_file.json')
    with tf.io.gfile.GFile(json_path, 'w') as f:
      json.dump(json_dict, f)
    return json_path

  def assert_expected_example_bbox(self, example):
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0.1])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0.0])
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [0.6])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [0.5])
    self.assertAllClose(
        example.features.feature['image/object/class/label']
        .int64_list.value, [1])
    self.assertAllEqual(
        example.features.feature['image/object/class/text']
        .bytes_list.value, [b'animal'])
    self.assertAllClose(
        example.features.feature['image/class/label']
        .int64_list.value, [1])
    self.assertAllEqual(
        example.features.feature['image/class/text']
        .bytes_list.value, [b'animal'])

    # Check other essential attributes.
    self.assertAllEqual(
        example.features.feature['image/height'].int64_list.value,
        [self.IMAGE_HEIGHT])
    self.assertAllEqual(
        example.features.feature['image/width'].int64_list.value,
        [self.IMAGE_WIDTH])
    self.assertAllEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [b'im_0'])
    self.assertTrue(
        example.features.feature['image/encoded'].bytes_list.value)

  def assert_expected_example(self, example):
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [])
    self.assertAllClose(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [])
    self.assertAllClose(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [])
    self.assertAllClose(
        example.features.feature['image/object/class/label']
        .int64_list.value, [1])
    self.assertAllEqual(
        example.features.feature['image/object/class/text']
        .bytes_list.value, [b'animal'])
    self.assertAllClose(
        example.features.feature['image/class/label']
        .int64_list.value, [1])
    self.assertAllEqual(
        example.features.feature['image/class/text']
        .bytes_list.value, [b'animal'])

    # Check other essential attributes.
    self.assertAllEqual(
        example.features.feature['image/height'].int64_list.value,
        [self.IMAGE_HEIGHT])
    self.assertAllEqual(
        example.features.feature['image/width'].int64_list.value,
        [self.IMAGE_WIDTH])
    self.assertAllEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [b'im_0'])
    self.assertTrue(
        example.features.feature['image/encoded'].bytes_list.value)

  def test_beam_pipeline(self):
    num_frames = 1
    temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
    json_path = self._create_json_file(temp_dir, num_frames)
    output_tfrecord = temp_dir+'/output'
    self._write_random_images_to_directory(temp_dir, num_frames)
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        runner='DirectRunner')
    p = beam.Pipeline(options=pipeline_options)
    create_cococameratraps_tfexample_main.create_pipeline(
        p, temp_dir, json_path,
        output_tfrecord_prefix=output_tfrecord)
    p.run()
    filenames = tf.io.gfile.glob(output_tfrecord + '-?????-of-?????')
    actual_output = []
    record_iterator = tf.data.TFRecordDataset(
        tf.convert_to_tensor(filenames)).as_numpy_iterator()
    for record in record_iterator:
      actual_output.append(record)
    self.assertEqual(len(actual_output), num_frames)
    self.assert_expected_example(tf.train.Example.FromString(
        actual_output[0]))

  def test_beam_pipeline_bbox(self):
    num_frames = 1
    temp_dir = tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))
    json_path = self._create_json_file(temp_dir, num_frames, keep_bboxes=True)
    output_tfrecord = temp_dir+'/output'
    self._write_random_images_to_directory(temp_dir, num_frames)
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        runner='DirectRunner')
    p = beam.Pipeline(options=pipeline_options)
    create_cococameratraps_tfexample_main.create_pipeline(
        p, temp_dir, json_path,
        output_tfrecord_prefix=output_tfrecord,
        keep_bboxes=True)
    p.run()
    filenames = tf.io.gfile.glob(output_tfrecord+'-?????-of-?????')
    actual_output = []
    record_iterator = tf.data.TFRecordDataset(
        tf.convert_to_tensor(filenames)).as_numpy_iterator()
    for record in record_iterator:
      actual_output.append(record)
    self.assertEqual(len(actual_output), num_frames)
    self.assert_expected_example_bbox(tf.train.Example.FromString(
        actual_output[0]))


if __name__ == '__main__':
  tf.test.main()
