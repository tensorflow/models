# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tfrecord_lib."""

import os

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from official.vision.data import create_coco_tf_record as create_coco_tf_record_lib
from official.vision.data import tfrecord_lib


FLAGS = flags.FLAGS


def process_sample(x):
  d = {'x': x}
  return tf.train.Example(features=tf.train.Features(feature=d)), 0


def parse_function(example_proto):

  feature_description = {
      'x': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
  }
  return tf.io.parse_single_example(example_proto, feature_description)


class TfrecordLibTest(parameterized.TestCase):

  def test_write_tf_record_dataset(self):
    data = [(tfrecord_lib.convert_to_feature(i),) for i in range(17)]

    path = os.path.join(FLAGS.test_tmpdir, 'train')

    tfrecord_lib.write_tf_record_dataset(
        path, data, process_sample, 3, multiple_processes=0)
    tfrecord_files = tf.io.gfile.glob(path + '*')

    self.assertLen(tfrecord_files, 3)

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_function)

    read_values = set(d['x'] for d in dataset.as_numpy_iterator())
    self.assertSetEqual(read_values, set(range(17)))

  def test_convert_to_feature_float(self):

    proto = tfrecord_lib.convert_to_feature(0.0)
    self.assertEqual(proto.float_list.value[0], 0.0)

  def test_convert_to_feature_int(self):

    proto = tfrecord_lib.convert_to_feature(0)
    self.assertEqual(proto.int64_list.value[0], 0)

  def test_convert_to_feature_bytes(self):

    proto = tfrecord_lib.convert_to_feature(b'123')
    self.assertEqual(proto.bytes_list.value[0], b'123')

  def test_convert_to_feature_float_list(self):

    proto = tfrecord_lib.convert_to_feature([0.0, 1.0])
    self.assertSequenceAlmostEqual(proto.float_list.value, [0.0, 1.0])

  def test_convert_to_feature_int_list(self):

    proto = tfrecord_lib.convert_to_feature([0, 1])
    self.assertSequenceAlmostEqual(proto.int64_list.value, [0, 1])

  def test_convert_to_feature_bytes_list(self):

    proto = tfrecord_lib.convert_to_feature([b'123', b'456'])
    self.assertSequenceAlmostEqual(proto.bytes_list.value, [b'123', b'456'])

  def test_obj_annotation_tf_example(self):
    images = [
        {
            'id': 0,
            'file_name': 'example1.jpg',
            'height': 512,
            'width': 512,
        },
        {
            'id': 1,
            'file_name': 'example2.jpg',
            'height': 512,
            'width': 512,
        },
    ]
    img_to_obj_annotation = {
        0: [{
            'id': 0,
            'image_id': 0,
            'category_id': 1,
            'bbox': [3, 1, 511, 510],
            'area': 260610.00,
            'segmentation': [],
            'iscrowd': 0,
        }],
        1: [{
            'id': 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': [1, 1, 100, 150],
            'area': 15000.00,
            'segmentation': [],
            'iscrowd': 0,
        }],
    }
    id_to_name_map = {
        0: 'Super-Class',
        1: 'Class-1',
    }

    temp_dir = FLAGS.test_tmpdir
    image_dir = os.path.join(temp_dir, 'data')
    if not os.path.exists(image_dir):
      os.mkdir(image_dir)
    for image in images:
      image_path = os.path.join(image_dir, image['file_name'])
      tf.keras.utils.save_img(
          image_path,
          tf.ones(shape=(image['height'], image['width'], 3)).numpy(),
      )

    output_path = os.path.join(image_dir, 'train')
    coco_annotations_iter = create_coco_tf_record_lib.generate_annotations(
        images=images,
        image_dirs=[image_dir],
        panoptic_masks_dir=None,
        img_to_obj_annotation=img_to_obj_annotation,
        img_to_caption_annotation=None,
        img_to_panoptic_annotation=None,
        is_category_thing=None,
        id_to_name_map=id_to_name_map,
        include_panoptic_masks=False,
        include_masks=False,
    )

    tfrecord_lib.write_tf_record_dataset(
        output_path,
        coco_annotations_iter,
        create_coco_tf_record_lib.create_tf_example,
        1,
        multiple_processes=0,
    )
    tfrecord_files = tf.io.gfile.glob(output_path + '*')

    self.assertLen(tfrecord_files, 1)

    ds = tf.data.TFRecordDataset(tfrecord_files)
    assertion_count = 0
    for _ in ds:
      assertion_count += 1

    self.assertEqual(assertion_count, 2)


if __name__ == '__main__':
  tf.test.main()
