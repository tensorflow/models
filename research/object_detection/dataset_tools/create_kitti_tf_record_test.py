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

"""Test for create_kitti_tf_record.py."""

import os

import numpy as np
import PIL.Image
import six
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import create_kitti_tf_record


class CreateKittiTFRecordTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def test_dict_to_tf_example(self):
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.rand(256, 256, 3)
    save_path = os.path.join(self.get_temp_dir(), image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    annotations = {}
    annotations['2d_bbox_left'] = np.array([64])
    annotations['2d_bbox_top'] = np.array([64])
    annotations['2d_bbox_right'] = np.array([192])
    annotations['2d_bbox_bottom'] = np.array([192])
    annotations['type'] = ['car']
    annotations['truncated'] = np.array([1])
    annotations['alpha'] = np.array([2])
    annotations['3d_bbox_height'] = np.array([10])
    annotations['3d_bbox_width'] = np.array([11])
    annotations['3d_bbox_length'] = np.array([12])
    annotations['3d_bbox_x'] = np.array([13])
    annotations['3d_bbox_y'] = np.array([14])
    annotations['3d_bbox_z'] = np.array([15])
    annotations['3d_bbox_rot_y'] = np.array([4])

    label_map_dict = {
        'background': 0,
        'car': 1,
    }

    example = create_kitti_tf_record.prepare_example(
        save_path,
        annotations,
        label_map_dict)

    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [six.b(save_path)])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [six.b(save_path)])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value,
        [six.b('png')])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0.25])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0.25])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [0.75])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [0.75])
    self._assertProtoEqual(
        example.features.feature['image/object/class/text'].bytes_list.value,
        [six.b('car')])
    self._assertProtoEqual(
        example.features.feature['image/object/class/label'].int64_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/truncated'].float_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/alpha'].float_list.value,
        [2])
    self._assertProtoEqual(example.features.feature[
        'image/object/3d_bbox/height'].float_list.value, [10])
    self._assertProtoEqual(
        example.features.feature['image/object/3d_bbox/width'].float_list.value,
        [11])
    self._assertProtoEqual(example.features.feature[
        'image/object/3d_bbox/length'].float_list.value, [12])
    self._assertProtoEqual(
        example.features.feature['image/object/3d_bbox/x'].float_list.value,
        [13])
    self._assertProtoEqual(
        example.features.feature['image/object/3d_bbox/y'].float_list.value,
        [14])
    self._assertProtoEqual(
        example.features.feature['image/object/3d_bbox/z'].float_list.value,
        [15])
    self._assertProtoEqual(
        example.features.feature['image/object/3d_bbox/rot_y'].float_list.value,
        [4])


if __name__ == '__main__':
  tf.test.main()
