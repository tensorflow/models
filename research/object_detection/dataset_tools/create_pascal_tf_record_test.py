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

"""Test for create_pascal_tf_record.py."""

import os

import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import create_pascal_tf_record


class DictToTFExampleTest(tf.test.TestCase):

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

    data = {
        'folder': '',
        'filename': image_file_name,
        'size': {
            'height': 256,
            'width': 256,
        },
        'object': [
            {
                'difficult': 1,
                'bndbox': {
                    'xmin': 64,
                    'ymin': 64,
                    'xmax': 192,
                    'ymax': 192,
                },
                'name': 'person',
                'truncated': 0,
                'pose': '',
            },
        ],
    }

    label_map_dict = {
        'background': 0,
        'person': 1,
        'notperson': 2,
    }

    example = create_pascal_tf_record.dict_to_tf_example(
        data, self.get_temp_dir(), label_map_dict, image_subdirectory='')
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value, ['jpeg'])
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
        ['person'])
    self._assertProtoEqual(
        example.features.feature['image/object/class/label'].int64_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/difficult'].int64_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/truncated'].int64_list.value,
        [0])
    self._assertProtoEqual(
        example.features.feature['image/object/view'].bytes_list.value, [''])


if __name__ == '__main__':
  tf.test.main()
