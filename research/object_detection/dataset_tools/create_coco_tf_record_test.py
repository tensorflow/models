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
"""Test for create_coco_tf_record.py."""

import io
import json
import os

import numpy as np
import PIL.Image
import six
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import create_coco_tf_record


class CreateCocoTFRecordTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def _assertProtoClose(self, proto_field, expectation):
    """Helper function to assert if a proto field nearly equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertAllClose(proto_list, expectation)

  def test_create_tf_example(self):
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.rand(256, 256, 3)
    tmp_dir = self.get_temp_dir()
    save_path = os.path.join(tmp_dir, image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    image = {
        'file_name': image_file_name,
        'height': 256,
        'width': 256,
        'id': 11,
    }

    annotations_list = [{
        'area': .5,
        'iscrowd': False,
        'image_id': 11,
        'bbox': [64, 64, 128, 128],
        'category_id': 2,
        'id': 1000,
    }]

    image_dir = tmp_dir
    category_index = {
        1: {
            'name': 'dog',
            'id': 1
        },
        2: {
            'name': 'cat',
            'id': 2
        },
        3: {
            'name': 'human',
            'id': 3
        }
    }

    (_, example,
     num_annotations_skipped, _, _) = create_coco_tf_record.create_tf_example(
         image, annotations_list, image_dir, category_index)

    self.assertEqual(num_annotations_skipped, 0)
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [six.b(image_file_name)])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [six.b(str(image['id']))])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value,
        [six.b('jpeg')])
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
        [six.b('cat')])

  def test_create_tf_example_with_instance_masks(self):
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.rand(8, 8, 3)
    tmp_dir = self.get_temp_dir()
    save_path = os.path.join(tmp_dir, image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    image = {
        'file_name': image_file_name,
        'height': 8,
        'width': 8,
        'id': 11,
    }

    annotations_list = [{
        'area': .5,
        'iscrowd': False,
        'image_id': 11,
        'bbox': [0, 0, 8, 8],
        'segmentation': [[4, 0, 0, 0, 0, 4], [8, 4, 4, 8, 8, 8]],
        'category_id': 1,
        'id': 1000,
    }]

    image_dir = tmp_dir
    category_index = {
        1: {
            'name': 'dog',
            'id': 1
        },
    }

    (_, example,
     num_annotations_skipped, _, _) = create_coco_tf_record.create_tf_example(
         image, annotations_list, image_dir, category_index, include_masks=True)

    self.assertEqual(num_annotations_skipped, 0)
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [8])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [8])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [six.b(image_file_name)])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [six.b(str(image['id']))])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value,
        [six.b('jpeg')])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [1])
    self._assertProtoEqual(
        example.features.feature['image/object/class/text'].bytes_list.value,
        [six.b('dog')])
    encoded_mask_pngs = [
        io.BytesIO(encoded_masks) for encoded_masks in example.features.feature[
            'image/object/mask'].bytes_list.value
    ]
    pil_masks = [
        np.array(PIL.Image.open(encoded_mask_png))
        for encoded_mask_png in encoded_mask_pngs
    ]
    self.assertEqual(len(pil_masks), 1)
    self.assertAllEqual(pil_masks[0],
                        [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])

  def test_create_tf_example_with_keypoints(self):
    image_dir = self.get_temp_dir()
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.randint(low=0, high=256, size=(256, 256, 3)).astype(
        np.uint8)
    save_path = os.path.join(image_dir, image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    image = {
        'file_name': image_file_name,
        'height': 256,
        'width': 256,
        'id': 11,
    }

    min_x, min_y = 64, 64
    max_x, max_y = 128, 128
    keypoints = []
    num_visible_keypoints = 0
    xv = []
    yv = []
    vv = []
    for _ in range(17):
      xc = min_x + int(np.random.rand()*(max_x - min_x))
      yc = min_y + int(np.random.rand()*(max_y - min_y))
      vis = np.random.randint(0, 3)
      xv.append(xc)
      yv.append(yc)
      vv.append(vis)
      keypoints.extend([xc, yc, vis])
      num_visible_keypoints += (vis > 0)

    annotations_list = [{
        'area': 0.5,
        'iscrowd': False,
        'image_id': 11,
        'bbox': [64, 64, 128, 128],
        'category_id': 1,
        'id': 1000
    }]

    keypoint_annotations_dict = {
        1000: {
            'keypoints': keypoints,
            'num_keypoints': num_visible_keypoints
        }
    }

    category_index = {
        1: {
            'name': 'person',
            'id': 1
        }
    }

    _, example, _, num_keypoint_annotation_skipped, _ = (
        create_coco_tf_record.create_tf_example(
            image,
            annotations_list,
            image_dir,
            category_index,
            include_masks=False,
            keypoint_annotations_dict=keypoint_annotations_dict))

    self.assertEqual(num_keypoint_annotation_skipped, 0)
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [six.b(image_file_name)])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [six.b(str(image['id']))])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value,
        [six.b('jpeg')])
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
        [six.b('person')])
    self._assertProtoClose(
        example.features.feature['image/object/keypoint/x'].float_list.value,
        np.array(xv, dtype=np.float32) / 256)
    self._assertProtoClose(
        example.features.feature['image/object/keypoint/y'].float_list.value,
        np.array(yv, dtype=np.float32) / 256)
    self._assertProtoEqual(
        example.features.feature['image/object/keypoint/text'].bytes_list.value,
        create_coco_tf_record._COCO_KEYPOINT_NAMES)
    self._assertProtoEqual(
        example.features.feature[
            'image/object/keypoint/visibility'].int64_list.value, vv)

  def test_create_tf_example_with_dense_pose(self):
    image_dir = self.get_temp_dir()
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.randint(low=0, high=256, size=(256, 256, 3)).astype(
        np.uint8)
    save_path = os.path.join(image_dir, image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    image = {
        'file_name': image_file_name,
        'height': 256,
        'width': 256,
        'id': 11,
    }

    min_x, min_y = 64, 64
    max_x, max_y = 128, 128
    keypoints = []
    num_visible_keypoints = 0
    xv = []
    yv = []
    vv = []
    for _ in range(17):
      xc = min_x + int(np.random.rand()*(max_x - min_x))
      yc = min_y + int(np.random.rand()*(max_y - min_y))
      vis = np.random.randint(0, 3)
      xv.append(xc)
      yv.append(yc)
      vv.append(vis)
      keypoints.extend([xc, yc, vis])
      num_visible_keypoints += (vis > 0)

    annotations_list = [{
        'area': 0.5,
        'iscrowd': False,
        'image_id': 11,
        'bbox': [64, 64, 128, 128],
        'category_id': 1,
        'id': 1000
    }]

    num_points = 45
    dp_i = np.random.randint(1, 25, (num_points,)).astype(np.float32)
    dp_u = np.random.randn(num_points)
    dp_v = np.random.randn(num_points)
    dp_x = np.random.rand(num_points)*256.
    dp_y = np.random.rand(num_points)*256.
    densepose_annotations_dict = {
        1000: {
            'dp_I': dp_i,
            'dp_U': dp_u,
            'dp_V': dp_v,
            'dp_x': dp_x,
            'dp_y': dp_y,
            'bbox': [64, 64, 128, 128],
        }
    }

    category_index = {
        1: {
            'name': 'person',
            'id': 1
        }
    }

    _, example, _, _, num_densepose_annotation_skipped = (
        create_coco_tf_record.create_tf_example(
            image,
            annotations_list,
            image_dir,
            category_index,
            include_masks=False,
            densepose_annotations_dict=densepose_annotations_dict))

    self.assertEqual(num_densepose_annotation_skipped, 0)
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [six.b(image_file_name)])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [six.b(str(image['id']))])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value,
        [six.b('jpeg')])
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
        [six.b('person')])
    self._assertProtoEqual(
        example.features.feature['image/object/densepose/num'].int64_list.value,
        [num_points])
    self.assertAllEqual(
        example.features.feature[
            'image/object/densepose/part_index'].int64_list.value,
        dp_i.astype(np.int64) - create_coco_tf_record._DP_PART_ID_OFFSET)
    self.assertAllClose(
        example.features.feature['image/object/densepose/u'].float_list.value,
        dp_u)
    self.assertAllClose(
        example.features.feature['image/object/densepose/v'].float_list.value,
        dp_v)
    expected_dp_x = (64 + dp_x * 128. / 256.) / 256.
    expected_dp_y = (64 + dp_y * 128. / 256.) / 256.
    self.assertAllClose(
        example.features.feature['image/object/densepose/x'].float_list.value,
        expected_dp_x)
    self.assertAllClose(
        example.features.feature['image/object/densepose/y'].float_list.value,
        expected_dp_y)

  def test_create_sharded_tf_record(self):
    tmp_dir = self.get_temp_dir()
    image_paths = ['tmp1_image.jpg', 'tmp2_image.jpg']
    for image_path in image_paths:
      image_data = np.random.rand(256, 256, 3)
      save_path = os.path.join(tmp_dir, image_path)
      image = PIL.Image.fromarray(image_data, 'RGB')
      image.save(save_path)

    images = [{
        'file_name': image_paths[0],
        'height': 256,
        'width': 256,
        'id': 11,
    }, {
        'file_name': image_paths[1],
        'height': 256,
        'width': 256,
        'id': 12,
    }]

    annotations = [{
        'area': .5,
        'iscrowd': False,
        'image_id': 11,
        'bbox': [64, 64, 128, 128],
        'category_id': 2,
        'id': 1000,
    }]

    category_index = [{
        'name': 'dog',
        'id': 1
    }, {
        'name': 'cat',
        'id': 2
    }, {
        'name': 'human',
        'id': 3
    }]
    groundtruth_data = {'images': images, 'annotations': annotations,
                        'categories': category_index}
    annotation_file = os.path.join(tmp_dir, 'annotation.json')
    with open(annotation_file, 'w') as annotation_fid:
      json.dump(groundtruth_data, annotation_fid)

    output_path = os.path.join(tmp_dir, 'out.record')
    create_coco_tf_record._create_tf_record_from_coco_annotations(
        annotation_file,
        tmp_dir,
        output_path,
        False,
        2)
    self.assertTrue(os.path.exists(output_path + '-00000-of-00002'))
    self.assertTrue(os.path.exists(output_path + '-00001-of-00002'))


if __name__ == '__main__':
  tf.test.main()
