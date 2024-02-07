# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for parsers."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.pointpillars.dataloaders import parsers


def _mock_decoded_example(num_pillars, num_points_per_pillar,
                          num_features_per_point, num_boxes):
  frame_id = np.random.randint(0, 10, dtype=np.int64)
  pillars = np.random.rand(num_pillars, num_points_per_pillar,
                           num_features_per_point).astype(np.float32)
  indices = np.random.randint(0, 10, size=[num_pillars, 2], dtype=np.int32)
  classes = np.random.randint(0, 10, size=[num_boxes], dtype=np.int32)
  boxes = np.random.rand(num_boxes, 4).astype(np.float32)
  heading = np.random.rand(num_boxes, 1).astype(np.float32)
  z = np.random.rand(num_boxes, 1).astype(np.float32)
  height = np.random.rand(num_boxes, 1).astype(np.float32)
  difficulty = np.random.randint(0, 10, size=[num_boxes], dtype=np.int32)

  decoded_example = {
      'frame_id': tf.convert_to_tensor(frame_id, dtype=tf.int64),
      'pillars': tf.convert_to_tensor(pillars, dtype=tf.float32),
      'indices': tf.convert_to_tensor(indices, dtype=tf.int32),
      'gt_classes': tf.convert_to_tensor(classes, dtype=tf.int32),
      'gt_boxes': tf.convert_to_tensor(boxes, dtype=tf.float32),
      'gt_attributes': {
          'heading': tf.convert_to_tensor(heading, dtype=tf.float32),
          'z': tf.convert_to_tensor(z, dtype=tf.float32),
          'height': tf.convert_to_tensor(height, dtype=tf.float32),
      },
      'gt_difficulty': tf.convert_to_tensor(difficulty, dtype=tf.int32),
  }
  return decoded_example


class ParserTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('all', 1, 10, True),
      ('vehicle', 10, 2, True),
      ('pedestrian', 1, 10, False),
      ('cyclist', 10, 2, False),
  )
  def test_shape(self, classes, num_boxes, max_num_boxes, is_training):
    min_level = 1
    max_level = 3
    image_size = (32, 32)
    anchor_sizes = [(1.1, 2.2)]
    num_anchors_per_location = len(anchor_sizes)
    match_threshold = 0.5
    unmatched_threshold = 0.5
    parser = parsers.Parser(classes, min_level, max_level, image_size,
                            anchor_sizes, match_threshold, unmatched_threshold,
                            max_num_boxes, 'float32')

    num_pillars = 2
    num_points_per_pillar = 3
    num_features_per_point = 4
    decoded_example = _mock_decoded_example(num_pillars, num_points_per_pillar,
                                            num_features_per_point, num_boxes)
    features, labels = parser.parse_fn(is_training=is_training)(
        decoded_tensors=decoded_example)
    features = tf.nest.map_structure(lambda x: x.numpy(), features)
    labels = tf.nest.map_structure(lambda x: x.numpy(), labels)

    self.assertAllEqual(
        (num_pillars, num_points_per_pillar, num_features_per_point),
        features['pillars'].shape)
    self.assertAllEqual(
        (num_pillars, 2), features['indices'].shape)
    total_num_anchors = 0
    for level in range(min_level, max_level + 1):
      stride = 2**level
      h_i = image_size[0] / stride
      w_i = image_size[1] / stride
      total_num_anchors += h_i * w_i * num_anchors_per_location
      self.assertAllEqual((h_i, w_i, num_anchors_per_location),
                          labels['cls_targets'][str(level)].shape)
      self.assertAllEqual((h_i, w_i, num_anchors_per_location * 4),
                          labels['box_targets'][str(level)].shape)
      self.assertAllEqual(
          (h_i, w_i, num_anchors_per_location),
          labels['attribute_targets']['heading'][str(level)].shape)
      self.assertAllEqual(
          (h_i, w_i, num_anchors_per_location),
          labels['attribute_targets']['height'][str(level)].shape)
      self.assertAllEqual(
          (h_i, w_i, num_anchors_per_location),
          labels['attribute_targets']['z'][str(level)].shape)
      if not is_training:
        self.assertAllEqual((h_i, w_i, num_anchors_per_location * 4),
                            labels['anchor_boxes'][str(level)].shape)

    self.assertAllEqual((total_num_anchors,),
                        labels['cls_weights'].shape)
    self.assertAllEqual((total_num_anchors,),
                        labels['box_weights'].shape)

    if not is_training:
      self.assertAllEqual((2,), labels['image_shape'].shape)
      groundtruths = labels['groundtruths']
      self.assertEmpty(groundtruths['frame_id'].shape)
      self.assertEmpty(groundtruths['num_detections'].shape)
      self.assertAllEqual(
          (max_num_boxes,), groundtruths['classes'].shape)
      self.assertAllEqual(
          (max_num_boxes, 4), groundtruths['boxes'].shape)
      self.assertAllEqual(
          (max_num_boxes, 1), groundtruths['attributes']['heading'].shape)
      self.assertAllEqual(
          (max_num_boxes, 1), groundtruths['attributes']['height'].shape)
      self.assertAllEqual(
          (max_num_boxes, 1), groundtruths['attributes']['z'].shape)
      self.assertAllEqual(
          (max_num_boxes,), groundtruths['difficulty'].shape)


if __name__ == '__main__':
  tf.test.main()
