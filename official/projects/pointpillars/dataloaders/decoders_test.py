# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for decoders."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.dataloaders import decoders
from official.vision.data.tfrecord_lib import convert_to_feature


def _mock_serialized_example(num_pillars, num_points_per_pillar,
                             num_features_per_point, num_boxes):
  frame_id = np.random.randint(0, 10, dtype=np.int64)
  pillars = np.random.rand(num_pillars, num_points_per_pillar,
                           num_features_per_point).astype(np.float32)
  indices = np.random.randint(0, 10, size=[num_pillars, 2], dtype=np.int32)
  classes = np.random.randint(0, 10, size=[num_boxes], dtype=np.int32)
  ymin = np.random.rand(num_boxes).astype(np.float32)
  xmin = np.random.rand(num_boxes).astype(np.float32)
  ymax = np.random.rand(num_boxes).astype(np.float32)
  xmax = np.random.rand(num_boxes).astype(np.float32)
  heading = np.random.rand(num_boxes).astype(np.float32)
  z = np.random.rand(num_boxes).astype(np.float32)
  height = np.random.rand(num_boxes).astype(np.float32)
  difficulty = np.random.randint(0, 10, size=[num_boxes], dtype=np.int32)

  feature = {
      'frame_id': convert_to_feature(frame_id, 'int64'),
      'pillars': convert_to_feature(pillars.tobytes(), 'bytes'),
      'indices': convert_to_feature(indices.tobytes(), 'bytes'),
      'bbox/class': convert_to_feature(classes, 'int64_list'),
      'bbox/ymin': convert_to_feature(ymin, 'float_list'),
      'bbox/xmin': convert_to_feature(xmin, 'float_list'),
      'bbox/ymax': convert_to_feature(ymax, 'float_list'),
      'bbox/xmax': convert_to_feature(xmax, 'float_list'),
      'bbox/heading': convert_to_feature(heading, 'float_list'),
      'bbox/z': convert_to_feature(z, 'float_list'),
      'bbox/height': convert_to_feature(height, 'float_list'),
      'bbox/difficulty': convert_to_feature(difficulty, 'int64_list'),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  serialized_example = example.SerializeToString()
  return serialized_example


class ExampleDecoderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (2, 10, 1, 1),
      (3, 2, 10, 10),
  )
  def test_shape(self, num_pillars, num_points_per_pillar,
                 num_features_per_point, num_boxes):
    image_config = cfg.ImageConfig()
    pillar_config = cfg.PillarsConfig()
    pillar_config.num_pillars = num_pillars
    pillar_config.num_points_per_pillar = num_points_per_pillar
    pillar_config.num_features_per_point = num_features_per_point

    decoder = decoders.ExampleDecoder(image_config, pillar_config)
    serialized_example = _mock_serialized_example(num_pillars,
                                                  num_points_per_pillar,
                                                  num_features_per_point,
                                                  num_boxes)
    decoded_example = decoder.decode(
        tf.convert_to_tensor(value=serialized_example))
    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_example)

    self.assertAllEqual(
        (num_pillars, num_points_per_pillar, num_features_per_point),
        results['pillars'].shape)
    self.assertAllEqual(
        (num_pillars, 2), results['indices'].shape)
    self.assertAllEqual(
        (num_boxes,), results['gt_classes'].shape)
    self.assertAllEqual(
        (num_boxes, 4), results['gt_boxes'].shape)
    self.assertAllEqual(
        (num_boxes, 1), results['gt_attributes']['heading'].shape)
    self.assertAllEqual(
        (num_boxes, 1), results['gt_attributes']['z'].shape)
    self.assertAllEqual(
        (num_boxes, 1), results['gt_attributes']['height'].shape)
    self.assertAllEqual(
        (num_boxes,), results['gt_difficulty'].shape)


if __name__ == '__main__':
  tf.test.main()
