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

"""Tests for WOD processor."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.utils.wod_processor import WodProcessor
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2


def _mock_label(x, y, length, width, label_type, num_lidar_points):
  label = label_pb2.Label()
  label.box.center_x = x
  label.box.center_y = y
  label.box.center_z = 0.0
  label.box.length = length
  label.box.width = width
  label.box.height = 1.0
  label.box.heading = 0.0
  label.type = label_type
  label.detection_difficulty_level = 1
  label.num_lidar_points_in_box = num_lidar_points
  return label


class WodProcessorTest(tf.test.TestCase, parameterized.TestCase):

  def test_compute_pillars_and_labels(self):
    # Mock image and pillars config and construct WodProcessor.
    image_config = cfg.ImageConfig(
        x_range=(-10.0, 10.0),
        y_range=(-10.0, 10.0),
        resolution=1.0)
    num_pillars = 10
    num_points_per_pillar = 10
    num_features_per_point = 10
    pillars_config = cfg.PillarsConfig(
        num_pillars=num_pillars,
        num_points_per_pillar=num_points_per_pillar,
        num_features_per_point=num_features_per_point)
    wod_processor = WodProcessor(image_config, pillars_config)

    # Mock point cloud as the input of compute_pillars.
    num_points_in_point_cloud = 10
    num_raw_features = 5
    points = np.ones([num_points_in_point_cloud, num_raw_features], np.float32)
    points_loc = np.random.randint(
        low=0, high=image_config.height, size=[num_points_in_point_cloud, 2])

    # Check if the computed pillars and indices have expected shapes.
    pillars, indices, k = wod_processor.compute_pillars(points, points_loc)
    self.assertAllEqual(
        pillars.shape.as_list(),
        [num_pillars, num_points_per_pillar, num_features_per_point])
    self.assertAllEqual(indices.shape.as_list(), [num_pillars, 2])
    self.assertEqual(k, num_pillars)

    # Mock labels as the input of extract_labels.
    frame = dataset_pb2.Frame()
    # Should be filtered due to SIGN
    frame.laser_labels.append(_mock_label(0.0, 0.0, 2.0, 1.0, 3, 10))
    # Should be filtered due to no lidar points
    frame.laser_labels.append(_mock_label(0.0, 0.0, 2.0, 1.0, 1, 0))
    # Should be filtered due to empty area
    frame.laser_labels.append(_mock_label(0.0, 0.0, 0.0, 0.0, 1, 10))
    # Should be processed
    frame.laser_labels.append(_mock_label(0.0, 0.0, 2.0, 1.0, 1, 10))

    # Check if the extracted labels have expected shapes.
    results = wod_processor.extract_labels(frame)
    self.assertAllEqual(results[0].shape.as_list(), [1,])


if __name__ == '__main__':
  tf.test.main()
