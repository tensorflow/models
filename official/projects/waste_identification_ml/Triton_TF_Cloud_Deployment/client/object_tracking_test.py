# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import unittest

import pandas as pd

from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import object_tracking


TEST_IMAGES = pd.DataFrame({
    'x': [1, 2, 3],
    'y': [4, 5, 6],
    'frame': [0, 0, 1],
    'bbox_0': [1, 2, 3],
    'bbox_1': [4, 5, 6],
    'bbox_2': [7, 8, 9],
    'bbox_3': [10, 11, 12],
    'major_axis_length': [13, 14, 15],
    'minor_axis_length': [16, 17, 18],
    'perimeter': [19, 20, 21],
    'source_name': ['source_name_1', 'source_name_2', 'source_name_3'],
    'image_name': ['image_name_1', 'image_name_2', 'image_name_3'],
    'detection_scores': [0.1, 0.2, 0.3],
    'detection_classes_names': ['class_name_1', 'class_name_2', 'class_name_3'],
    'detection_classes': [1, 2, 3],
    'color': ['red', 'blue', 'green'],
    'creation_time': [100, 200, 300],
})


class ObjectTrackingTest(unittest.TestCase):

  def test_object_tracking_retains_columns(self):
    """Tests that object tracking correctly retains columns not used in tracking."""
    df = TEST_IMAGES.copy()
    expected_columns = [
        'source_name',
        'image_name',
        'detection_scores',
        'detection_classes_names',
        'detection_classes',
        'color',
        'creation_time',
    ]

    tracking_result = object_tracking.apply_tracking(df, 10, 10, 10)

    self.assertTrue(all(key in tracking_result for key in expected_columns))

  def test_object_tracking_drops_columns(self):
    """Tests that object tracking correctly drops unneeded columns."""
    df = TEST_IMAGES.copy()

    tracking_result = object_tracking.apply_tracking(df, 10, 10, 10)

    self.assertNotIn('frame', tracking_result.columns)

if __name__ == '__main__':
  unittest.main()
