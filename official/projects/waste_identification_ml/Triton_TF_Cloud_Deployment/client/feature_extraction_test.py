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
import numpy as np
import pandas as pd
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import feature_extraction

TEST_IMAGE = np.array(
    [
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55],
        [20, 30, 40, 50, 60],
        [25, 35, 45, 55, 65],
        [30, 40, 50, 60, 70],
    ],
    dtype=np.uint8,
)

# Create dummy masks (e.g., two masks)
TEST_MASKS = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
        ],
    ],
    dtype=np.int32,
)

# Create empty masks (all zeros)
EMPTY_MASKS = np.zeros((2, 5, 5), dtype=np.int32)

# Simulate the results dictionary, assuming masks are under the key 'masks'
TEST_RESULTS = {'masks': TEST_MASKS}
EMPTY_RESULTS = {'masks': EMPTY_MASKS}


# Expected DataFrame for comparison
COMPARISON_DATA = {
    'area': [4.0, 4.0],
    'bbox_0': [1, 3],
    'bbox_1': [1, 2],
    'bbox_2': [3, 5],
    'bbox_3': [3, 4],
    'convex_area': [4.0, 4.0],
    'bbox_area': [4.0, 4.0],
    'major_axis_length': [2.0, 2.0],
    'minor_axis_length': [2.0, 2.0],
    'eccentricity': [0.0, 0.0],
    'y': [1.5, 3.5],
    'x': [1.5, 2.5],
    'label': [1, 1],
    'mean_intensity': [32.5, 52.5],
    'max_intensity': [40.0, 60.0],
    'min_intensity': [25.0, 45.0],
    'perimeter': [4.0, 4.0],
}


class TestExtractProperties(unittest.TestCase):

  def test_extract_properties(self):
    # Call the function
    features_df = feature_extraction.extract_properties(
        TEST_IMAGE, TEST_RESULTS, 'masks'
    )
    # Check if the DataFrames are equal
    self.assertTrue(features_df.equals(pd.DataFrame(COMPARISON_DATA)))

  def test_extract_properties_empty_masks(self):
    """Test feature extraction with empty masks."""
    features_df = feature_extraction.extract_properties(
        TEST_IMAGE, EMPTY_RESULTS, 'masks'
    )
    # Expecting an empty DataFrame if there are no valid masks
    self.assertTrue(features_df.empty)


if __name__ == '__main__':
  unittest.main()
