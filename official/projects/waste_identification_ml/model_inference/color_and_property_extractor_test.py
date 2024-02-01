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

import unittest

import numpy as np
import pandas as pd

from official.projects.waste_identification_ml.model_inference import color_and_property_extractor


class ColorAndPropertyExtractor(unittest.TestCase):

  def test_extract_properties_and_object_masks(self):
    final_array = {
        'detection_masks_reframed': np.array([[
            [0, 1, 1, 0, 0],
        ]]),
        'detection_boxes': np.array([[[0, 0, 5, 5]]]),
    }
    original_image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    expected_list = [
        pd.DataFrame({
            'area': [2.0],
            'bbox-0': [0],
            'bbox-1': [1],
            'bbox-2': [1],
            'bbox-3': [3],
            'convex_area': [2.0],
            'bbox_area': [2.0],
            'major_axis_length': [2.0],
            'minor_axis_length': [0.0],
            'eccentricity': [1.0],
            'centroid-0': [0.0],
            'centroid-1': [1.5],
        }),
    ]
    expected_mask = [
        np.array([[
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]])
    ]

    result_list, result_masks = (
        color_and_property_extractor.extract_properties_and_object_masks(
            final_array, 5, 5, original_image
        )
    )

    self.assertTrue(expected_list[0].equals(result_list[0]))
    self.assertTrue(np.array_equal(expected_mask[0], result_masks[0]))

  def test_find_dominant_color(self):
    original_image = np.array([
        [
            (0, 0, 0),
            (255, 255, 255),
        ],
        [
            (128, 128, 128),
            (128, 128, 128),
        ],
    ])

    result = color_and_property_extractor.find_dominant_color(
        original_image, black_threshold=50
    )

    self.assertEqual(result, (170, 170, 170))

  def test_find_dominant_color_black(self):
    original_image = np.array([
        [
            (0, 0, 0),
            (0, 0, 0),
        ],
        [
            (0, 0, 0),
            (0, 0, 0),
        ],
    ])

    result = color_and_property_extractor.find_dominant_color(
        original_image, black_threshold=50
    )

    self.assertEqual(result, ('Na', 'Na', 'Na'))

  def test_est_color(self):
    result = color_and_property_extractor.est_color((255, 0, 0))

    self.assertEqual(result, 'red')

  def test_generic_color(self):
    test_colors = np.array(
        [(255, 0, 0), (55, 118, 171), (73, 128, 41), (231, 112, 13)]
    )
    expected_colors = ['red', 'blue', 'green', 'orange']

    result = color_and_property_extractor.get_generic_color_name(test_colors)

    self.assertEqual(result, expected_colors)


if __name__ == '__main__':
  unittest.main()
