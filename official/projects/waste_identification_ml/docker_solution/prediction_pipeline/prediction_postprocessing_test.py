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
from unittest import mock
import numpy as np
from official.projects.waste_identification_ml.docker_solution.prediction_pipeline import prediction_postprocessing


class PostprocessingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.results1 = {
        'detection_boxes': [np.array([[0, 0, 100, 100], [100, 100, 200, 200]])],
        'detection_masks': [
            np.zeros((1, 512, 1024), dtype=np.uint8),
            np.ones((1, 512, 1024), dtype=np.uint8),
        ],
        'detection_scores': [[0.9, 0.8]],
        'detection_classes': [1, 2],
        'detection_classes_names': ['class_1', 'class_2'],
    }

    self.results2 = {
        'detection_boxes': [
            np.array([[50, 50, 150, 150], [150, 150, 250, 250]])
        ],
        'detection_masks': [
            np.full((1, 512, 1024), 0.5, dtype=np.uint8),
            np.full((1, 512, 1024), 0.5, dtype=np.uint8),
        ],
        'detection_scores': [[0.9, 0.8]],
        'detection_classes': [2, 1],
        'detection_classes_names': ['class_2', 'class_1'],
    }

    self.category_indices = [[1, 2], [2, 1]]

    self.category_index = {
        1: {'id': 1, 'name': 'class_1'},
        2: {'id': 2, 'name': 'class_2'},
    }
    self.height = 512
    self.width = 1024

  def test_merge_predictions(self):
    results = prediction_postprocessing.merge_predictions(
        [self.results1, self.results2],
        0.8,
        self.category_indices,
        self.category_index,
        4,
    )

    self.assertEqual(results['num_detections'], 4)
    self.assertEqual(results['detection_scores'].shape, (4,))
    self.assertEqual(results['detection_boxes'].shape, (4, 4))
    self.assertEqual(results['detection_classes'].shape, (4,))
    self.assertEqual(
        results['detection_classes_names'],
        ['class_1', 'class_2', 'class_1', 'class_2'],
    )
    self.assertEqual(results['detection_masks_reframed'].shape, (4, 512, 1024))

  @mock.patch('postprocessing.find_similar_masks')
  def test_merge_predictions_calls_find_similar_masks(
      self, mock_find_similar_masks
  ):
    prediction_postprocessing.merge_predictions(
        [self.results1, self.results2],
        0.8,
        self.category_indices,
        self.category_index,
        4,
    )

    mock_find_similar_masks.assert_called_once_with(
        self.results1,
        self.results2,
        4,
        0.8,
        self.category_indices,
        self.category_index,
        0.3 * 512 * 1024,
    )

  def test_merge_predictions_with_empty_results(self):
    results = prediction_postprocessing.merge_predictions(
        [{}, {}],
        0.8,
        self.category_indices,
        self.category_index,
        4,
    )

    self.assertEqual(results['num_detections'], 0)
    self.assertEqual(results['detection_scores'].shape, (0,))
    self.assertEqual(results['detection_boxes'].shape, (0, 4))
    self.assertEqual(results['detection_classes'].shape, (0,))
    self.assertEqual(results['detection_classes_names'], [])
    self.assertEqual(results['detection_masks_reframed'].shape, (0, 512, 1024))

  def test_merge_predictions_with_invalid_category_indices(self):
    category_indices = [[1, 3], [2, 4]]

    with self.assertRaises(ValueError):
      prediction_postprocessing.merge_predictions(
          [self.results1, self.results2],
          0.8,
          category_indices,
          self.category_index,
          4,
      )

  def test_transform_bounding_boxes(self):
    results = {
        'detection_boxes': np.array([[
            [0.1, 0.2, 0.4, 0.5],  # Normalized coordinates
            [0.3, 0.3, 0.6, 0.7],
        ]])
    }

    # Expected output for the adjusted height and width
    expected_transformed_boxes = [
        [
            int(0.1 * self.height),
            int(0.2 * self.width),
            int(0.4 * self.height),
            int(0.5 * self.width),
        ],
        [
            int(0.3 * self.height),
            int(0.3 * self.width),
            int(0.6 * self.height),
            int(0.7 * self.width),
        ],
    ]

    transformed_boxes = prediction_postprocessing._transform_bounding_boxes(
        results
    )

    self.assertEqual(transformed_boxes, expected_transformed_boxes)
