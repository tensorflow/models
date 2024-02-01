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

import numpy as np
import tensorflow as tf, tf_keras
from official.projects.waste_identification_ml.data_generation import utils


def compare_elements(elem_list1, elem_list2):
  if len(elem_list1) != len(elem_list2):
    return False

  for elem1, elem2 in zip(elem_list1, elem_list2):
    for key in elem1:
      if key not in elem2:
        return False
      if isinstance(elem1[key], np.ndarray) or isinstance(
          elem2[key], np.ndarray
      ):
        if not np.array_equal(elem1[key], elem2[key]):
          return False
      else:
        if elem1[key] != elem2[key]:
          return False

  return True


class MyTest(tf.test.TestCase):

  def test_convert_coordinates(self):
    coord = [10.0, 20.0, 30.0, 40.0]
    expected_output = [10.0, 20.0, 40.0, 60.0]
    actual_output = utils.convert_bbox_format(coord)
    self.assertEqual(expected_output, actual_output)

  def test_area_key(self):
    masks = [{'area': 10.0}, {'area': 20.0}, {'area': 30.0}]
    upper_multiplier = 1.5
    lower_multiplier = 0.5
    leng = [i['area'] for i in masks]
    q1, _, q3 = np.percentile(leng, [25, 50, 75])
    iqr = q3 - q1
    expected_upper_bound = q3 + upper_multiplier * iqr
    expected_lower_bound = q1 * lower_multiplier
    actual_upper_bound, actual_lower_bound = utils._calculate_area_bounds(
        masks, upper_multiplier, lower_multiplier
    )
    self.assertEqual(
        (expected_upper_bound, expected_lower_bound),
        (actual_upper_bound, actual_lower_bound),
    )

  def test_square_bbox(self):
    bbox = [0.0, 0.0, 2.0, 2.0]
    expected_ratio = 1.0
    actual_ratio = utils._aspect_ratio(bbox)
    self.assertEqual(expected_ratio, actual_ratio)

  def test_same_size_masks(self):
    elem1 = {
        'segmentation': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'area': 3,
    }

    elem2 = {
        'segmentation': np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]]),
        'area': 3,
    }

    expected_score = 2.0 / 3.0  # Intersection is 2, smaller mask area is 3
    actual_score = utils._calculate_intersection_score(elem1, elem2)
    self.assertAlmostEqual(expected_score, actual_score)

  def test_different_size_masks_error(self):
    elem1 = {'segmentation': np.array([[1, 0], [0, 1]]), 'area': 2}

    elem2 = {
        'segmentation': np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]]),
        'area': 3,
    }

    with self.assertRaises(ValueError) as context:
      utils._calculate_intersection_score(elem1, elem2)

    self.assertEqual(
        str(context.exception), 'The masks must have the same dimensions.'
    )


class TestFilterNestedSimilarMasks(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Create some mock binary mask data to use in the tests
    self.mask1 = np.array([[0, 1], [1, 0]])
    self.mask2 = np.array([[1, 0], [0, 1]])
    self.larger_mask = np.array([[1, 1], [1, 1]])

  def test_same_size_masks(self):
    # Test the case where all masks are of the same size
    elements = [
        {'segmentation': self.mask1, 'area': 2},
        {'segmentation': self.mask2, 'area': 2},
    ]
    expected_output = elements  # All masks are retained as none are nested
    actual_output = utils.filter_nested_similar_masks(elements)
    self.assertEqual(actual_output, expected_output)

  def test_nested_masks(self):
    # Test the case where one mask is nested within another
    elements = [
        {'segmentation': self.mask1, 'area': 2},
        {'segmentation': self.larger_mask, 'area': 4},
    ]
    expected_output = [{
        'segmentation': self.larger_mask,
        'area': 4,
    }]  # Only the larger mask is retained
    actual_output = utils.filter_nested_similar_masks(elements)
    self.assertEqual(actual_output, expected_output)


class TestGenerateCocoJson(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.image = np.array([[0, 1], [1, 0]])
    self.masks = [np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])]
    self.category_name = 'example_category'
    self.file_name = 'example_file'

  def test_generate_coco_json(self):
    coco_dict = utils.generate_coco_json(
        masks=self.masks,
        image=self.image,
        category_name=self.category_name,
        file_name=self.file_name,
    )

    # Check the keys present in the output dictionary
    self.assertIn('images', coco_dict)
    self.assertIn('categories', coco_dict)
    self.assertIn('annotations', coco_dict)

    # Check the file name in the images dictionary
    self.assertEqual(coco_dict['images'][0]['file_name'], self.file_name)

    # Check the category name in the categories dictionary
    self.assertEqual(coco_dict['categories'][0]['name'], self.category_name)


if __name__ == '__main__':
  tf.test.main()
