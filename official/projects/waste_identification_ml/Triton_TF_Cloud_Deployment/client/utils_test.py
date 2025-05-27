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

import os
import tempfile
import unittest
import numpy as np
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import utils


class TestLoadLabels(unittest.TestCase):

  def test_load_labels(self):
    # Create a temporary CSV file within the test
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_csv:
      temp_csv.write('Label\nBottle\nCan\nCup\n')
      temp_csv_path = temp_csv.name

    try:
      # Call the function under test
      category_indices, category_index = utils.load_labels(temp_csv_path)

      # Expected results
      expected_list = ['Label', 'Bottle', 'Can', 'Cup']
      expected_dict = {
          1: {'id': 1, 'name': 'Label', 'supercategory': 'objects'},
          2: {'id': 2, 'name': 'Bottle', 'supercategory': 'objects'},
          3: {'id': 3, 'name': 'Can', 'supercategory': 'objects'},
          4: {'id': 4, 'name': 'Cup', 'supercategory': 'objects'},
      }

      self.assertEqual(category_indices, expected_list)
      self.assertEqual(category_index, expected_dict)

    finally:
      # Ensure the temporary file is deleted even if assertions fail
      os.remove(temp_csv_path)

  def test_files_paths_with_images(self):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
      # Create some image and non-image files
      filenames = ['img2.jpg', 'img1.png', 'doc1.txt', 'photo.gif']
      for filename in filenames:
        open(os.path.join(temp_dir, filename), 'a').close()

      # Call the function under test
      result = utils.files_paths(temp_dir)

      # Expected image files sorted naturally
      expected = [
          os.path.join(temp_dir, 'img1.png'),
          os.path.join(temp_dir, 'img2.jpg'),
          os.path.join(temp_dir, 'photo.gif'),
      ]

      self.assertEqual(result, expected)

  def test_files_paths_with_no_images(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      # Create only non-image files
      filenames = ['doc1.txt', 'readme.md']
      for filename in filenames:
        open(os.path.join(temp_dir, filename), 'a').close()

      result = utils.files_paths(temp_dir)
      self.assertEqual(result, [])  # Should return an empty list

  def test_files_paths_empty_folder(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      result = utils.files_paths(temp_dir)
      self.assertEqual(result, [])

  def test_resize_multiple_masks(self):
    # Create two 5x5 masks
    mask1 = np.zeros((5, 5), dtype=np.uint8)
    mask2 = np.ones((5, 5), dtype=np.uint8)
    masks = np.array([mask1, mask2])

    resized_masks = utils.resize_each_mask(masks, 3, 3)

    self.assertEqual(resized_masks.shape, (2, 3, 3))
    self.assertTrue((resized_masks[0] == 0).all())
    self.assertTrue((resized_masks[1] == 1).all())

  def test_keeps_biggest_mask(self):
    # Create larger masks to satisfy the area check (area >= 4000)
    mask_small = np.zeros((100, 100), dtype=int)
    mask_small[0:40, 0:40] = 1  # Area = 1600 (will be skipped)

    mask_medium = np.zeros((100, 100), dtype=int)
    mask_medium[0:70, 0:70] = 1  # Area = 4900 (passes area condition)

    mask_large = np.zeros((100, 100), dtype=int)
    mask_large[:, :] = 1  # Area = 10000 (passes area condition)

    masks = np.array([mask_small, mask_medium, mask_large])

    # Run filter_masks without specifying area_threshold
    result = utils.filter_masks(masks, iou_threshold=0.5)

    # Expect only the largest mask (index 2) to remain
    self.assertEqual(result, [2])


if __name__ == '__main__':
  unittest.main()
