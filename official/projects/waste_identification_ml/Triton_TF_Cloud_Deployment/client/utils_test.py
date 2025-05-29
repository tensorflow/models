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

import os
import tempfile
import unittest
import numpy as np
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import utils
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client.utils import BoundingBox
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client.utils import ImageSize


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

  def test_filter_with_boolean_indices(self):
    results = {
        'detection_masks': np.random.rand(1, 3, 5, 5),
        'detection_masks_resized': np.random.rand(3, 5, 5),
        'detection_boxes': np.random.rand(1, 3, 4),
        'detection_classes': np.array([[1, 2, 3]]),
        'detection_scores': np.array([[0.9, 0.8, 0.3]]),
        'image_info': np.array([[640, 480]]),
    }

    valid_indices = [True, False, True]

    output = utils.filter_detections(results, valid_indices)

    self.assertEqual(output['detection_masks'].shape[1], 2)
    self.assertEqual(output['detection_masks_resized'].shape[0], 2)
    self.assertEqual(output['detection_boxes'].shape[1], 2)
    self.assertEqual(output['detection_classes'].shape[1], 2)
    self.assertEqual(output['detection_scores'].shape[1], 2)
    self.assertTrue(np.array_equal(output['image_info'], results['image_info']))
    self.assertEqual(output['num_detections'][0], 2)

  def test_filter_with_integer_indices(self):
    results = {
        'detection_masks': np.random.rand(1, 4, 5, 5),
        'detection_masks_resized': np.random.rand(4, 5, 5),
        'detection_boxes': np.random.rand(1, 4, 4),
        'detection_classes': np.array([[1, 2, 3, 4]]),
        'detection_scores': np.array([[0.9, 0.8, 0.3, 0.6]]),
        'image_info': np.array([[640, 480]]),
    }

    valid_indices = [0, 2]  # Keep detections at index 0 and 2

    output = utils.filter_detections(results, valid_indices)

    self.assertEqual(output['detection_masks'].shape[1], 2)
    self.assertEqual(output['detection_masks_resized'].shape[0], 2)
    self.assertEqual(output['detection_boxes'].shape[1], 2)
    self.assertEqual(output['detection_classes'].shape[1], 2)
    self.assertEqual(output['detection_scores'].shape[1], 2)
    self.assertEqual(output['num_detections'][0], 2)

  def test_both_dimensions_below_min_size(self):
    height, width, min_size = 800, 900, 1024

    result = utils.adjust_image_size(height, width, min_size)

    self.assertEqual(result, (800, 900))  # No scaling should happen

  def test_height_below_min_size(self):
    height, width, min_size = 900, 1200, 1024

    result = utils.adjust_image_size(height, width, min_size)

    self.assertEqual(result, (900, 1200))  # No scaling

  def test_width_below_min_size(self):
    height, width, min_size = 1300, 800, 1024

    result = utils.adjust_image_size(height, width, min_size)

    self.assertEqual(result, (1300, 800))  # No scaling

  def test_both_dimensions_above_min_size(self):
    height, width, min_size = 2048, 1536, 1024
    expected_scale = min(height / min_size, width / min_size)
    expected_height = int(height / expected_scale)
    expected_width = int(width / expected_scale)

    result = utils.adjust_image_size(height, width, min_size)

    self.assertEqual(result, (expected_height, expected_width))

  def test_exact_min_size(self):
    height, width, min_size = 1024, 1024, 1024

    result = utils.adjust_image_size(height, width, min_size)

    self.assertEqual(result, (1024, 1024))  # Already meets the requirement

  def test_extract_and_resize_single_object(self):
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255  # white image

    # Define a simple binary mask (1 in a 4x4 box)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 1

    # Box coordinates match the mask
    boxes = np.array([[[2, 3, 6, 7]]], dtype=np.int32)  # shape (1, 1, 4)

    results = {'masks': [mask], 'boxes': boxes}

    cropped_objects = utils.extract_and_resize_objects(
        results, 'masks', 'boxes', image, resize_factor=0.5
    )

    self.assertEqual(len(cropped_objects), 1)
    obj = cropped_objects[0]

    # Original crop size is (4, 4), so resized should be (2, 2)
    self.assertEqual(obj.shape[:2], (2, 2))

    # Should still be 3 channels
    self.assertEqual(obj.shape[2], 3)

    # The output pixels in mask area should be non-zero
    self.assertTrue(np.any(obj > 0))

  def test_resize_bbox_scaling(self):
    bbox = BoundingBox(y1=50, x1=100, y2=150, x2=200)
    old_size = ImageSize(height=200, width=400)
    new_size = ImageSize(height=400, width=800)

    expected = (100, 200, 300, 400)
    result = utils.resize_bbox(bbox, old_size, new_size)

    self.assertEqual(result, expected)

  def test_resize_bbox_no_scaling(self):
    bbox = BoundingBox(y1=10, x1=20, y2=30, x2=40)
    old_size = ImageSize(height=100, width=100)
    new_size = ImageSize(height=100, width=100)

    expected = (10, 20, 30, 40)
    result = utils.resize_bbox(bbox, old_size, new_size)

    self.assertEqual(result, expected)


if __name__ == '__main__':
  unittest.main()
