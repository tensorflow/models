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

import tempfile
import time
import unittest

import numpy as np

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import batched_io

TEST_IMAGE_PATH = "/path/to/test_image.jpg"
TEST_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)
TEST_MASKS = np.ones((10, 10), dtype=np.uint8)
TEST_BOXES = [[0, 0, 5, 5]]


class BatchedIoTest(unittest.TestCase):
  """Tests for batched_io."""

  def test_extract_masked_object(self):
    image = TEST_IMAGE.copy()
    image[0:6, 0:6] = [128, 128, 128]  # Add gray square
    mask = TEST_MASKS.copy()
    mask[1:5, 1:5] = 1  # Mask within gray square
    box = [0, 0, 6, 6]  # Bounding box of gray square

    masked_object = batched_io.extract_masked_object(image, mask, box)

    with self.subTest(name="CropDimensionsCorrect"):
      # Check crop dimension, should be full bounding box dimensions.
      self.assertEqual(masked_object.shape, (6, 6, 3))
    with self.subTest(name="BackgroundIsBlack"):
      boolean_mask = masked_object.astype(bool)
      self.assertTrue(np.all(masked_object[~boolean_mask] == 1))
    with self.subTest(name="ObjectIsRGB"):
      self.assertEqual(masked_object.shape[-1], 3)  # RGB

  def test_save_masked_objects_saves_files(self):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = [
        np.ones((10, 10), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
    ]
    masks[1][5:, 5:] = 1
    boxes = [[0, 0, 5, 5], [5, 5, 10, 10]]
    source_image_path = TEST_IMAGE_PATH

    with tempfile.TemporaryDirectory() as temp_dir:
      saved_files = batched_io.save_masked_objects(
          image, masks, boxes, source_image_path, temp_dir
      )

      self.assertEqual(len(saved_files), 2)

  def test_batched_mask_writer_queues_saves(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      writer = batched_io.BatchedMaskWriter(output_dir=temp_dir)

      writer.add_batch(TEST_IMAGE, TEST_MASKS, TEST_BOXES, TEST_IMAGE_PATH)
      writer.add_batch(
          TEST_IMAGE, TEST_MASKS, TEST_BOXES, "/another/path/to/image.jpg"
      )
      time.sleep(1)

      self.assertEqual(len(writer.futures), 2)


if __name__ == "__main__":
  unittest.main()
