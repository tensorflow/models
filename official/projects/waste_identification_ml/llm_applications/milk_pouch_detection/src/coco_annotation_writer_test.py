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

"""Unit tests for CocoAnnotationWriter class."""

import json
import os
import tempfile
import unittest

import numpy as np

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import coco_annotation_writer

# Simple binary mask (20x20 square in center of 100x100 image)
TEST_MASK = np.zeros((100, 100), dtype=np.uint8)
TEST_MASK[40:60, 40:60] = 1

# Bounding box [x, y, width, height]
TEST_BOX = np.array([40, 40, 20, 20])


class TestCocoAnnotationWriterAddImage(unittest.TestCase):
  """Tests for the add_image method."""

  def test_add_image_returns_correct_id(self):
    """Test that add_image returns the correct image ID."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    image_id = writer.add_image("/path/to/image1.jpg", 800, 600)

    self.assertEqual(image_id, 0)

  def test_add_image_increments_counter(self):
    """Test that image ID counter increments with each image."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    writer.add_image("/path/to/image1.jpg", 800, 600)
    writer.add_image("/path/to/image2.jpg", 1024, 768)
    id3 = writer.add_image("/path/to/image3.jpg", 640, 480)

    self.assertEqual(id3, 2)

  def test_add_image_extracts_basename(self):
    """Test that only the filename is stored, not the full path."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    writer.add_image("/long/path/to/directory/image.jpg", 800, 600)

    image_info = writer.coco_output["images"][0]
    self.assertEqual(image_info["file_name"], "image.jpg")

  def test_add_image_appends_to_list(self):
    """Test that multiple images are appended to the images list."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    writer.add_image("/path/to/image1.jpg", 800, 600)
    writer.add_image("/path/to/image2.jpg", 1024, 768)

    self.assertEqual(len(writer.coco_output["images"]), 2)


class TestCocoAnnotationWriterAddAnnotations(unittest.TestCase):
  """Tests for the add_annotations method."""

  def test_add_annotations_returns_count(self):
    """Test that add_annotations returns the number of annotations added."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    count = writer.add_annotations(
        image_id=0, boxes=[TEST_BOX], masks=[TEST_MASK]
    )

    self.assertEqual(count, 1)

  def test_add_annotations_multiple_objects(self):
    """Test adding multiple annotations for one image."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[10:30, 10:30] = 1
    box2 = np.array([10, 10, 20, 20])

    count = writer.add_annotations(
        image_id=0, boxes=[TEST_BOX, box2], masks=[TEST_MASK, mask2]
    )

    self.assertEqual(count, 2)

  def test_add_annotations_links_to_image_id(self):
    """Test that annotations are linked to the correct image ID."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    writer.add_annotations(image_id=42, boxes=[TEST_BOX], masks=[TEST_MASK])

    annotation = writer.coco_output["annotations"][0]
    self.assertEqual(annotation["image_id"], 42)


class TestCocoAnnotationWriterSave(unittest.TestCase):
  """Tests for the save method."""

  def test_save_preserves_data(self):
    """Test that saved data matches the internal coco_output."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")
    writer.add_image("/path/to/image.jpg", 800, 600)

    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = os.path.join(tmpdir, "test_output.json")
      writer.save(output_path)

      with open(output_path, "r") as f:
        saved_data = json.load(f)

      self.assertEqual(saved_data["images"], writer.coco_output["images"])


class TestCocoAnnotationWriterGetStatistics(unittest.TestCase):
  """Tests for the get_statistics method."""

  def test_get_statistics_returns_correct_image_count(self):
    """Test that statistics reflect the correct number of images."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")
    writer.add_image("/path/to/image1.jpg", 800, 600)
    writer.add_image("/path/to/image2.jpg", 1024, 768)

    stats = writer.get_statistics()

    self.assertEqual(stats["num_images"], 2)

  def test_get_statistics_returns_correct_annotation_count(self):
    """Test that statistics reflect the correct number of annotations."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    writer.add_annotations(
        image_id=0, boxes=[TEST_BOX, TEST_BOX], masks=[TEST_MASK, TEST_MASK]
    )

    stats = writer.get_statistics()

    self.assertEqual(stats["num_annotations"], 2)

  def test_get_statistics_initial_state(self):
    """Test that statistics are correct for a new writer with no data."""
    writer = coco_annotation_writer.CocoAnnotationWriter("test_category")

    stats = writer.get_statistics()

    self.assertEqual(stats["num_images"], 0)
    self.assertEqual(stats["num_annotations"], 0)


if __name__ == "__main__":
  unittest.main()
