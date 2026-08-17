# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Unit tests for auto_labeler_pipeline split_train_val."""

import os

from absl.testing import absltest
from absl.testing import parameterized

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import split_train_val


class SplitTrainValTest(parameterized.TestCase):
  """Tests dataset discovery, filtering, validation, and splitting."""

  def setUp(self):
    super().setUp()
    self.root_dir = self.create_tempdir().full_path
    self.dataset_a = os.path.join(self.root_dir, "dataset_a")
    self.dataset_b = os.path.join(self.root_dir, "dataset_b")
    os.makedirs(os.path.join(self.dataset_a, "images"), exist_ok=True)
    os.makedirs(os.path.join(self.dataset_b, "images"), exist_ok=True)

  def test_discover_dataset_directories_valid(self):
    """Verifies dataset subfolders are naturally sorted and discovered."""
    directories = split_train_val.discover_dataset_directories(self.root_dir)
    self.assertEqual(
        directories,
        [
            ("dataset_a", self.dataset_a),
            ("dataset_b", self.dataset_b),
        ],
    )

  def test_discover_dataset_directories_not_found(self):
    """Ensures FileNotFoundError is raised when root_dir does not exist."""
    with self.assertRaises(FileNotFoundError):
      split_train_val.discover_dataset_directories(
          os.path.join(self.root_dir, "non_existent")
      )

  def test_discover_dataset_directories_empty(self):
    """Ensures ValueError is raised when root_dir has no subdirectories."""
    empty_dir = self.create_tempdir().full_path
    with self.assertRaises(ValueError):
      split_train_val.discover_dataset_directories(empty_dir)

  def test_validate_dataset_paths_valid(self):
    """Verifies expected source and output paths are returned."""
    directories = [("dataset_a", self.dataset_a)]
    validated = split_train_val.validate_dataset_paths(
        directories, "images", "train_val_images"
    )
    self.assertEqual(
        validated,
        [(
            "dataset_a",
            os.path.join(self.dataset_a, "images"),
            os.path.join(self.dataset_a, "train_val_images"),
        )],
    )

  def test_validate_dataset_paths_missing_input(self):
    """Ensures FileNotFoundError is raised if input images folder is missing."""
    bad_dir = os.path.join(self.root_dir, "dataset_missing")
    os.makedirs(bad_dir, exist_ok=True)
    with self.assertRaises(FileNotFoundError):
      split_train_val.validate_dataset_paths(
          [("dataset_missing", bad_dir)], "images", "train_val_images"
      )

  def test_validate_dataset_paths_existing_output(self):
    """Ensures FileExistsError is raised if output folder already exists."""
    os.makedirs(os.path.join(self.dataset_a, "train_val_images"), exist_ok=True)
    with self.assertRaises(FileExistsError):
      split_train_val.validate_dataset_paths(
          [("dataset_a", self.dataset_a)], "images", "train_val_images"
      )

  def test_get_sorted_image_names(self):
    """Verifies images are sorted naturally and non-images/subdirs ignored."""
    images_dir = os.path.join(self.dataset_a, "images")
    for name in ["img10.jpg", "img2.png", "img1.JPG", "readme.txt"]:
      with open(os.path.join(images_dir, name), "w") as f:
        f.write("test")
    os.makedirs(os.path.join(images_dir, "subdir.jpg"), exist_ok=True)

    sorted_names = split_train_val.get_sorted_image_names(images_dir)
    self.assertEqual(sorted_names, ["img1.JPG", "img2.png", "img10.jpg"])

  def test_filter_every_nth_image(self):
    """Verifies keep_every_nth filtering logic."""
    names = [f"img_{i}.jpg" for i in range(10)]
    filtered_2 = split_train_val.filter_every_nth_image(names, 2)
    self.assertEqual(
        filtered_2,
        ["img_0.jpg", "img_2.jpg", "img_4.jpg", "img_6.jpg", "img_8.jpg"],
    )
    filtered_3 = split_train_val.filter_every_nth_image(names, 3)
    self.assertEqual(
        filtered_3, ["img_0.jpg", "img_3.jpg", "img_6.jpg", "img_9.jpg"]
    )

  def test_check_for_duplicates(self):
    """Verifies FileExistsError is raised when files collide."""
    images_dir = os.path.join(self.dataset_a, "images")
    with open(os.path.join(images_dir, "existing.jpg"), "w") as f:
      f.write("data")

    with self.assertRaises(FileExistsError):
      split_train_val.check_for_duplicates(["existing.jpg"], images_dir)

  def test_process_dataset_flat(self):
    """Verifies end-to-end splitting on a dataset with flat images folder.

    train_ratio is treated as the VAL fraction; train gets the majority.
    """
    images_dir = os.path.join(self.dataset_a, "images")
    for i in range(10):
      with open(os.path.join(images_dir, f"img_{i}.jpg"), "w") as f:
        f.write("content")

    output_dir = os.path.join(self.dataset_a, "train_val_images")
    train_count, val_count = split_train_val.process_dataset(
        dataset_name="dataset_a",
        source_folder=images_dir,
        output_folder=output_dir,
        train_split_name="train",
        val_split_name="val",
        keep_every_nth=2,
        train_ratio=0.2,
    )

    # 10 images total -> keep every 2nd -> 5 images kept
    # train_ratio 0.2 is the VAL fraction: int(5 * 0.2) = 1 val, 4 train
    self.assertEqual(train_count, 4)
    self.assertEqual(val_count, 1)
    # First filtered image (img_0.jpg) goes to val; the rest go to train.
    self.assertTrue(
        os.path.exists(os.path.join(output_dir, "val", "img_0.jpg"))
    )
    self.assertTrue(
        os.path.exists(os.path.join(output_dir, "train", "img_2.jpg"))
    )


if __name__ == "__main__":
  absltest.main()
