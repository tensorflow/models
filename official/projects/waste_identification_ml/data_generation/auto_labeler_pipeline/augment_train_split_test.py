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

"""Unit tests for augment_train_split."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import PIL.Image

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import augment_train_split


class AugmentTrainSplitTest(parameterized.TestCase):

  def test_is_augmented_filename(self):
    self.assertFalse(augment_train_split.is_augmented_filename("image1.jpg"))
    self.assertFalse(augment_train_split.is_augmented_filename("image1.png"))
    self.assertTrue(augment_train_split.is_augmented_filename("img_vflip.jpg"))
    self.assertTrue(augment_train_split.is_augmented_filename("img_blur.jpg"))
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_rot45.jpeg")
    )

  def test_discover_target_folders_single_variant(self):
    train_dir = self.create_tempdir("train").full_path
    class_a = os.path.join(train_dir, "class_a")
    class_b = os.path.join(train_dir, "class_b")
    os.makedirs(class_a)
    os.makedirs(class_b)

    targets = augment_train_split.discover_target_folders(
        train_dir, crop_variants=("raw",)
    )
    self.assertEqual(
        targets, [("class_a", class_a), ("class_b", class_b)]
    )

  def test_discover_target_folders_multi_variant(self):
    train_dir = self.create_tempdir("train").full_path
    class_a = os.path.join(train_dir, "class_a")
    var1 = os.path.join(class_a, "raw")
    var2 = os.path.join(class_a, "imagenet_mean_background")
    os.makedirs(var1)
    os.makedirs(var2)

    targets = augment_train_split.discover_target_folders(
        train_dir, crop_variants=("raw", "imagenet_mean_background")
    )
    self.assertEqual(
        targets,
        [
            ("class_a/raw", var1),
            ("class_a/imagenet_mean_background", var2),
        ],
    )

  def test_validate_no_pre_existing_augmentations(self):
    class_dir = self.create_tempdir("class_a").full_path
    # Normal image should not trigger FileExistsError
    PIL.Image.new("RGB", (10, 10)).save(
        os.path.join(class_dir, "orig.jpg")
    )
    augment_train_split.validate_no_pre_existing_augmentations(
        [("class_a", class_dir)]
    )

    # Augmented image should raise FileExistsError
    PIL.Image.new("RGB", (10, 10)).save(
        os.path.join(class_dir, "orig_vflip.jpg")
    )
    with self.assertRaises(FileExistsError):
      augment_train_split.validate_no_pre_existing_augmentations(
          [("class_a", class_dir)]
      )

  def test_process_class_folder(self):
    class_dir = self.create_tempdir("class_a").full_path
    orig_path = os.path.join(class_dir, "test_img.jpg")
    PIL.Image.new("RGB", (32, 32), color=(255, 0, 0)).save(orig_path)

    augment_train_split.process_class_folder(
        class_name="class_a",
        class_folder=class_dir,
        augmentations_to_apply=("vflip", "hflip", "rot90"),
        rotation_fill_color=(124, 116, 104),
    )

    files = sorted(os.listdir(class_dir))
    self.assertEqual(
        files,
        [
            "test_img.jpg",
            "test_img_hflip.jpg",
            "test_img_rot90.jpg",
            "test_img_vflip.jpg",
        ],
    )


if __name__ == "__main__":
  absltest.main()
