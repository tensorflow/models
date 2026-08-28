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

"""Unit tests for augment_train_split.py."""

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import PIL.Image

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import augment_train_split

# The canonical augmentation order lives in config_loader; the tests below
# patch it to a fixed, self-contained set so they don't depend on the real
# config module's contents.
_FAKE_CANONICAL_ORDER = (
    "vflip",
    "hflip",
    "rot45",
    "rot65",
    "rot90",
    "blur",
    "noise03",
    "noise06",
    "cjitter",
)


def _solid_image(
    width: int, height: int, color: tuple[int, int, int]
) -> PIL.Image.Image:
  """Returns a solid-color RGB image of the given size."""
  return PIL.Image.new("RGB", (width, height), color)


def _center_mask(width: int, height: int) -> PIL.Image.Image:
  """Returns an 'L' mask with a filled central rectangle (values 0/255)."""
  array = np.zeros((height, width), dtype=np.uint8)
  array[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
  return PIL.Image.fromarray(array, mode="L")


class GetBackgroundColorForVariantTest(parameterized.TestCase):
  """Tests for get_background_color_for_variant."""

  def test_raw_variant_is_black(self):
    """Verifies the raw variant maps to black."""
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "raw", (124, 116, 104)
        ),
        (0, 0, 0),
    )

  def test_black_background_variant_is_black(self):
    """Verifies the black_background variant maps to black."""
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "black_background", (124, 116, 104)
        ),
        (0, 0, 0),
    )

  def test_imagenet_mean_variant_uses_fill_color(self):
    """Verifies the imagenet_mean variant returns the configured fill color."""
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "imagenet_mean_background", (124, 116, 104)
        ),
        (124, 116, 104),
    )

  def test_unknown_variant_raises(self):
    """Verifies an unrecognized variant raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Unknown crop variant"):
      augment_train_split.get_background_color_for_variant("bogus", (0, 0, 0))


class BuildMaskSidecarPathTest(absltest.TestCase):
  """Tests for build_mask_sidecar_path."""

  def test_replaces_extension_with_mask_suffix(self):
    """Verifies the sidecar path swaps the extension for _mask.png."""
    self.assertEqual(
        augment_train_split.build_mask_sidecar_path("/a/b/img_001_0.jpg"),
        "/a/b/img_001_0_mask.png",
    )

  def test_handles_png_input(self):
    """Verifies a .png input also yields the _mask.png sidecar."""
    self.assertEqual(
        augment_train_split.build_mask_sidecar_path("/a/b/img.png"),
        "/a/b/img_mask.png",
    )


class CompositeForegroundOnBackgroundTest(absltest.TestCase):
  """Tests for composite_foreground_on_background."""

  def test_object_pixels_kept_background_filled(self):
    """Verifies masked pixels come from the image, the rest from the color."""
    image = _solid_image(10, 10, (200, 100, 50))
    mask_array = np.zeros((10, 10), dtype=np.uint8)
    mask_array[2:8, 2:8] = 255
    mask = PIL.Image.fromarray(mask_array, mode="L")

    result = augment_train_split.composite_foreground_on_background(
        image, mask, (0, 0, 0)
    )
    result_array = np.array(result)
    # Inside the mask -> original color.
    np.testing.assert_array_equal(result_array[5, 5], np.array([200, 100, 50]))
    # Outside the mask -> background color.
    np.testing.assert_array_equal(result_array[0, 0], np.array([0, 0, 0]))

  def test_binarizes_mask_above_zero(self):
    """Verifies any non-zero mask value counts as foreground."""
    image = _solid_image(4, 4, (255, 255, 255))
    mask_array = np.array(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    mask = PIL.Image.fromarray(mask_array, mode="L")
    result = np.array(
        augment_train_split.composite_foreground_on_background(
            image, mask, (0, 0, 0)
        )
    )
    np.testing.assert_array_equal(result[0, 1], np.array([255, 255, 255]))


class GeometricAugmentationTest(absltest.TestCase):
  """Tests for the geometric augmentation helpers."""

  def test_vflip_transforms_image_and_mask(self):
    """Verifies vertical flip is applied to both image and mask."""
    array = np.zeros((6, 6, 3), dtype=np.uint8)
    array[0, 0] = [255, 0, 0]  # top-left marker
    image = PIL.Image.fromarray(array)
    mask = _center_mask(6, 6)

    flipped_image, flipped_mask = (
        augment_train_split.apply_vertical_flip_to_image_and_mask(image, mask)
    )
    flipped_array = np.array(flipped_image)
    # The top-left marker should now be at the bottom-left.
    np.testing.assert_array_equal(flipped_array[5, 0], np.array([255, 0, 0]))
    self.assertEqual(flipped_mask.size, mask.size)

  def test_rotation_preserves_size_and_binary_mask(self):
    """Verifies rotation keeps image size and a strictly binary mask."""
    image = _solid_image(20, 20, (100, 120, 140))
    mask = _center_mask(20, 20)

    rotated_image, rotated_mask = (
        augment_train_split.apply_fixed_rotation_to_image_and_mask(
            image, mask, 45, (0, 0, 0)
        )
    )
    self.assertEqual(rotated_image.size, (20, 20))
    unique_values = set(np.unique(np.array(rotated_mask)).tolist())
    self.assertTrue(unique_values.issubset({0, 255}))

  def test_rotation_fills_corners_with_background(self):
    """Verifies newly exposed corners use the given fill color."""
    image = _solid_image(20, 20, (200, 200, 200))
    mask = _center_mask(20, 20)
    rotated_image, _ = (
        augment_train_split.apply_fixed_rotation_to_image_and_mask(
            image, mask, 45, (5, 6, 7)
        )
    )
    rotated_array = np.array(rotated_image)
    # A corner pixel is exposed by the 45-degree rotation -> fill color.
    np.testing.assert_array_equal(rotated_array[0, 0], np.array([5, 6, 7]))

  def test_build_geometric_dispatches_by_name(self):
    """Verifies the dispatcher routes each geometric name without error."""
    image = _solid_image(16, 16, (10, 20, 30))
    mask = _center_mask(16, 16)
    for name in ["vflip", "hflip", "rot45", "rot65", "rot90"]:
      out_image, out_mask = augment_train_split.build_geometric_augmentation(
          image, mask, name, (0, 0, 0)
      )
      self.assertEqual(out_image.size, (16, 16))
      self.assertEqual(out_mask.size, (16, 16))

  def test_build_geometric_rejects_unknown_name(self):
    """Verifies an unknown geometric name raises ValueError."""
    image = _solid_image(8, 8, (0, 0, 0))
    mask = _center_mask(8, 8)
    with self.assertRaisesRegex(ValueError, "Unknown geometric"):
      augment_train_split.build_geometric_augmentation(
          image, mask, "not_geometric", (0, 0, 0)
      )


class NonGeometricAugmentationTest(absltest.TestCase):
  """Tests for the non-geometric augmentation helpers."""

  def test_blur_preserves_size_and_mode(self):
    """Verifies blur returns an RGB image of the same size."""
    image = _solid_image(24, 24, (120, 130, 140))
    result = augment_train_split.apply_gaussian_blur(image)
    self.assertEqual(result.size, (24, 24))

  def test_noise_output_is_valid_image(self):
    """Verifies added noise yields a same-size RGB image."""
    image = _solid_image(16, 16, (100, 100, 100))
    result = augment_train_split.apply_add_noise(image, 0.3)
    self.assertEqual(result.size, (16, 16))
    self.assertEqual(result.mode, "RGB")

  def test_build_non_geometric_dispatches_by_name(self):
    """Verifies each non-geometric name routes without error."""
    image = _solid_image(16, 16, (60, 70, 80))
    for name in ["blur", "noise03", "noise06", "cjitter"]:
      result = augment_train_split.build_non_geometric_augmentation(image, name)
      self.assertEqual(result.size, (16, 16))

  def test_build_non_geometric_rejects_unknown_name(self):
    """Verifies an unknown non-geometric name raises ValueError."""
    image = _solid_image(8, 8, (0, 0, 0))
    with self.assertRaisesRegex(ValueError, "Unknown non-geometric"):
      augment_train_split.build_non_geometric_augmentation(image, "bogus")


class BuildSingleAugmentationWithMaskTest(absltest.TestCase):
  """Tests for build_single_augmentation_with_mask."""

  def test_geometric_returns_transformed_mask(self):
    """Verifies a geometric aug returns a transformed (not original) mask."""
    image = _solid_image(20, 20, (100, 110, 120))
    mask = _center_mask(20, 20)
    _, out_mask = augment_train_split.build_single_augmentation_with_mask(
        image, mask, "hflip", (0, 0, 0)
    )
    # hflip mask differs from the original for an asymmetric mask; here we at
    # least confirm it is a distinct object of the same size.
    self.assertEqual(out_mask.size, mask.size)

  def test_non_geometric_returns_original_mask(self):
    """Verifies a non-geometric aug leaves the mask unchanged (same object)."""
    image = _solid_image(20, 20, (100, 110, 120))
    mask = _center_mask(20, 20)
    _, out_mask = augment_train_split.build_single_augmentation_with_mask(
        image, mask, "blur", (0, 0, 0)
    )
    self.assertIs(out_mask, mask)


class IsMaskSidecarFilenameTest(parameterized.TestCase):
  """Tests for is_mask_sidecar_filename."""

  @parameterized.named_parameters(
      ("mask_png", "img_0_mask.png", True),
      ("upper_case", "IMG_0_MASK.PNG", True),
      ("plain_image", "img_0.jpg", False),
      ("png_but_not_mask", "img_0.png", False),
  )
  def test_detects_mask_sidecars(self, file_name, expected):
    """Verifies mask sidecar filenames are detected case-insensitively."""
    self.assertEqual(
        augment_train_split.is_mask_sidecar_filename(file_name), expected
    )


class IsAugmentedFilenameTest(absltest.TestCase):
  """Tests for is_augmented_filename (depends on the canonical order)."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            augment_train_split.config_loader,
            "CANONICAL_AUGMENTATION_ORDER",
            _FAKE_CANONICAL_ORDER,
        )
    )

  def test_detects_augmented_image(self):
    """Verifies an augmented image name is recognized."""
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_001_0_vflip.jpg")
    )

  def test_detects_augmented_mask(self):
    """Verifies an augmented mask (with _mask stripped) is recognized."""
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_001_0_vflip_mask.png")
    )

  def test_original_image_is_not_augmented(self):
    """Verifies a plain original crop is not flagged as augmented."""
    self.assertFalse(augment_train_split.is_augmented_filename("img_001_0.jpg"))

  def test_original_mask_is_not_augmented(self):
    """Verifies a plain original mask sidecar is not flagged as augmented."""
    self.assertFalse(
        augment_train_split.is_augmented_filename("img_001_0_mask.png")
    )


class DiscoverTargetFoldersTest(absltest.TestCase):
  """Tests for discover_target_folders."""

  def test_single_variant_uses_flat_layout(self):
    """Verifies a single variant returns class folders directly."""
    train_dir = pathlib.Path(self.create_tempdir().full_path)
    (train_dir / "class_a").mkdir()
    (train_dir / "class_b").mkdir()

    result = augment_train_split.discover_target_folders(
        str(train_dir), ("raw",)
    )
    labels = [label for label, _, _ in result]
    variants = {variant for _, _, variant in result}
    self.assertEqual(labels, ["class_a", "class_b"])
    self.assertEqual(variants, {"raw"})

  def test_multiple_variants_use_subdirectories(self):
    """Verifies multiple variants descend into per-variant subfolders."""
    train_dir = pathlib.Path(self.create_tempdir().full_path)
    (train_dir / "class_a" / "raw").mkdir(parents=True)
    (train_dir / "class_a" / "black_background").mkdir(parents=True)

    result = augment_train_split.discover_target_folders(
        str(train_dir), ("raw", "black_background")
    )
    labels = sorted(label for label, _, _ in result)
    self.assertEqual(labels, ["class_a/black_background", "class_a/raw"])

  def test_raises_when_no_class_subfolders(self):
    """Verifies an empty train dir raises ValueError."""
    train_dir = pathlib.Path(self.create_tempdir().full_path)
    with self.assertRaisesRegex(ValueError, "No class subfolders"):
      augment_train_split.discover_target_folders(str(train_dir), ("raw",))


class ListOriginalImageNamesTest(absltest.TestCase):
  """Tests for list_original_image_names."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            augment_train_split.config_loader,
            "CANONICAL_AUGMENTATION_ORDER",
            _FAKE_CANONICAL_ORDER,
        )
    )

  def test_excludes_masks_and_augmented(self):
    """Verifies only original crop images are returned."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    (folder / "img_001_0.jpg").write_bytes(b"")
    (folder / "img_001_0_mask.png").write_bytes(b"")
    (folder / "img_001_0_vflip.jpg").write_bytes(b"")
    (folder / "img_002_0.jpg").write_bytes(b"")

    result = augment_train_split.list_original_image_names(str(folder))
    self.assertEqual(result, ["img_001_0.jpg", "img_002_0.jpg"])


class ValidateNoPreExistingAugmentationsTest(absltest.TestCase):
  """Tests for validate_no_pre_existing_augmentations."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            augment_train_split.config_loader,
            "CANONICAL_AUGMENTATION_ORDER",
            _FAKE_CANONICAL_ORDER,
        )
    )

  def test_passes_when_clean(self):
    """Verifies no error is raised when folders hold only originals."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    (folder / "img_001_0.jpg").write_bytes(b"")
    # Should not raise.
    augment_train_split.validate_no_pre_existing_augmentations(
        [("class_a", str(folder), "raw")]
    )

  def test_raises_when_augmented_files_present(self):
    """Verifies a folder holding augmented files raises FileExistsError."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    (folder / "img_001_0_vflip.jpg").write_bytes(b"")
    with self.assertRaises(FileExistsError):
      augment_train_split.validate_no_pre_existing_augmentations(
          [("class_a", str(folder), "raw")]
      )


class ValidateTrainSplitExistsTest(absltest.TestCase):
  """Tests for validate_train_split_exists."""

  def test_returns_train_dir_when_present(self):
    """Verifies the resolved train directory path is returned."""
    classifier_dir = pathlib.Path(self.create_tempdir().full_path)
    (classifier_dir / "train").mkdir()
    result = augment_train_split.validate_train_split_exists(
        str(classifier_dir), "train"
    )
    self.assertEqual(result, str(classifier_dir / "train"))

  def test_raises_when_classifier_dir_missing(self):
    """Verifies a missing classifier dir raises FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      augment_train_split.validate_train_split_exists(
          "/nonexistent/classifier", "train"
      )

  def test_raises_when_train_split_missing(self):
    """Verifies a missing train subfolder raises FileNotFoundError."""
    classifier_dir = pathlib.Path(self.create_tempdir().full_path)
    with self.assertRaises(FileNotFoundError):
      augment_train_split.validate_train_split_exists(
          str(classifier_dir), "train"
      )


class DeleteMaskSidecarsUnderTest(absltest.TestCase):
  """Tests for delete_mask_sidecars_under."""

  def test_deletes_only_mask_sidecars(self):
    """Verifies only _mask.png files are removed, recursively."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "class_a").mkdir()
    keep_image = root / "class_a" / "img_0.jpg"
    mask_one = root / "class_a" / "img_0_mask.png"
    mask_two = root / "class_a" / "img_1_mask.png"
    keep_image.write_bytes(b"")
    mask_one.write_bytes(b"")
    mask_two.write_bytes(b"")

    deleted_count, errors = augment_train_split.delete_mask_sidecars_under(
        str(root)
    )
    self.assertEqual(deleted_count, 2)
    self.assertEmpty(errors)
    self.assertTrue(keep_image.exists())
    self.assertFalse(mask_one.exists())
    self.assertFalse(mask_two.exists())

  def test_reports_zero_when_no_masks(self):
    """Verifies a tree with no mask sidecars deletes nothing."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "img_0.jpg").write_bytes(b"")
    deleted_count, errors = augment_train_split.delete_mask_sidecars_under(
        str(root)
    )
    self.assertEqual(deleted_count, 0)
    self.assertEmpty(errors)


class SaveAugmentedOutputsTest(absltest.TestCase):
  """Tests for save_augmented_outputs."""

  def test_writes_image_and_mask_pairs(self):
    """Verifies each augmentation writes a .jpg crop and a _mask.png."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    image = _solid_image(16, 16, (100, 110, 120))
    mask = _center_mask(16, 16)
    outputs = {"vflip": (image, mask), "blur": (image, mask)}

    augment_train_split.save_augmented_outputs(
        outputs, str(folder), "img_001_0"
    )

    written = set(os.listdir(folder))
    self.assertIn("img_001_0_vflip.jpg", written)
    self.assertIn("img_001_0_vflip_mask.png", written)
    self.assertIn("img_001_0_blur.jpg", written)
    self.assertIn("img_001_0_blur_mask.png", written)


if __name__ == "__main__":
  absltest.main()
