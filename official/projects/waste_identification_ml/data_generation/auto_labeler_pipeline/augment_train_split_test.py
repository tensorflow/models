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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import PIL.Image
import PIL.ImageDraw

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import augment_train_split

_ROTATION_FILL_COLOR = (124, 116, 104)
_CROP_SIZE = (64, 64)
_OBJECT_POLYGON = [(15, 5), (55, 15), (60, 50), (30, 60), (10, 40)]
_OBJECT_COLOR = (200, 30, 30)


def _make_crop_and_mask(
    background_color: tuple[int, int, int],
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Returns an (image, mask) pair with a solid polygon on the background.

  The image is a filled polygon on a solid background; the mask is white
  inside the same polygon and black elsewhere.

  Args:
      background_color: RGB fill color for the background.

  Returns:
      A tuple of (image, mask).
  """
  image = PIL.Image.new("RGB", _CROP_SIZE, background_color)
  mask = PIL.Image.new("L", _CROP_SIZE, 0)
  PIL.ImageDraw.Draw(image).polygon(_OBJECT_POLYGON, fill=_OBJECT_COLOR)
  PIL.ImageDraw.Draw(mask).polygon(_OBJECT_POLYGON, fill=255)
  return image, mask


def _write_crop_and_mask_to(folder, base_name, background_color):
  """Writes <base_name>.jpg + <base_name>_mask.png into folder."""
  os.makedirs(folder, exist_ok=True)
  image, mask = _make_crop_and_mask(background_color)
  image.save(os.path.join(folder, f"{base_name}.jpg"), quality=95)
  mask.save(os.path.join(folder, f"{base_name}_mask.png"))


class AugmentTrainSplitTest(parameterized.TestCase):

  # ── Background-color-per-variant ────────────────────────────────────────

  def test_get_background_color_raw_and_black_are_black(self):
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "raw", _ROTATION_FILL_COLOR
        ),
        (0, 0, 0),
    )
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "black_background", _ROTATION_FILL_COLOR
        ),
        (0, 0, 0),
    )

  def test_get_background_color_mean_uses_rotation_fill_color(self):
    self.assertEqual(
        augment_train_split.get_background_color_for_variant(
            "imagenet_mean_background", _ROTATION_FILL_COLOR
        ),
        _ROTATION_FILL_COLOR,
    )

  def test_get_background_color_unknown_variant_raises(self):
    with self.assertRaises(ValueError):
      augment_train_split.get_background_color_for_variant(
          "unknown_variant", _ROTATION_FILL_COLOR
      )

  # ── Filename classification ─────────────────────────────────────────────

  def test_is_mask_sidecar_filename(self):
    self.assertTrue(
        augment_train_split.is_mask_sidecar_filename("img_mask.png")
    )
    self.assertTrue(
        augment_train_split.is_mask_sidecar_filename("img_vflip_mask.png")
    )
    self.assertFalse(augment_train_split.is_mask_sidecar_filename("img.jpg"))
    self.assertFalse(
        augment_train_split.is_mask_sidecar_filename("img_vflip.jpg")
    )

  def test_is_augmented_filename_plain_originals_are_not_augmented(self):
    self.assertFalse(augment_train_split.is_augmented_filename("image1.jpg"))
    self.assertFalse(augment_train_split.is_augmented_filename("image1.png"))
    # Originals' mask sidecars are NOT augmented outputs.
    self.assertFalse(
        augment_train_split.is_augmented_filename("image1_mask.png")
    )

  def test_is_augmented_filename_detects_augmented_images(self):
    self.assertTrue(augment_train_split.is_augmented_filename("img_vflip.jpg"))
    self.assertTrue(augment_train_split.is_augmented_filename("img_blur.jpg"))
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_rot45.jpeg")
    )

  def test_is_augmented_filename_detects_augmented_mask_sidecars(self):
    # The augment stage writes '<base>_<aug>_mask.png' for each augmentation.
    # is_augmented_filename must recognise these too so pre-existing checks
    # and mask-sidecar exclusion both work.
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_vflip_mask.png")
    )
    self.assertTrue(
        augment_train_split.is_augmented_filename("img_rot90_mask.png")
    )

  # ── Sidecar path derivation ─────────────────────────────────────────────

  def test_build_mask_sidecar_path(self):
    self.assertEqual(
        augment_train_split.build_mask_sidecar_path(
            "/some/dir/image_001_0.jpg"
        ),
        "/some/dir/image_001_0_mask.png",
    )

  # ── Compositing ─────────────────────────────────────────────────────────

  def test_composite_foreground_on_background_uses_background_outside_mask(
      self,
  ):
    # Every background pixel (mask == 0) must be exactly the requested
    # background color; every foreground pixel (mask > 0) must equal the
    # source image pixel.
    image = PIL.Image.new("RGB", (32, 32), color=(200, 30, 30))
    mask = PIL.Image.new("L", (32, 32), 0)
    PIL.ImageDraw.Draw(mask).rectangle([10, 10, 20, 20], fill=255)

    result = augment_train_split.composite_foreground_on_background(
        image, mask, (124, 116, 104)
    )
    result_array = np.array(result)
    mask_array = np.array(mask)

    background_pixels = result_array[mask_array == 0]
    foreground_pixels = result_array[mask_array > 0]

    self.assertTrue(np.all(background_pixels == np.array((124, 116, 104))))
    self.assertTrue(np.all(foreground_pixels == np.array((200, 30, 30))))

  # ── Geometric augmentations ─────────────────────────────────────────────

  def test_build_geometric_augmentation_flips_change_mask(self):
    # vflip and hflip must actually move the mask geometry.
    _, mask = _make_crop_and_mask((0, 0, 0))
    image = PIL.Image.new("RGB", _CROP_SIZE, color=(200, 0, 0))
    original_mask_array = np.array(mask)

    for aug in ("vflip", "hflip"):
      _, transformed_mask = augment_train_split.build_geometric_augmentation(
          image, mask, aug, (0, 0, 0)
      )
      self.assertFalse(
          np.array_equal(np.array(transformed_mask), original_mask_array),
          msg=f"{aug} did not change mask geometry",
      )

  def test_build_geometric_augmentation_rotations_keep_mask_binary(self):
    # After rotation the mask must still be strictly {0, 255} — nearest-
    # neighbor interpolation must be used, not bilinear.
    _, mask = _make_crop_and_mask((0, 0, 0))
    image = PIL.Image.new("RGB", _CROP_SIZE, color=(200, 0, 0))
    for aug in ("rot45", "rot65", "rot90"):
      _, transformed_mask = augment_train_split.build_geometric_augmentation(
          image, mask, aug, _ROTATION_FILL_COLOR
      )
      unique_values = np.unique(np.array(transformed_mask))
      self.assertTrue(
          set(unique_values.tolist()).issubset({0, 255}),
          msg=f"{aug} left non-binary mask values: {unique_values}",
      )

  def test_build_geometric_augmentation_unknown_name_raises(self):
    _, mask = _make_crop_and_mask((0, 0, 0))
    image = PIL.Image.new("RGB", _CROP_SIZE, color=(200, 0, 0))
    with self.assertRaises(ValueError):
      augment_train_split.build_geometric_augmentation(
          image, mask, "unknown_aug", (0, 0, 0)
      )

  # ── Non-geometric augmentations ─────────────────────────────────────────

  def test_build_non_geometric_augmentation_unknown_name_raises(self):
    image = PIL.Image.new("RGB", _CROP_SIZE, color=(200, 0, 0))
    with self.assertRaises(ValueError):
      augment_train_split.build_non_geometric_augmentation(image, "unknown_aug")

  # ── Single augmentation end-to-end ──────────────────────────────────────

  @parameterized.parameters(
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
  def test_build_single_augmentation_background_is_solid_color(self, aug):
    # For EVERY augmentation, the saved output's background (mask == 0)
    # must be exactly the passed-in background color. This is the core
    # guarantee of the foreground-only pipeline.
    background_color = _ROTATION_FILL_COLOR
    image, mask = _make_crop_and_mask(background_color)

    augmented_image, augmented_mask = (
        augment_train_split.build_single_augmentation_with_mask(
            image, mask, aug, background_color
        )
    )
    image_array = np.array(augmented_image)
    mask_array = np.array(augmented_mask)
    background_pixels = image_array[mask_array == 0]

    if background_pixels.size == 0:
      self.skipTest("Mask covers whole image in this test setup.")

    self.assertTrue(
        np.all(background_pixels == np.array(background_color)),
        msg=(
            f"[{aug}] background pixels not exactly {background_color}; "
            f"sample: {background_pixels[:5].tolist()}"
        ),
    )

  def test_build_single_augmentation_unknown_name_raises(self):
    image, mask = _make_crop_and_mask((0, 0, 0))
    with self.assertRaises(ValueError):
      augment_train_split.build_single_augmentation_with_mask(
          image, mask, "unknown_aug", (0, 0, 0)
      )

  def test_build_augmented_images_returns_all_requested_names(self):
    image, mask = _make_crop_and_mask((0, 0, 0))
    outputs = augment_train_split.build_augmented_images_with_masks(
        image, mask, ("vflip", "hflip"), (0, 0, 0)
    )
    self.assertEqual(set(outputs), {"vflip", "hflip"})
    for out_image, out_mask in outputs.values():
      self.assertIsInstance(out_image, PIL.Image.Image)
      self.assertIsInstance(out_mask, PIL.Image.Image)

  # ── Target-folder discovery ─────────────────────────────────────────────

  def test_discover_target_folders_single_variant(self):
    # Single-variant layout: each class folder is a target, tagged with the
    # single configured variant.
    train_dir = self.create_tempdir().full_path
    class_a = os.path.join(train_dir, "class_a")
    class_b = os.path.join(train_dir, "class_b")
    os.makedirs(class_a)
    os.makedirs(class_b)

    targets = augment_train_split.discover_target_folders(
        train_dir, crop_variants=("raw",)
    )
    self.assertEqual(
        targets,
        [
            ("class_a", class_a, "raw"),
            ("class_b", class_b, "raw"),
        ],
    )

  def test_discover_target_folders_multi_variant(self):
    train_dir = self.create_tempdir().full_path
    class_a = os.path.join(train_dir, "class_a")
    var_raw = os.path.join(class_a, "raw")
    var_mean = os.path.join(class_a, "imagenet_mean_background")
    os.makedirs(var_raw)
    os.makedirs(var_mean)

    targets = augment_train_split.discover_target_folders(
        train_dir, crop_variants=("raw", "imagenet_mean_background")
    )
    self.assertEqual(
        targets,
        [
            ("class_a/raw", var_raw, "raw"),
            (
                "class_a/imagenet_mean_background",
                var_mean,
                "imagenet_mean_background",
            ),
        ],
    )

  def test_discover_target_folders_empty_raises(self):
    train_dir = self.create_tempdir().full_path
    with self.assertRaises(ValueError):
      augment_train_split.discover_target_folders(
          train_dir, crop_variants=("raw",)
      )

  # ── Pre-existing augmentations guard ────────────────────────────────────

  def test_validate_no_pre_existing_augmentations_ok_when_clean(self):
    class_dir = self.create_tempdir().full_path
    PIL.Image.new("RGB", (10, 10)).save(os.path.join(class_dir, "orig.jpg"))
    augment_train_split.validate_no_pre_existing_augmentations(
        [("class_a", class_dir, "raw")]
    )

  def test_validate_no_pre_existing_augmentations_raises_when_dirty(self):
    class_dir = self.create_tempdir().full_path
    PIL.Image.new("RGB", (10, 10)).save(
        os.path.join(class_dir, "orig_vflip.jpg")
    )
    with self.assertRaises(FileExistsError):
      augment_train_split.validate_no_pre_existing_augmentations(
          [("class_a", class_dir, "raw")]
      )

  # ── Original-image listing ──────────────────────────────────────────────

  def test_list_original_image_names_excludes_masks_and_augmented(self):
    folder = self.create_tempdir().full_path
    # Original and its mask sidecar.
    open(os.path.join(folder, "img.jpg"), "w").close()
    open(os.path.join(folder, "img_mask.png"), "w").close()
    # A leftover augmentation from a previous run.
    open(os.path.join(folder, "img_vflip.jpg"), "w").close()
    # An unrelated non-image file.
    open(os.path.join(folder, "readme.txt"), "w").close()

    names = augment_train_split.list_original_image_names(folder)
    self.assertEqual(names, ["img.jpg"])

  # ── Per-folder processing ───────────────────────────────────────────────

  def test_process_target_folder_writes_augmented_pairs(self):
    # Given one original crop + its mask sidecar, process_target_folder
    # must write one augmented .jpg and one augmented _mask.png per
    # configured augmentation.
    folder = self.create_tempdir().full_path
    _write_crop_and_mask_to(folder, "orig", (0, 0, 0))

    augment_train_split.process_target_folder(
        target_label="class_a",
        target_path=folder,
        variant_name="raw",
        augmentations_to_apply=("vflip", "hflip"),
        rotation_fill_color=_ROTATION_FILL_COLOR,
    )
    files = set(os.listdir(folder))
    # Original + its mask are preserved; two augmented pairs are added.
    self.assertIn("orig.jpg", files)
    self.assertIn("orig_mask.png", files)
    self.assertIn("orig_vflip.jpg", files)
    self.assertIn("orig_vflip_mask.png", files)
    self.assertIn("orig_hflip.jpg", files)
    self.assertIn("orig_hflip_mask.png", files)

  def test_process_target_folder_missing_mask_raises_after_pass(self):
    # If a mask sidecar is missing for one image, process_target_folder
    # must raise FileNotFoundError so the caller stops. Nothing about a
    # missing mask should be silent.
    folder = self.create_tempdir().full_path
    # Original with mask.
    _write_crop_and_mask_to(folder, "have_mask", (0, 0, 0))
    # Original without mask.
    PIL.Image.new("RGB", _CROP_SIZE, color=(0, 0, 0)).save(
        os.path.join(folder, "no_mask.jpg"), quality=95
    )

    with self.assertRaises(FileNotFoundError):
      augment_train_split.process_target_folder(
          target_label="class_a",
          target_path=folder,
          variant_name="raw",
          augmentations_to_apply=("vflip",),
          rotation_fill_color=_ROTATION_FILL_COLOR,
      )

  # ── Cleanup helpers ─────────────────────────────────────────────────────

  def test_delete_mask_sidecars_under_removes_only_masks(self):
    root = self.create_tempdir().full_path
    class_a = os.path.join(root, "class_a")
    class_b = os.path.join(root, "class_b")
    for folder in (class_a, class_b):
      os.makedirs(folder)
      # Two masks and one non-mask per folder.
      open(os.path.join(folder, "img_mask.png"), "w").close()
      open(os.path.join(folder, "img_vflip_mask.png"), "w").close()
      open(os.path.join(folder, "img.jpg"), "w").close()

    deleted_count, errors = augment_train_split.delete_mask_sidecars_under(root)
    self.assertEqual(deleted_count, 4)
    self.assertEqual(errors, [])
    # Non-mask files must be untouched.
    for folder in (class_a, class_b):
      self.assertTrue(os.path.exists(os.path.join(folder, "img.jpg")))
      self.assertFalse(os.path.exists(os.path.join(folder, "img_mask.png")))

  def test_cleanup_mask_sidecars_in_directory_missing_dir_is_silent(self):
    # If the directory does not exist, cleanup must not raise; it just
    # prints a skip message. This keeps the try/finally in main() safe.
    augment_train_split.cleanup_mask_sidecars_in_directory(
        "/definitely/does/not/exist", "train split"
    )

  # ── End-to-end via main() ───────────────────────────────────────────────

  def _build_classifier_dataset(self, root, class_names):
    """Builds a minimal classifier dataset with mask sidecars in train."""
    classifier_dir = os.path.join(root, "classifier")
    for class_name in class_names:
      train_class = os.path.join(classifier_dir, "train", class_name)
      val_class = os.path.join(classifier_dir, "val", class_name)
      _write_crop_and_mask_to(train_class, "img_000_0", (0, 0, 0))
      # Val has NO mask sidecars, matching production.
      os.makedirs(val_class, exist_ok=True)
      PIL.Image.new("RGB", _CROP_SIZE, color=(0, 0, 0)).save(
          os.path.join(val_class, "img_000_0.jpg"), quality=95
      )
    return classifier_dir

  @mock.patch.object(augment_train_split.config_loader, "load_config")
  def test_main_writes_augmentations_and_cleans_up_masks(
      self, mock_load_config
  ):
    root = self.create_tempdir().full_path
    classifier_dir = self._build_classifier_dataset(root, ["class_a"])

    mock_config = mock.Mock()
    mock_config.classifier_dir = classifier_dir
    mock_config.train_split_name = "train"
    mock_config.val_split_name = "val"
    mock_config.crop_variants = ("raw",)
    mock_config.rotation_fill_color = _ROTATION_FILL_COLOR
    mock_config.active_augmentations = ("vflip",)
    mock_config.prompt_to_detect = "packets"
    mock_load_config.return_value = mock_config

    augment_train_split.main(config_path="/dummy/config.yaml")

    train_class = os.path.join(classifier_dir, "train", "class_a")
    val_class = os.path.join(classifier_dir, "val", "class_a")

    train_files = set(os.listdir(train_class))
    val_files = set(os.listdir(val_class))

    # Augmentation was written.
    self.assertIn("img_000_0_vflip.jpg", train_files)
    # No mask sidecars remain under train.
    self.assertFalse(any(name.endswith("_mask.png") for name in train_files))
    # Val was untouched: originals only, no augmented copies.
    self.assertEqual(val_files, {"img_000_0.jpg"})

  @mock.patch.object(augment_train_split.config_loader, "load_config")
  def test_main_cleans_up_masks_even_on_failure(self, mock_load_config):
    # Cleanup lives in the finally block, so it must run even when the
    # augmentation loop itself raises. We trigger a failure by removing the
    # mask sidecar for one image so process_target_folder raises.
    root = self.create_tempdir().full_path
    classifier_dir = self._build_classifier_dataset(root, ["class_a"])

    train_class = os.path.join(classifier_dir, "train", "class_a")
    # Add a second image WITHOUT a mask sidecar to force a failure.
    PIL.Image.new("RGB", _CROP_SIZE, color=(0, 0, 0)).save(
        os.path.join(train_class, "no_mask.jpg"), quality=95
    )

    mock_config = mock.Mock()
    mock_config.classifier_dir = classifier_dir
    mock_config.train_split_name = "train"
    mock_config.val_split_name = "val"
    mock_config.crop_variants = ("raw",)
    mock_config.rotation_fill_color = _ROTATION_FILL_COLOR
    mock_config.active_augmentations = ("vflip",)
    mock_config.prompt_to_detect = "packets"
    mock_load_config.return_value = mock_config

    with self.assertRaises(FileNotFoundError):
      augment_train_split.main(config_path="/dummy/config.yaml")

    # Even though main raised, no mask sidecars must remain under train.
    train_files = set(os.listdir(train_class))
    self.assertFalse(any(name.endswith("_mask.png") for name in train_files))


if __name__ == "__main__":
  absltest.main()
