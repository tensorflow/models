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

"""Unit tests for segmentation.py."""

import os
import pathlib
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import torch

# Mock supervision before importing segmentation
sys.modules.setdefault("supervision", mock.MagicMock())

# pylint: disable=g-import-not-at-top
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import segmentation
# pylint: enable=g-import-not-at-top


def _rectangle_mask(height: int, width: int) -> np.ndarray:
  """Returns a bool mask with a filled central rectangle."""
  mask = np.zeros((height, width), dtype=bool)
  mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
  return mask


def _make_state(
    num_detections: int, height: int, width: int
) -> dict[str, Any]:
  """Builds a minimal state dict with rectangular masks and boxes."""
  masks = torch.zeros((num_detections, 1, height, width), dtype=torch.bool)
  boxes = torch.zeros((num_detections, 4), dtype=torch.float32)
  for index in range(num_detections):
    masks[index, 0] = torch.from_numpy(_rectangle_mask(height, width))
    boxes[index] = torch.tensor([0, 0, width, height], dtype=torch.float32)
  scores = torch.linspace(0.5, 0.95, steps=num_detections)
  return {"masks": masks, "boxes": boxes, "scores": scores}


class BuildVariantDirectoriesTest(absltest.TestCase):
  """Tests for build_variant_directories."""

  def test_single_variant_uses_class_folder_directly(self):
    """Verifies one variant maps to the class folder itself (flat layout)."""
    class_folder = pathlib.Path(self.create_tempdir().full_path) / "class_a"
    result = segmentation.build_variant_directories(str(class_folder), ("raw",))
    self.assertEqual(result, {"raw": str(class_folder)})
    self.assertTrue(class_folder.is_dir())

  def test_multiple_variants_get_subdirectories(self):
    """Verifies each variant gets its own subdirectory under the class folder."""
    class_folder = pathlib.Path(self.create_tempdir().full_path) / "class_a"
    result = segmentation.build_variant_directories(
        str(class_folder), ("raw", "black_background")
    )
    self.assertEqual(
        result,
        {
            "raw": str(class_folder / "raw"),
            "black_background": str(class_folder / "black_background"),
        },
    )
    self.assertTrue((class_folder / "raw").is_dir())
    self.assertTrue((class_folder / "black_background").is_dir())


class GetBackgroundColorForVariantTest(parameterized.TestCase):
  """Tests for get_background_color_for_variant."""

  @parameterized.named_parameters(
      ("raw", "raw", (0, 0, 0)),
      ("black", "black_background", (0, 0, 0)),
      ("imagenet_mean", "imagenet_mean_background", (124, 116, 104)),
  )
  def test_returns_expected_color(self, variant, expected):
    """Verifies each variant resolves to its background color."""
    self.assertEqual(
        segmentation.get_background_color_for_variant(variant, (124, 116, 104)),
        expected,
    )

  def test_unknown_variant_raises(self):
    """Verifies an unknown variant raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Unknown crop variant"):
      segmentation.get_background_color_for_variant("bogus", (0, 0, 0))


class BuildVariantCropTest(absltest.TestCase):
  """Tests for build_variant_crop."""

  def setUp(self):
    super().setUp()
    self.image_array = np.full((50, 50, 3), 120, dtype=np.uint8)
    self.mask = _rectangle_mask(50, 50)
    self.box = [0, 0, 50, 50]

  def test_raw_variant_returns_pil_image(self):
    """Verifies the raw variant returns a PIL image for a valid box."""
    result = segmentation.build_variant_crop(
        self.image_array, self.mask, self.box, (64, 64), "raw", (0, 0, 0)
    )
    self.assertIsInstance(result, Image.Image)

  def test_letterboxed_variant_matches_crop_size(self):
    """Verifies a letterboxed variant returns an image of the crop size."""
    result = segmentation.build_variant_crop(
        self.image_array,
        self.mask,
        self.box,
        (64, 64),
        "black_background",
        (0, 0, 0),
    )
    self.assertEqual(result.size, (64, 64))

  def test_unknown_variant_raises(self):
    """Verifies an unknown variant raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Unknown crop variant"):
      segmentation.build_variant_crop(
          self.image_array, self.mask, self.box, (64, 64), "bogus", (0, 0, 0)
      )


class BuildVariantMaskTest(absltest.TestCase):
  """Tests for build_variant_mask."""

  def setUp(self):
    super().setUp()
    self.mask = _rectangle_mask(50, 50)
    self.box = [0, 0, 50, 50]

  def test_letterboxed_variant_matches_crop_size(self):
    """Verifies a letterboxed variant mask has the crop-size shape."""
    result = segmentation.build_variant_mask(
        self.mask, self.box, (64, 64), "black_background"
    )
    self.assertEqual(result.shape, (64, 64))
    self.assertEqual(result.dtype, np.uint8)

  def test_raw_variant_returns_box_shaped_mask(self):
    """Verifies the raw variant mask matches the box crop, not the canvas."""
    result = segmentation.build_variant_mask(
        self.mask, self.box, (64, 64), "raw"
    )
    self.assertEqual(result.shape, (50, 50))

  def test_unknown_variant_raises(self):
    """Verifies an unknown variant raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Unknown crop variant"):
      segmentation.build_variant_mask(self.mask, self.box, (64, 64), "bogus")


class GenerateSelectedCropsTest(absltest.TestCase):
  """Tests for generate_selected_crops."""

  def test_filters_below_score_threshold(self):
    """Verifies detections below the score threshold are skipped."""
    image = Image.new("RGB", (50, 50), (100, 100, 100))
    state = _make_state(num_detections=2, height=50, width=50)
    # scores are [0.5, 0.95]; threshold 0.9 keeps only the second.
    records = segmentation.generate_selected_crops(
        image,
        state,
        score_threshold=0.9,
        crop_size=(64, 64),
        variants=("raw",),
        rotation_fill_color=(0, 0, 0),
        build_masks=True,
    )
    self.assertLen(records, 1)
    detection_index, _, _ = records[0]
    self.assertEqual(detection_index, 1)

  def test_build_masks_false_yields_none_masks(self):
    """Verifies mask entries are None when build_masks is False."""
    image = Image.new("RGB", (50, 50), (100, 100, 100))
    state = _make_state(num_detections=1, height=50, width=50)
    records = segmentation.generate_selected_crops(
        image,
        state,
        score_threshold=0.0,
        crop_size=(64, 64),
        variants=("raw",),
        rotation_fill_color=(0, 0, 0),
        build_masks=False,
    )
    _, _, variant_to_mask = records[0]
    self.assertIsNone(variant_to_mask["raw"])

  def test_produces_crop_per_variant(self):
    """Verifies every requested variant appears in the crop mapping."""
    image = Image.new("RGB", (50, 50), (100, 100, 100))
    state = _make_state(num_detections=1, height=50, width=50)
    records = segmentation.generate_selected_crops(
        image,
        state,
        score_threshold=0.0,
        crop_size=(64, 64),
        variants=("raw", "black_background"),
        rotation_fill_color=(0, 0, 0),
        build_masks=True,
    )
    _, variant_to_crop, _ = records[0]
    self.assertCountEqual(variant_to_crop.keys(), ["raw", "black_background"])


class SaveCropImageTest(absltest.TestCase):
  """Tests for save_crop_image."""

  def test_writes_jpeg(self):
    """Verifies a crop is written to disk and reloads as an image."""
    output_dir = pathlib.Path(self.create_tempdir().full_path)
    crop = Image.new("RGB", (16, 16), (10, 20, 30))
    output_path = output_dir / "crop.jpg"
    segmentation.save_crop_image(crop, str(output_path))
    self.assertTrue(output_path.exists())
    with Image.open(output_path) as reloaded:
      self.assertEqual(reloaded.size, (16, 16))


class SaveMaskSidecarTest(absltest.TestCase):
  """Tests for save_mask_sidecar."""

  def test_writes_single_channel_png(self):
    """Verifies a mask is written as a single-channel PNG."""
    output_dir = pathlib.Path(self.create_tempdir().full_path)
    mask = np.ones((16, 16), dtype=np.uint8) * 255
    output_path = output_dir / "crop_mask.png"
    segmentation.save_mask_sidecar(mask, str(output_path))
    self.assertTrue(output_path.exists())
    with Image.open(output_path) as reloaded:
      self.assertEqual(reloaded.mode, "L")
      self.assertEqual(reloaded.size, (16, 16))


class SaveOneDetectionTest(absltest.TestCase):
  """Tests for save_one_detection."""

  def test_writes_crops_and_masks_for_each_variant(self):
    """Verifies crops and mask sidecars are written per variant."""
    root = pathlib.Path(self.create_tempdir().full_path)
    raw_dir = root / "raw"
    black_dir = root / "black_background"
    raw_dir.mkdir()
    black_dir.mkdir()

    crop = Image.new("RGB", (16, 16), (0, 0, 0))
    mask = np.ones((16, 16), dtype=np.uint8) * 255
    variant_to_crop = {"raw": crop, "black_background": crop}
    variant_to_mask = {"raw": mask, "black_background": mask}
    variant_directories = {
        "raw": str(raw_dir),
        "black_background": str(black_dir),
    }

    segmentation.save_one_detection(
        detection_index=0,
        variant_to_crop=variant_to_crop,
        variant_to_mask=variant_to_mask,
        filename="img_001",
        variant_directories=variant_directories,
        write_masks=True,
    )
    self.assertTrue((raw_dir / "img_001_0.jpg").exists())
    self.assertTrue((raw_dir / "img_001_0_mask.png").exists())
    self.assertTrue((black_dir / "img_001_0.jpg").exists())

  def test_skips_none_crops(self):
    """Verifies a None crop (degenerate box) writes nothing for that variant."""
    root = pathlib.Path(self.create_tempdir().full_path)
    raw_dir = root / "raw"
    raw_dir.mkdir()

    variant_to_crop = {"raw": None}
    variant_to_mask = {"raw": None}
    variant_directories = {"raw": str(raw_dir)}

    segmentation.save_one_detection(
        detection_index=0,
        variant_to_crop=variant_to_crop,
        variant_to_mask=variant_to_mask,
        filename="img_001",
        variant_directories=variant_directories,
        write_masks=True,
    )
    self.assertEmpty(os.listdir(raw_dir))

  def test_write_masks_false_skips_mask(self):
    """Verifies no mask sidecar is written when write_masks is False."""
    root = pathlib.Path(self.create_tempdir().full_path)
    raw_dir = root / "raw"
    raw_dir.mkdir()

    crop = Image.new("RGB", (16, 16), (0, 0, 0))
    variant_to_crop = {"raw": crop}
    variant_to_mask = {"raw": np.ones((16, 16), dtype=np.uint8) * 255}
    variant_directories = {"raw": str(raw_dir)}

    segmentation.save_one_detection(
        detection_index=0,
        variant_to_crop=variant_to_crop,
        variant_to_mask=variant_to_mask,
        filename="img_001",
        variant_directories=variant_directories,
        write_masks=False,
    )
    written = os.listdir(raw_dir)
    self.assertIn("img_001_0.jpg", written)
    self.assertNotIn("img_001_0_mask.png", written)


class FormatElapsedTimeTest(parameterized.TestCase):
  """Tests for format_elapsed_time."""

  @parameterized.named_parameters(
      ("seconds", 45, "0h 0m 45s"),
      ("minutes", 130, "0h 2m 10s"),
      ("hours", 3661, "1h 1m 1s"),
  )
  def test_formats_elapsed_seconds(self, seconds, expected):
    """Verifies elapsed seconds render as 'Hh Mm Ss'."""
    self.assertEqual(segmentation.format_elapsed_time(seconds), expected)


class BuildRfdetrModelTest(absltest.TestCase):
  """Tests for build_rfdetr_model.

  Note: unlike filter_sparse_images, segmentation.py does NOT call
  optimize_for_inference (that line is commented out), so this test only
  verifies construction.
  """

  def test_raises_when_rfdetr_unavailable(self):
    """Verifies a missing rfdetr package surfaces as ImportError."""
    with mock.patch.object(segmentation, "RFDETRSegMedium", None):
      with self.assertRaises(ImportError):
        segmentation.build_rfdetr_model("/tmp/checkpoint.pth")

  def test_builds_model_without_optimize(self):
    """Verifies the model is constructed from the checkpoint weights."""
    fake_model = mock.Mock()
    fake_class = mock.Mock(return_value=fake_model)
    with mock.patch.object(segmentation, "RFDETRSegMedium", fake_class):
      result = segmentation.build_rfdetr_model("/tmp/ckpt.pth")
    fake_class.assert_called_once_with(pretrain_weights="/tmp/ckpt.pth")
    fake_model.optimize_for_inference.assert_not_called()
    self.assertIs(result, fake_model)


class ValidateClassifierOutputDirTest(absltest.TestCase):
  """Tests for validate_classifier_output_dir."""

  def test_passes_when_absent(self):
    """Verifies a non-existent output dir does not raise."""
    root = pathlib.Path(self.create_tempdir().full_path)
    segmentation.validate_classifier_output_dir(str(root / "classifier"))

  def test_raises_when_present(self):
    """Verifies an existing output dir raises FileExistsError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    classifier = root / "classifier"
    classifier.mkdir()
    with self.assertRaises(FileExistsError):
      segmentation.validate_classifier_output_dir(str(classifier))


if __name__ == "__main__":
  absltest.main()
