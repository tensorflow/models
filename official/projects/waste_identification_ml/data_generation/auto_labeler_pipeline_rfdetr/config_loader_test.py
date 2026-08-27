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

"""Unit tests for config_loader.py."""

import dataclasses
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import yaml

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import config_loader


def _valid_config_mapping() -> dict[str, Any]:
  """Returns a fully valid config mapping usable as a test baseline.

  Individual tests copy this and mutate a single field so each test isolates
  exactly one validation rule.
  """
  return {
      "root_dir": "/data/run",
      "rfdetr_checkpoint_path": "/models/ckpt.pth",
      "cuda_visible_devices": "0",
      "input_images_folder_name": "images",
      "train_val_folder_name": "train_val_images",
      "train_split_name": "train",
      "val_split_name": "val",
      "keep_every_nth": 3,
      "train_ratio": 0.15,
      "min_detections": 2,
      "crop_variants": ["raw"],
      "max_cpu_workers": 16,
      "queue_maxsize": 32,
      "rotation_fill_color": [124, 116, 104],
      "predict_threshold": 0.3,
      "score_threshold": 0.0,
      "containment_threshold": 0.98,
      "merge_containment_threshold": 0.7,
      "max_short_side": 1024,
      "crop_size": [256, 256],
      "augmentations": ["vflip", "hflip", "blur"],
  }


class DeriveSiblingDirTest(parameterized.TestCase):
  """Tests for _derive_sibling_dir."""

  @parameterized.named_parameters(
      ("no_trailing_sep", "/data/run", "_classifier", "/data/run_classifier"),
      ("trailing_sep", "/data/run/", "_empty", "/data/run_empty"),
      ("single_component", "/run", "_empty", "/run_empty"),
  )
  def test_appends_suffix_to_final_component(self, root, suffix, expected):
    """Verifies the suffix is appended to the final path component."""
    self.assertEqual(
        config_loader._derive_sibling_dir(root, suffix), expected
    )


class RequireNumberInRangeTest(absltest.TestCase):
  """Tests for _require_number_in_range."""

  def test_accepts_value_in_range(self):
    """Verifies an in-range value is returned as a float."""
    result = config_loader._require_number_in_range(0.5, "f", 0.0, 1.0)
    self.assertEqual(result, 0.5)
    self.assertIsInstance(result, float)

  def test_accepts_bounds_inclusively(self):
    """Verifies the range check is inclusive of both endpoints."""
    self.assertEqual(
        config_loader._require_number_in_range(0.0, "f", 0.0, 1.0), 0.0
    )
    self.assertEqual(
        config_loader._require_number_in_range(1.0, "f", 0.0, 1.0), 1.0
    )

  def test_rejects_bool(self):
    """Verifies booleans are rejected even though bool subclasses int."""
    with self.assertRaisesRegex(config_loader.ConfigError, "must be a number"):
      config_loader._require_number_in_range(True, "f", 0.0, 1.0)

  def test_rejects_out_of_range(self):
    """Verifies a value outside the range raises."""
    with self.assertRaisesRegex(config_loader.ConfigError, "between"):
      config_loader._require_number_in_range(1.5, "f", 0.0, 1.0)

  def test_rejects_non_number(self):
    """Verifies a non-numeric value raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_number_in_range("0.5", "f", 0.0, 1.0)


class RequirePositiveIntTest(absltest.TestCase):
  """Tests for _require_positive_int."""

  def test_accepts_positive_int(self):
    """Verifies a positive integer is returned unchanged."""
    self.assertEqual(config_loader._require_positive_int(5, "f"), 5)

  def test_rejects_zero(self):
    """Verifies zero is rejected (must be at least 1)."""
    with self.assertRaisesRegex(config_loader.ConfigError, "at least 1"):
      config_loader._require_positive_int(0, "f")

  def test_rejects_bool(self):
    """Verifies booleans are rejected."""
    with self.assertRaisesRegex(
        config_loader.ConfigError, "must be an integer"
    ):
      config_loader._require_positive_int(True, "f")

  def test_rejects_float(self):
    """Verifies a float is rejected even when integral in value."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_positive_int(3.0, "f")


class RequireNonEmptyStringTest(absltest.TestCase):
  """Tests for _require_non_empty_string."""

  def test_accepts_non_empty(self):
    """Verifies a non-empty string is returned unchanged."""
    self.assertEqual(config_loader._require_non_empty_string("abc", "f"), "abc")

  def test_rejects_empty(self):
    """Verifies an empty string raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_non_empty_string("", "f")

  def test_rejects_whitespace_only(self):
    """Verifies a whitespace-only string raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_non_empty_string("   ", "f")

  def test_rejects_non_string(self):
    """Verifies a non-string value raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_non_empty_string(123, "f")


class RequireAllowedFolderNameTest(absltest.TestCase):
  """Tests for _require_allowed_folder_name."""

  def test_accepts_allowed_value(self):
    """Verifies the single allowed value passes."""
    self.assertEqual(
        config_loader._require_allowed_folder_name(
            "images", "input_images_folder_name"
        ),
        "images",
    )

  def test_rejects_other_value(self):
    """Verifies any other value raises with the allowed set."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._require_allowed_folder_name(
          "imgs", "input_images_folder_name"
      )


class ValidateCropSizeTest(absltest.TestCase):
  """Tests for _validate_crop_size."""

  def test_accepts_two_positive_ints(self):
    """Verifies a valid two-element size returns a tuple of ints."""
    self.assertEqual(config_loader._validate_crop_size([256, 128]), (256, 128))

  def test_rejects_wrong_length(self):
    """Verifies a size without exactly two elements raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_crop_size([256])

  def test_rejects_non_positive(self):
    """Verifies a non-positive dimension raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_crop_size([256, 0])

  def test_rejects_non_sequence(self):
    """Verifies a scalar value raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_crop_size(256)


class ValidateCropVariantsTest(absltest.TestCase):
  """Tests for _validate_crop_variants."""

  def test_reorders_to_canonical(self):
    """Verifies variants are reordered to ALLOWED_CROP_VARIANTS order."""
    result = config_loader._validate_crop_variants(
        ["imagenet_mean_background", "raw"]
    )
    self.assertEqual(result, ("raw", "imagenet_mean_background"))

  def test_rejects_empty(self):
    """Verifies an empty list raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_crop_variants([])

  def test_rejects_unknown_variant(self):
    """Verifies an unknown variant name raises."""
    with self.assertRaisesRegex(config_loader.ConfigError, "Unknown crop"):
      config_loader._validate_crop_variants(["raw", "purple_background"])

  def test_rejects_duplicate(self):
    """Verifies a duplicated variant raises."""
    with self.assertRaisesRegex(config_loader.ConfigError, "Duplicate"):
      config_loader._validate_crop_variants(["raw", "raw"])


class ValidateRotationFillColorTest(absltest.TestCase):
  """Tests for _validate_rotation_fill_color."""

  def test_accepts_valid_rgb(self):
    """Verifies three in-range integers return an RGB tuple."""
    self.assertEqual(
        config_loader._validate_rotation_fill_color([124, 116, 104]),
        (124, 116, 104),
    )

  def test_accepts_bounds(self):
    """Verifies 0 and 255 are accepted at the channel bounds."""
    self.assertEqual(
        config_loader._validate_rotation_fill_color([0, 255, 0]),
        (0, 255, 0),
    )

  def test_rejects_wrong_length(self):
    """Verifies a color without exactly three channels raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_rotation_fill_color([0, 0])

  def test_rejects_out_of_range_channel(self):
    """Verifies a channel above 255 raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_rotation_fill_color([0, 0, 256])

  def test_rejects_bool_channel(self):
    """Verifies a boolean channel value raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_rotation_fill_color([True, 0, 0])


class ValidateAugmentationsTest(absltest.TestCase):
  """Tests for _validate_augmentations."""

  def test_reorders_to_canonical(self):
    """Verifies augmentations are reordered to canonical order."""
    result = config_loader._validate_augmentations(["blur", "vflip", "rot90"])
    self.assertEqual(result, ("vflip", "rot90", "blur"))

  def test_rejects_empty(self):
    """Verifies an empty augmentation list raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_augmentations([])

  def test_rejects_unknown(self):
    """Verifies an unknown augmentation name raises."""
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Unknown augmentation"
    ):
      config_loader._validate_augmentations(["vflip", "sepia"])

  def test_rejects_duplicate(self):
    """Verifies a duplicated augmentation raises."""
    with self.assertRaisesRegex(config_loader.ConfigError, "Duplicate"):
      config_loader._validate_augmentations(["vflip", "vflip"])


class ValidateCudaVisibleDevicesTest(parameterized.TestCase):
  """Tests for _validate_cuda_visible_devices."""

  def test_accepts_string(self):
    """Verifies a string value passes through unchanged."""
    self.assertEqual(
        config_loader._validate_cuda_visible_devices("0"), "0"
    )

  def test_coerces_int_to_string(self):
    """Verifies an integer is coerced to its string form."""
    self.assertEqual(config_loader._validate_cuda_visible_devices(1), "1")

  def test_rejects_bool(self):
    """Verifies a boolean raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_cuda_visible_devices(True)

  def test_rejects_float(self):
    """Verifies a float raises."""
    with self.assertRaises(config_loader.ConfigError):
      config_loader._validate_cuda_visible_devices(0.5)


class LoadConfigTest(absltest.TestCase):
  """Tests for the public load_config entry point."""

  def _write_config(self, mapping: dict[str, Any]) -> str:
    """Writes a mapping to a temp YAML file and returns its path."""
    config_path = pathlib.Path(self.create_tempdir().full_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(config_path)

  def test_loads_valid_config(self):
    """Verifies a valid config produces a fully populated PipelineConfig."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.load_config(config_path)
    self.assertIsInstance(config, config_loader.PipelineConfig)
    self.assertEqual(config.root_dir, "/data/run")
    self.assertEqual(config.keep_every_nth, 3)
    self.assertEqual(config.crop_variants, ("raw",))

  def test_derives_sibling_directories(self):
    """Verifies classifier_dir and rejected_dir are derived from root_dir."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.load_config(config_path)
    self.assertEqual(config.classifier_dir, "/data/run_classifier")
    self.assertEqual(config.rejected_dir, "/data/run_empty")

  def test_augmentations_reordered_to_canonical(self):
    """Verifies the loaded augmentations follow canonical order."""
    mapping = _valid_config_mapping()
    mapping["augmentations"] = ["blur", "vflip"]
    config_path = self._write_config(mapping)
    config = config_loader.load_config(config_path)
    self.assertEqual(config.augmentations, ("vflip", "blur"))

  def test_returned_config_is_frozen(self):
    """Verifies PipelineConfig is immutable (frozen dataclass)."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.load_config(config_path)
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(config, "root_dir", "/other")

  def test_raises_when_file_missing(self):
    """Verifies a missing config file raises ConfigError."""
    with self.assertRaisesRegex(config_loader.ConfigError, "does not exist"):
      config_loader.load_config("/nonexistent/config.yaml")

  def test_raises_on_invalid_yaml(self):
    """Verifies malformed YAML raises ConfigError."""
    config_path = pathlib.Path(self.create_tempdir().full_path) / "bad.yaml"
    config_path.write_text("root_dir: [unclosed", encoding="utf-8")
    with self.assertRaisesRegex(config_loader.ConfigError, "not valid YAML"):
      config_loader.load_config(str(config_path))

  def test_raises_when_top_level_not_mapping(self):
    """Verifies a non-mapping top-level document raises ConfigError."""
    config_path = pathlib.Path(self.create_tempdir().full_path) / "list.yaml"
    config_path.write_text("- a\n- b\n", encoding="utf-8")
    with self.assertRaisesRegex(config_loader.ConfigError, "top-level mapping"):
      config_loader.load_config(str(config_path))

  def test_raises_on_missing_required_field(self):
    """Verifies a missing required key raises naming the field."""
    mapping = _valid_config_mapping()
    del mapping["train_ratio"]
    config_path = self._write_config(mapping)
    with self.assertRaisesRegex(config_loader.ConfigError, "train_ratio"):
      config_loader.load_config(config_path)

  def test_raises_on_out_of_range_threshold(self):
    """Verifies an out-of-range threshold raises ConfigError."""
    mapping = _valid_config_mapping()
    mapping["predict_threshold"] = 1.5
    config_path = self._write_config(mapping)
    with self.assertRaises(config_loader.ConfigError):
      config_loader.load_config(config_path)

  def test_loads_actual_config_file(self):
    """Verifies that the workspace config.yaml loads with expected values."""
    config_path = pathlib.Path(__file__).parent / "config.yaml"
    config = config_loader.load_config(str(config_path))
    self.assertIsInstance(config, config_loader.PipelineConfig)
    self.assertEqual(
        config.root_dir,
        "/home/umairsabir/new_data/test_data/saahas_milk_packet/exp",
    )
    self.assertEqual(
        config.classifier_dir,
        "/home/umairsabir/new_data/test_data/saahas_milk_packet/exp_classifier",
    )
    self.assertEqual(
        config.rejected_dir,
        "/home/umairsabir/new_data/test_data/saahas_milk_packet/exp_empty",
    )
    self.assertEqual(config.keep_every_nth, 3)
    self.assertAlmostEqual(config.train_ratio, 0.15)
    self.assertEqual(config.min_detections, 2)
    self.assertEqual(config.crop_variants, ("raw",))
    self.assertEqual(config.predict_threshold, 0.3)
    self.assertEqual(config.score_threshold, 0.0)
    self.assertEqual(config.containment_threshold, 0.98)
    self.assertEqual(config.merge_containment_threshold, 0.7)
    self.assertEqual(config.max_short_side, 1024)
    self.assertEqual(config.crop_size, (256, 256))
    self.assertIn("vflip", config.augmentations)


if __name__ == "__main__":
  absltest.main()
