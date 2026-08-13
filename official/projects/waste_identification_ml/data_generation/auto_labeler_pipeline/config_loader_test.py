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

"""Unit tests for auto_labeler_pipeline config_loader."""

import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import yaml

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader


class ConfigLoaderTest(parameterized.TestCase):
  """Tests loading, validation, and typed property access for pipeline config."""

  def setUp(self):
    super().setUp()
    self.valid_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )

  def _create_temp_yaml(self, config_dict: Any) -> str:
    """Helper to dump a config dictionary to a temporary YAML file."""
    content = yaml.dump(config_dict)
    temp_file = self.create_tempfile(content=content)
    return temp_file.full_path

  def _get_valid_config_dict(self) -> dict[str, Any]:
    """Returns a valid dictionary representation of pipeline configuration."""
    return {
        "root_dir": "/data/datasets/milk_packet/2026-06-23",
        "sam3_checkpoint_path": "/models/sam3.pt",
        "cuda_visible_devices": "0",
        "prompt_to_detect": "packets",
        "keep_every_nth": 6,
        "train_ratio": 0.10,
        "min_detections": 2,
        "crop_variants": ["imagenet_mean_background"],
        "prompts": {
            "bottles and containers": {
                "detection": {
                    "confidence_threshold": 0.5,
                    "score_threshold": 0.20,
                    "containment_threshold": 0.98,
                    "max_short_side": 1024,
                    "crop_size": [256, 256],
                },
                "augmentations": [
                    "vflip",
                    "hflip",
                    "rot45",
                    "rot65",
                    "rot90",
                    "blur",
                ],
            },
            "packets": {
                "detection": {
                    "confidence_threshold": 0.3,
                    "score_threshold": 0.0,
                    "containment_threshold": 0.98,
                    "max_short_side": 1024,
                    "crop_size": [256, 256],
                },
                "augmentations": [
                    "vflip",
                    "hflip",
                    "rot45",
                    "rot65",
                    "rot90",
                    "blur",
                    "noise03",
                    "noise06",
                    "cjitter",
                ],
            },
        },
    }

  def test_load_config_actual_file(self):
    """Verifies that the workspace config.yaml loads with expected values."""
    config = config_loader.load_config(self.valid_config_path)
    self.assertIsInstance(config, config_loader.PipelineConfig)
    self.assertEqual(
        config.root_dir, "/home/umairsabir/test_set/milk_packet/2026-06-23"
    )
    self.assertEqual(
        config.classifier_dir,
        "/home/umairsabir/test_set/milk_packet/2026-06-23_classifier",
    )
    self.assertEqual(
        config.rejected_dir,
        "/home/umairsabir/test_set/milk_packet/2026-06-23_empty",
    )
    self.assertEqual(config.prompt_to_detect, "packets")
    self.assertEqual(config.keep_every_nth, 6)
    self.assertAlmostEqual(config.train_ratio, 0.10)
    self.assertEqual(config.min_detections, 2)
    self.assertEqual(config.crop_variants, ("imagenet_mean_background",))

    # Active properties
    self.assertEqual(config.active_detection.confidence_threshold, 0.3)
    self.assertEqual(config.active_detection.score_threshold, 0.0)
    self.assertEqual(config.active_detection.crop_size, (256, 256))
    self.assertEqual(config.active_prompt, config.prompts["packets"])
    self.assertIn("vflip", config.active_augmentations)

  def test_load_config_file_not_found(self):
    """Ensures ConfigError is raised when file does not exist."""
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Config file does not exist"
    ):
      config_loader.load_config("/non_existent/config.yaml")

  def test_load_config_invalid_yaml(self):
    """Ensures ConfigError is raised when YAML syntax is malformed."""
    temp_file = self.create_tempfile(content="root_dir: [unclosed_list\n")
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Config file is not valid YAML"
    ):
      config_loader.load_config(temp_file.full_path)

  def test_load_config_not_a_mapping(self):
    """Ensures ConfigError is raised when root is not a mapping."""
    temp_file = self.create_tempfile(content="- item1\n- item2\n")
    with self.assertRaisesRegex(config_loader.ConfigError, "top-level mapping"):
      config_loader.load_config(temp_file.full_path)

  def test_load_config_missing_top_level_key(self):
    """Ensures ConfigError is raised when required top-level key is missing."""
    cfg_dict = self._get_valid_config_dict()
    del cfg_dict["root_dir"]
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "missing required field.*root_dir"
    ):
      config_loader.load_config(path)

  def test_load_config_empty_root_dir(self):
    """Ensures ConfigError is raised when root_dir is empty or whitespace."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["root_dir"] = "   "
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "root_dir must be a non-empty string"
    ):
      config_loader.load_config(path)

  def test_load_config_cuda_visible_devices_int_and_str(self):
    """Verifies cuda_visible_devices accepts both integer and string in YAML."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["cuda_visible_devices"] = 0
    path = self._create_temp_yaml(cfg_dict)
    loaded = config_loader.load_config(path)
    self.assertEqual(loaded.cuda_visible_devices, "0")

    cfg_dict["cuda_visible_devices"] = "0,1"
    path2 = self._create_temp_yaml(cfg_dict)
    loaded2 = config_loader.load_config(path2)
    self.assertEqual(loaded2.cuda_visible_devices, "0,1")

  def test_load_config_unknown_prompt_to_detect(self):
    """Ensures ConfigError is raised when prompt_to_detect is not in prompts."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["prompt_to_detect"] = "unknown_object"
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError,
        "prompt_to_detect 'unknown_object' has no block",
    ):
      config_loader.load_config(path)

  def test_load_config_invalid_crop_variants(self):
    """Ensures ConfigError is raised for unknown or duplicate crop variants."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["crop_variants"] = ["invalid_variant"]
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Unknown crop variant 'invalid_variant'"
    ):
      config_loader.load_config(path)

    cfg_dict["crop_variants"] = ["raw", "raw"]
    path_dup = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Duplicate crop variant 'raw'"
    ):
      config_loader.load_config(path_dup)

  def test_load_config_invalid_augmentations(self):
    """Ensures ConfigError is raised for unknown or duplicate augmentations."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["prompts"]["packets"]["augmentations"] = ["unknown_aug"]
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Unknown augmentation 'unknown_aug'"
    ):
      config_loader.load_config(path)

    cfg_dict["prompts"]["packets"]["augmentations"] = ["vflip", "vflip"]
    path_dup = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(
        config_loader.ConfigError, "Duplicate augmentation 'vflip'"
    ):
      config_loader.load_config(path_dup)

  def test_canonical_reordering(self):
    """Verifies augmentations and crop_variants are reordered canonically."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["crop_variants"] = ["imagenet_mean_background", "raw"]
    cfg_dict["prompts"]["packets"]["augmentations"] = ["rot90", "vflip", "blur"]
    path = self._create_temp_yaml(cfg_dict)
    config = config_loader.load_config(path)

    self.assertEqual(config.crop_variants, ("raw", "imagenet_mean_background"))
    self.assertEqual(config.active_augmentations, ("vflip", "rot90", "blur"))

  @parameterized.parameters(
      ("confidence_threshold", 1.5, "must be between 0.0 and 1.0"),
      ("confidence_threshold", -0.1, "must be between 0.0 and 1.0"),
      ("confidence_threshold", True, "must be a number"),
      ("score_threshold", 2.0, "must be between 0.0 and 1.0"),
      ("containment_threshold", -0.5, "must be between 0.0 and 1.0"),
      ("max_short_side", 0, "must be at least 1"),
      ("max_short_side", -10, "must be at least 1"),
      ("max_short_side", "1024", "must be an integer"),
  )
  def test_detection_threshold_validation(
      self, field_name: str, bad_value: Any, expected_error: str
  ):
    """Ensures detection thresholds validate type and numeric range."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["prompts"]["packets"]["detection"][field_name] = bad_value
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(config_loader.ConfigError, expected_error):
      config_loader.load_config(path)

  @parameterized.parameters(
      ([256], "must be a list of exactly two integers"),
      ([256, 256, 256], "must be a list of exactly two integers"),
      ([256, 0], "must be at least 1"),
      ([0, 256], "must be at least 1"),
      (["256", 256], "must be an integer"),
  )
  def test_crop_size_validation(self, bad_crop_size: Any, expected_error: str):
    """Ensures crop_size validates exactly two positive integers."""
    cfg_dict = self._get_valid_config_dict()
    cfg_dict["prompts"]["packets"]["detection"]["crop_size"] = bad_crop_size
    path = self._create_temp_yaml(cfg_dict)
    with self.assertRaisesRegex(config_loader.ConfigError, expected_error):
      config_loader.load_config(path)

  def test_derive_sibling_dir(self):
    """Verifies sibling directory path derivation with and without trailing slash."""
    self.assertEqual(
        config_loader._derive_sibling_dir("/data/run", "_classifier"),
        "/data/run_classifier",
    )
    self.assertEqual(
        config_loader._derive_sibling_dir("/data/run/", "_empty"),
        "/data/run_empty",
    )


if __name__ == "__main__":
  absltest.main()
