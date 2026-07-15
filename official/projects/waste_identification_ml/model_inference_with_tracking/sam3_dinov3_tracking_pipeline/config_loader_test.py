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

"""Unit tests verifying pipeline configuration loading, validation, and category mapping."""

import os
from typing import Any
from absl.testing import absltest
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader


class ConfigLoaderTest(absltest.TestCase):
  """Test suite covering YAML deserialization and category mapping validation rules."""

  def setUp(self):
    """Initializes common test fixtures and absolute paths to test configs."""
    super().setUp()
    self.valid_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )

  def assertRaisesConfigError(
      self,
      raw_section: dict[str, Any],
      classes: list[str],
      expected_message_regex: str,
  ):
    """Helper asserting that building collapsed categories config raises ConfigurationError."""
    with self.assertRaisesRegex(
        config_loader.ConfigurationError, expected_message_regex
    ):
      config_loader.build_collapsed_categories_config(raw_section, classes)

  def test_from_yaml_valid_config(self):
    """Verifies that a valid YAML configuration loads with expected typed data containers."""
    config = config_loader.PipelineConfig.from_yaml(self.valid_config_path)
    self.assertIsInstance(config, config_loader.PipelineConfig)
    self.assertEqual(config.models.dinov3.inference_image_size, 256)
    self.assertEqual(config.models.dinov3.classification_batch_size, 32)
    self.assertIsInstance(config.models.dinov3.image_mean, tuple)
    self.assertLen(config.models.dinov3.image_mean, 3)
    self.assertIn("bottles and containers", config.detection.configs)
    self.assertIsInstance(
        config.detection.configs["bottles and containers"].crop_size, tuple
    )
    self.assertEqual(
        config.detection.configs["bottles and containers"].crop_size, (256, 256)
    )
    self.assertFalse(config.collapsed_categories.enable)
    self.assertEmpty(config.collapsed_categories.category_names)

  def test_from_yaml_file_not_found(self):
    """Ensures ConfigurationError is raised when the YAML file path does not exist."""
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig.from_yaml(
          "/non_existent_path/bad_config.yaml"
      )

  def test_from_yaml_invalid_syntax(self):
    """Ensures ConfigurationError is raised when the file contains malformed YAML syntax."""
    temp_file = self.create_tempfile(
        file_path="bad_config.yaml",
        content="paths:\n  input_image_directory: [unclosed_list\n",
    )
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig.from_yaml(temp_file.full_path)

  def test_collapsed_categories_config_enabled_valid(self):
    """Verifies valid category mappings when the collapsed categories feature is enabled."""
    classes = ["class_a", "class_b", "class_c"]
    raw_section = {
        "enable": True,
        "mapping": {
            "cat_1": ["class_a", "class_b"],
            "cat_2": ["class_c"],
        },
    }
    cfg = config_loader.build_collapsed_categories_config(raw_section, classes)
    self.assertTrue(cfg.enable)
    self.assertEqual(cfg.get_category_for_class("class_a"), "cat_1")
    self.assertEqual(cfg.get_category_for_class("class_c"), "cat_2")
    self.assertEqual(cfg.category_names, ["cat_1", "cat_2"])

  def test_collapsed_categories_config_missing_classes(self):
    """Ensures an error is raised when top-level classes are omitted from category mappings."""
    classes = ["class_a", "class_b", "class_c"]
    raw_section = {
        "enable": True,
        "mapping": {
            "cat_1": ["class_a"],
        },
    }
    self.assertRaisesConfigError(
        raw_section, classes, "not assigned to any collapsed category"
    )

  def test_collapsed_categories_config_duplicate_class(self):
    """Ensures an error is raised if a class is assigned to multiple categories simultaneously."""
    classes = ["class_a", "class_b"]
    raw_section = {
        "enable": True,
        "mapping": {
            "cat_1": ["class_a", "class_b"],
            "cat_2": ["class_b"],
        },
    }
    self.assertRaisesConfigError(raw_section, classes, "assigned to both")

  def test_collapsed_categories_get_category_for_class_not_found(self):
    """Verifies lookup behavior when querying an unassigned class on an active config."""
    cfg = config_loader.CollapsedCategoriesConfig(
        enable=True,
        mapping={"cat_1": ["class_a"]},
    )
    with self.assertRaisesRegex(
        config_loader.ConfigurationError,
        "not assigned to any collapsed category",
    ):
      cfg.get_category_for_class("unknown_class")

  def test_collapsed_categories_disabled_return_vals(self):
    """Ensures lookups return None and empty lists when the collapsed categories feature is disabled."""
    cfg = config_loader.CollapsedCategoriesConfig(enable=False, mapping={})
    self.assertIsNone(cfg.get_category_for_class("any_class"))
    self.assertEmpty(cfg.category_names)


if __name__ == "__main__":
  absltest.main()
