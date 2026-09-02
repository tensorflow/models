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

import copy
import dataclasses
import pathlib
from typing import Any

from absl.testing import absltest
import yaml

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader


def _valid_config_mapping() -> dict[str, Any]:
  """Returns a fully valid config mapping usable as a test baseline.

  Individual tests deep-copy this and mutate a single section so each test
  isolates exactly one rule.
  """
  return {
      "paths": {
          "local": {
              "enable": True,
              "input_image_directory": "/data/in",
              "output_root_directory": "/data/out",
          },
          "gcs": {
              "enable": False,
              "input_uri": "gs://bucket/in/",
              "output_uri": "gs://bucket/out/",
              "temp_input_directory": "/tmp/in",
              "temp_output_directory": "/tmp/out",
          },
          "output_frame_subfolder": "tracked_frames",
          "output_video_filename": "tracked_output.mp4",
          "track_grid_subfolder": "track_grids_by_category",
      },
      "bigquery": {
          "enable": False,
          "project_id": "proj",
          "dataset_id": "ds",
          "table_id": "tbl",
          "overwrite": True,
      },
      "models": {
          "rfdetr": {
              "checkpoint_path": "/models/rfdetr.pth",
              "device": "cuda",
              "image_file_extensions": ["*.png", "*.jpg"],
              "predict_threshold": 0.2,
          },
          "dinov3": {
              "repo_dir": "/models/dinov3",
              "checkpoint_path": "/models/dinov3.pth",
              "model_name": "dinov3_vitl16",
              "inference_image_size": 256,
              "classification_batch_size": 32,
              "image_mean": [0.485, 0.456, 0.406],
              "image_std": [0.229, 0.224, 0.225],
          },
      },
      "classes": ["clean_grade1", "dirty_grade3"],
      "preprocessing": {"max_short_side": 1024},
      "post_processing": {
          "containment_threshold": 0.98,
          "merge_containment_threshold": 0.7,
          "score_threshold": 0.2,
      },
      "cropping": {"crop_size": [256, 256], "crop_buffer_pixels": 5},
      "tracking": {
          "enable": True,
          "bytetrack_minimum_iou_threshold": 0.1,
          "bytetrack_minimum_consecutive_frames": 2,
      },
      "visualization": {
          "save_frames": True,
          "save_video": False,
          "save_track_grids": True,
          "output_video_fps": 1,
          "show_confidence_in_labels": True,
          "background_blend_color_rgb": [124, 116, 104],
          "track_grid_columns_per_row": 5,
          "track_grid_thumbnail_size_inches": 3,
          "track_grid_dpi": 150,
      },
      "collapsed_categories": {
          "enable": True,
          "mapping": {
              "grade1": ["clean_grade1"],
              "grade3": ["dirty_grade3"],
          },
      },
  }


class CollapsedCategoriesConfigTest(absltest.TestCase):
  """Tests for the CollapsedCategoriesConfig behavior methods."""

  def test_get_category_for_class_when_disabled_returns_none(self):
    """Verifies a disabled config returns None for any class."""
    config = config_loader.CollapsedCategoriesConfig(enable=False, mapping={})
    self.assertIsNone(config.get_category_for_class("anything"))

  def test_get_category_for_class_returns_matching_category(self):
    """Verifies the containing category is returned for a mapped class."""
    config = config_loader.CollapsedCategoriesConfig(
        enable=True, mapping={"grade1": ["a", "b"], "grade3": ["c"]}
    )
    self.assertEqual(config.get_category_for_class("c"), "grade3")

  def test_get_category_for_class_raises_for_unmapped_class(self):
    """Verifies an enabled config raises when the class is unmapped."""
    config = config_loader.CollapsedCategoriesConfig(
        enable=True, mapping={"grade1": ["a"]}
    )
    with self.assertRaises(config_loader.ConfigurationError):
      config.get_category_for_class("missing")

  def test_get_category_names_when_disabled_is_empty(self):
    """Verifies a disabled config reports no category names."""
    config = config_loader.CollapsedCategoriesConfig(enable=False, mapping={})
    self.assertEqual(config.get_category_names(), [])

  def test_get_category_names_preserves_declaration_order(self):
    """Verifies category names are returned in mapping insertion order."""
    config = config_loader.CollapsedCategoriesConfig(
        enable=True, mapping={"grade3": ["c"], "grade1": ["a"]}
    )
    self.assertEqual(config.get_category_names(), ["grade3", "grade1"])


class BuildPathsConfigTest(absltest.TestCase):
  """Tests for _build_paths_config source-selection validation."""

  def test_local_only_is_valid(self):
    """Verifies enabling only local produces a PathsConfig."""
    section = _valid_config_mapping()["paths"]
    result = config_loader.PipelineConfig._build_paths_config(section)
    self.assertTrue(result.local.enable)
    self.assertFalse(result.gcs.enable)

  def test_raises_when_both_sources_enabled(self):
    """Verifies enabling both local and GCS raises."""
    section = _valid_config_mapping()["paths"]
    section["local"]["enable"] = True
    section["gcs"]["enable"] = True
    with self.assertRaisesRegex(
        config_loader.ConfigurationError, "Exactly one"
    ):
      config_loader.PipelineConfig._build_paths_config(section)

  def test_raises_when_neither_source_enabled(self):
    """Verifies enabling neither source raises."""
    section = _valid_config_mapping()["paths"]
    section["local"]["enable"] = False
    section["gcs"]["enable"] = False
    with self.assertRaisesRegex(
        config_loader.ConfigurationError, "Exactly one"
    ):
      config_loader.PipelineConfig._build_paths_config(section)


class BuildBigQueryConfigTest(absltest.TestCase):
  """Tests for _build_bigquery_config."""

  def test_missing_section_returns_disabled(self):
    """Verifies a None section yields a disabled BigQuery config."""
    result = config_loader.PipelineConfig._build_bigquery_config(None)
    self.assertFalse(result.enable)
    self.assertEqual(result.project_id, "")

  def test_disabled_section_returns_disabled(self):
    """Verifies an explicitly disabled section yields a disabled config."""
    result = config_loader.PipelineConfig._build_bigquery_config(
        {"enable": False}
    )
    self.assertFalse(result.enable)

  def test_enabled_with_all_ids_is_valid(self):
    """Verifies an enabled section with all IDs is accepted."""
    result = config_loader.PipelineConfig._build_bigquery_config({
        "enable": True,
        "project_id": "p",
        "dataset_id": "d",
        "table_id": "t",
        "overwrite": True,
    })
    self.assertTrue(result.enable)
    self.assertTrue(result.overwrite)

  def test_enabled_with_missing_id_raises(self):
    """Verifies an enabled section missing an ID raises, naming the field."""
    with self.assertRaisesRegex(config_loader.ConfigurationError, "table_id"):
      config_loader.PipelineConfig._build_bigquery_config({
          "enable": True,
          "project_id": "p",
          "dataset_id": "d",
          "table_id": "",
      })

  def test_overwrite_defaults_to_false(self):
    """Verifies overwrite defaults to False when omitted."""
    result = config_loader.PipelineConfig._build_bigquery_config({
        "enable": True,
        "project_id": "p",
        "dataset_id": "d",
        "table_id": "t",
    })
    self.assertFalse(result.overwrite)


class BuildCollapsedCategoriesConfigTest(absltest.TestCase):
  """Tests for _build_collapsed_categories_config validation."""

  def test_disabled_when_section_missing(self):
    """Verifies a None section yields a disabled config."""
    result = config_loader.PipelineConfig._build_collapsed_categories_config(
        raw_section=None, classes=["a"]
    )
    self.assertFalse(result.enable)
    self.assertEqual(result.mapping, {})

  def test_valid_full_partition_is_accepted(self):
    """Verifies a mapping covering every class exactly once is accepted."""
    result = config_loader.PipelineConfig._build_collapsed_categories_config(
        raw_section={
            "enable": True,
            "mapping": {"g1": ["a"], "g3": ["b", "c"]},
        },
        classes=["a", "b", "c"],
    )
    self.assertTrue(result.enable)
    self.assertEqual(result.get_category_for_class("b"), "g3")

  def test_raises_when_enabled_but_mapping_empty(self):
    """Verifies enabling the feature with an empty mapping raises."""
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig._build_collapsed_categories_config(
          raw_section={"enable": True, "mapping": {}}, classes=["a"]
      )

  def test_raises_when_category_maps_to_empty_list(self):
    """Verifies a category with an empty class list raises."""
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig._build_collapsed_categories_config(
          raw_section={"enable": True, "mapping": {"g1": []}},
          classes=["a"],
      )

  def test_raises_when_class_not_in_top_level_list(self):
    """Verifies a mapped class absent from `classes` raises."""
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig._build_collapsed_categories_config(
          raw_section={"enable": True, "mapping": {"g1": ["ghost"]}},
          classes=["a"],
      )

  def test_raises_when_class_in_two_categories(self):
    """Verifies a class assigned to two categories raises."""
    with self.assertRaisesRegex(config_loader.ConfigurationError, "both"):
      config_loader.PipelineConfig._build_collapsed_categories_config(
          raw_section={
              "enable": True,
              "mapping": {"g1": ["a"], "g3": ["a"]},
          },
          classes=["a"],
      )

  def test_raises_when_class_unmapped(self):
    """Verifies a class present in `classes` but unmapped raises."""
    with self.assertRaisesRegex(
        config_loader.ConfigurationError, "not assigned"
    ):
      config_loader.PipelineConfig._build_collapsed_categories_config(
          raw_section={"enable": True, "mapping": {"g1": ["a"]}},
          classes=["a", "b"],
      )


class FromYamlTest(absltest.TestCase):
  """Tests for the PipelineConfig.from_yaml entry point."""

  def _write_config(self, mapping: dict[str, Any]) -> str:
    """Writes a mapping to a temp YAML file and returns its path."""
    config_path = pathlib.Path(self.create_tempdir().full_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(config_path)

  def test_loads_valid_config(self):
    """Verifies a valid config parses into a populated PipelineConfig."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.PipelineConfig.from_yaml(config_path)
    self.assertIsInstance(config, config_loader.PipelineConfig)
    self.assertEqual(config.classes, ["clean_grade1", "dirty_grade3"])
    self.assertEqual(config.preprocessing.max_short_side, 1024)

  def test_coerces_tuple_fields(self):
    """Verifies list YAML values become tuples where the dataclass expects it."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.PipelineConfig.from_yaml(config_path)
    self.assertEqual(config.models.dinov3.image_mean, (0.485, 0.456, 0.406))
    self.assertEqual(config.cropping.crop_size, (256, 256))
    self.assertEqual(
        config.visualization.background_blend_color_rgb, (124, 116, 104)
    )

  def test_missing_file_raises_configuration_error(self):
    """Verifies a missing YAML file raises ConfigurationError."""
    with self.assertRaisesRegex(config_loader.ConfigurationError, "not found"):
      config_loader.PipelineConfig.from_yaml("/nonexistent/config.yaml")

  def test_both_sources_enabled_raises(self):
    """Verifies the source-XOR rule is enforced end to end."""
    mapping = copy.deepcopy(_valid_config_mapping())
    mapping["paths"]["gcs"]["enable"] = True
    config_path = self._write_config(mapping)
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig.from_yaml(config_path)

  def test_bigquery_enabled_missing_id_raises(self):
    """Verifies BigQuery validation runs through from_yaml."""
    mapping = copy.deepcopy(_valid_config_mapping())
    mapping["bigquery"] = {
        "enable": True,
        "project_id": "p",
        "dataset_id": "d",
        "table_id": "",
    }
    config_path = self._write_config(mapping)
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig.from_yaml(config_path)

  def test_collapsed_categories_partition_validated(self):
    """Verifies an incomplete category partition raises through from_yaml."""
    mapping = copy.deepcopy(_valid_config_mapping())
    # Empty the grade3 list so dirty_grade3 is left unmapped.
    mapping["collapsed_categories"]["mapping"]["grade3"] = []
    config_path = self._write_config(mapping)
    with self.assertRaises(config_loader.ConfigurationError):
      config_loader.PipelineConfig.from_yaml(config_path)

  def test_returned_config_is_frozen(self):
    """Verifies the root PipelineConfig is immutable."""
    config_path = self._write_config(_valid_config_mapping())
    config = config_loader.PipelineConfig.from_yaml(config_path)
    with self.assertRaises(dataclasses.FrozenInstanceError):
      config.classes = []


if __name__ == "__main__":
  absltest.main()
