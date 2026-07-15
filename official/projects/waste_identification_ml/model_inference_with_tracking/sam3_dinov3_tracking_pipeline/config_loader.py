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

"""Loads and validates pipeline configurations using typed data containers."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Self

import yaml


class ConfigurationError(Exception):
  """Base exception for configuration errors."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class PathsConfig:
  """File-system paths used by the pipeline.

  Attributes:
    input_image_directory: Root directory containing immediate subfolders, each
      holding the image frames of one capture session.
    output_root_directory: Root directory where a per-subfolder result directory
      will be created (mirroring the subfolder name).
    output_frame_subfolder: Name of the per-subfolder folder that stores
      annotated frames.
    output_video_filename: Filename of the annotated tracking video saved inside
      each per-subfolder result directory.
    track_grid_subfolder: Name of the per-subfolder folder that stores
      track-grid PNGs grouped by collapsed category and class.
  """

  input_image_directory: str
  output_root_directory: str
  output_frame_subfolder: str
  output_video_filename: str
  track_grid_subfolder: str


@dataclasses.dataclass(frozen=True)
class SAM3Config:
  """Configuration settings for the SAM3 model.

  Attributes:
    checkpoint_path: Path to the SAM3 model checkpoint file.
    device: Computational device for running inference (e.g., 'cuda' or 'cpu').
  """

  checkpoint_path: str
  device: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class DINOv3Config:
  """Configuration settings for the DINOv3 model.

  Attributes:
    repo_dir: Directory containing the DINOv3 code repository.
    checkpoint_path: Path to the DINOv3 model checkpoint file.
    model_name: Architecture name of the DINOv3 model (e.g., 'dinov3_vitl16').
    inference_image_size: Input image resolution for classification inference.
    classification_batch_size: Batch size used when classifying image crops.
    image_mean: Standard normalization RGB mean values.
    image_std: Standard normalization RGB standard deviation values.
  """

  repo_dir: str
  checkpoint_path: str
  model_name: str
  inference_image_size: int
  classification_batch_size: int
  image_mean: tuple[float, float, float]
  image_std: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class ModelsConfig:
  """Container for all model-related configurations.

  Attributes:
    sam3: Configuration for the SAM3 segmentation model.
    dinov3: Configuration for the DINOv3 classification model.
  """

  sam3: SAM3Config
  dinov3: DINOv3Config


@dataclasses.dataclass(frozen=True, kw_only=True)
class PromptConfig:
  """Detection and cropping hyperparameters for a specific text prompt.

  Attributes:
    confidence_threshold: Minimum confidence threshold for mask generation.
    score_threshold: Minimum score threshold to retain detections.
    containment_threshold: Ratio above which a smaller mask is removed if inside
      a larger mask.
    max_short_side: Maximum size of the image's short edge during resizing.
    crop_size: Output resolution (width, height) for bounding box crops.
    crop_buffer_pixels: Number of buffer pixels to expand around tight boxes.
    merge_overlap_threshold: Intersection over Union threshold for merging
      overlapping bounding boxes of the same prompt class.
  """

  confidence_threshold: float
  score_threshold: float
  containment_threshold: float
  max_short_side: int
  crop_size: tuple[int, int]
  crop_buffer_pixels: int
  merge_overlap_threshold: float = 0.7


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
  """Root configuration for text prompts and detection parameters.

  Attributes:
    active_prompt: Key identifying the currently active prompt configuration.
    image_file_extensions: List of image file extensions to process.
    configs: Mapping from prompt strings to their corresponding `PromptConfig`.
  """

  active_prompt: str
  image_file_extensions: list[str]
  configs: dict[str, PromptConfig]


@dataclasses.dataclass(frozen=True)
class TrackingConfig:
  """ByteTrack settings and a toggle to bypass tracking entirely.

  Attributes:
    bytetrack_minimum_iou_threshold: Minimum IoU for ByteTrack to link a
      detection to an existing track. Ignored when `enable` is False.
    bytetrack_minimum_consecutive_frames: Minimum frames a track must persist
      before ByteTrack emits it. Ignored when `enable` is False.
    enable: When True (default), ByteTrack runs normally and IDs are stable
      across frames. When False, tracking is bypassed entirely and every
      detection in every frame receives a fresh sequential ID. Use False when
      input images are independent (not consecutive video frames).
  """

  bytetrack_minimum_iou_threshold: float
  bytetrack_minimum_consecutive_frames: int
  enable: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class VisualizationConfig:
  """Options and visual styling settings for pipeline output visualization.

  Attributes:
    save_frames: Whether to save individual annotated frames.
    save_video: Whether to compile and save an output tracking video.
    save_track_grids: Whether to generate and save track gallery grid images.
    output_video_fps: Frames per second for generated tracking output videos.
    show_confidence_in_labels: Whether to overlay confidence scores on labels.
    background_blend_color_rgb: RGB background color used for crops.
    track_grid_columns_per_row: Number of track thumbnails per row in grids.
    track_grid_thumbnail_size_inches: Size in inches of each grid thumbnail.
    track_grid_dpi: Dots per inch (DPI) for saving track grid figures.
  """

  save_frames: bool
  save_video: bool
  save_track_grids: bool
  output_video_fps: int
  show_confidence_in_labels: bool
  background_blend_color_rgb: tuple[int, int, int]
  track_grid_columns_per_row: int
  track_grid_thumbnail_size_inches: int
  track_grid_dpi: int


@dataclasses.dataclass(frozen=True)
class CollapsedCategoriesConfig:
  """Optional grouping of fine-grained classes into broader categories.

  When enabled, every class in the pipeline's `classes` list must be assigned
  to exactly one category. The category for a given class is looked up via
  `get_category_for_class`. When disabled, the mapping is empty and no
  per-category reporting is performed.

  Attributes:
    enable: Whether the collapsed-category feature is active.
    mapping: Mapping from category name to the list of class names that fall
      under it. Empty when disabled.
  """

  enable: bool
  mapping: dict[str, list[str]] = dataclasses.field(default_factory=dict)

  def get_category_for_class(self, class_name: str) -> str | None:
    """Returns the category that contains the given class, or None if disabled.

    Args:
      class_name: The fine-grained class name to look up.

    Returns:
      The matching category name when the feature is enabled, or None when the
      feature is disabled.

    Raises:
      ConfigurationError: If the feature is enabled but the class is not present
        in any category (this indicates the validation in `from_yaml` was
        bypassed).
    """
    if not self.enable:
      return None
    for category_name, class_list in self.mapping.items():
      if class_name in class_list:
        return category_name
    raise ConfigurationError(
        f"Class '{class_name}' is not assigned to any collapsed category."
    )

  @property
  def category_names(self) -> list[str]:
    """Returns the configured category names in declaration order.

    Returns:
      List of category names. Empty list when the feature is disabled.
    """
    if not self.enable:
      return []
    return list(self.mapping.keys())


def build_collapsed_categories_config(
    raw_section: Mapping[str, Any] | None, classes: Sequence[str]
) -> CollapsedCategoriesConfig:
  """Builds and validates the CollapsedCategoriesConfig from raw YAML.

  The YAML section is expected to look like:
      collapsed_categories:
        enable: true
        mapping:
          category_name: [class_a, class_b]

  Validation rules when enabled:
      - Every class in `classes` must appear in exactly one category.
      - No class may appear in more than one category.
      - Every class in the mapping must be present in `classes`.

  Args:
    raw_section: The raw `collapsed_categories` dict from YAML, or None if the
      section was omitted entirely.
    classes: The full list of class names from the config.

  Returns:
    A validated CollapsedCategoriesConfig instance.

  Raises:
    ConfigurationError: If validation fails.
  """
  if raw_section is None or not raw_section.get("enable", False):
    return CollapsedCategoriesConfig(enable=False, mapping={})

  raw_mapping = raw_section.get("mapping") or {}
  if not isinstance(raw_mapping, Mapping) or not raw_mapping:
    raise ConfigurationError(
        "collapsed_categories.enable is true but 'mapping' is empty or "
        "missing."
    )

  seen_classes: dict[str, str] = {}
  mapping_dict: dict[str, list[str]] = {}
  class_names = set(classes)
  for category_name, class_list in raw_mapping.items():
    if not isinstance(class_list, Sequence) or isinstance(class_list, str):
      raise ConfigurationError(
          f"Category '{category_name}' must map to a list of class names."
      )
    if not class_list:
      raise ConfigurationError(
          f"Category '{category_name}' must map to a non-empty list of class "
          "names."
      )
    mapping_dict[str(category_name)] = list(class_list)
    for class_name in class_list:
      if class_name not in class_names:
        raise ConfigurationError(
            f"Class '{class_name}' in category '{category_name}' is not "
            "present in the top-level 'classes' list."
        )
      if class_name in seen_classes:
        raise ConfigurationError(
            f"Class '{class_name}' is assigned to both "
            f"'{seen_classes[class_name]}' and '{category_name}'."
        )
      seen_classes[class_name] = category_name

  unmapped_classes = [c for c in classes if c not in seen_classes]
  if unmapped_classes:
    raise ConfigurationError(
        "The following classes are not assigned to any collapsed category: "
        f"{unmapped_classes}"
    )

  return CollapsedCategoriesConfig(enable=True, mapping=mapping_dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PipelineConfig:
  """Root configuration object representing the entire pipeline state."""

  paths: PathsConfig
  models: ModelsConfig
  classes: list[str]
  detection: DetectionConfig
  tracking: TrackingConfig
  visualization: VisualizationConfig
  collapsed_categories: CollapsedCategoriesConfig

  @classmethod
  def from_yaml(cls, yaml_path: str) -> Self:
    """Parses the YAML file into a strictly typed configuration object.

    Args:
      yaml_path: Path to the YAML configuration file.

    Returns:
      A fully populated PipelineConfig.

    Raises:
      ConfigurationError: If the YAML file cannot be found, if the file content
        is invalid YAML, or if configuration validation fails.
    """
    try:
      with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    except OSError as err:
      raise ConfigurationError(
          f"Config file not found or inaccessible: {yaml_path}"
      ) from err
    except yaml.YAMLError as err:
      raise ConfigurationError(
          f"Invalid YAML syntax in {yaml_path}: {err}"
      ) from err

    if not isinstance(data, dict):
      raise ConfigurationError(
          f"Configuration root in {yaml_path} must be a dictionary."
      )

    try:
      prompt_configs = {}
      for name, cfg in data["detection"]["configs"].items():
        cfg_copy = dict(cfg)
        if "crop_size" in cfg_copy:
          cfg_copy["crop_size"] = tuple(cfg_copy["crop_size"])
        prompt_configs[name] = PromptConfig(**cfg_copy)

      classes = list(data["classes"])
      collapsed_categories = build_collapsed_categories_config(
          raw_section=data.get("collapsed_categories"),
          classes=classes,
      )

      models_data = data["models"]
      dinov3_data = dict(models_data["dinov3"])
      if "image_mean" in dinov3_data:
        dinov3_data["image_mean"] = tuple(dinov3_data["image_mean"])
      if "image_std" in dinov3_data:
        dinov3_data["image_std"] = tuple(dinov3_data["image_std"])

      visualization_data = dict(data["visualization"])
      if "background_blend_color_rgb" in visualization_data:
        visualization_data["background_blend_color_rgb"] = tuple(
            visualization_data["background_blend_color_rgb"]
        )

      return cls(
          paths=PathsConfig(**data["paths"]),
          models=ModelsConfig(
              sam3=SAM3Config(**models_data["sam3"]),
              dinov3=DINOv3Config(**dinov3_data),
          ),
          classes=classes,
          detection=DetectionConfig(
              active_prompt=data["detection"]["active_prompt"],
              image_file_extensions=list(
                  data["detection"]["image_file_extensions"]
              ),
              configs=prompt_configs,
          ),
          tracking=TrackingConfig(**data["tracking"]),
          visualization=VisualizationConfig(**visualization_data),
          collapsed_categories=collapsed_categories,
      )
    except ConfigurationError:
      raise
    except Exception as err:
      raise ConfigurationError(
          f"Error validating configuration structure in {yaml_path}: {err}"
      ) from err


