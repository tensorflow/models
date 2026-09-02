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

import dataclasses
from typing import Any, Self
import yaml


class ConfigurationError(Exception):
  """Base exception for configuration errors."""

  pass


@dataclasses.dataclass(frozen=True)
class LocalPathsConfig:
  """Local file-system input and output locations.

  Attributes:
      enable: Whether local paths are the active source.
      input_image_directory: Root directory containing immediate subfolders,
        each holding the image frames of one capture session.
      output_root_directory: Root directory where a per-subfolder result
        directory will be created (mirroring the subfolder name).
  """

  enable: bool
  input_image_directory: str
  output_root_directory: str


@dataclasses.dataclass(frozen=True)
class GCSPathsConfig:
  """Google Cloud Storage input and output locations.

  Attributes:
      enable: Whether GCS paths are the active source.
      input_uri: GCS URI (e.g. 'gs://bucket/path/') to read input from.
      output_uri: GCS URI (e.g. 'gs://bucket/path/') to write output to.
      temp_input_directory: Local scratch directory that GCS input is downloaded
        into. Pipeline-owned; cleared before each download.
      temp_output_directory: Local scratch directory that pipeline output is
        written to before being uploaded to GCS.
  """

  enable: bool
  input_uri: str
  output_uri: str
  temp_input_directory: str
  temp_output_directory: str


@dataclasses.dataclass(frozen=True)
class PathsConfig:
  """File-system paths used by the pipeline.

  Exactly one of `local` or `gcs` is enabled (validated at config-load
  time). The output subfolder and file names are shared across both
  sources.

  Attributes:
      local: Local input/output locations and their enable switch.
      gcs: GCS input/output locations and their enable switch.
      output_frame_subfolder: Name of the per-subfolder folder that stores
        annotated frames.
      output_video_filename: Filename of the annotated tracking video saved
        inside each per-subfolder result directory.
      track_grid_subfolder: Name of the per-subfolder folder that stores
        track-grid PNGs grouped by collapsed category and class.
  """

  local: LocalPathsConfig
  gcs: GCSPathsConfig
  output_frame_subfolder: str
  output_video_filename: str
  track_grid_subfolder: str


@dataclasses.dataclass(frozen=True)
class BigQueryConfig:
  """BigQuery ingestion settings.

  Independent of the input/output source: results may be written to
  BigQuery whether the pipeline runs in local or GCS mode.

  Attributes:
      enable: Whether pipeline results are written to BigQuery.
      project_id: Google Cloud project ID that owns the dataset.
      dataset_id: BigQuery dataset ID that holds the table.
      table_id: BigQuery table ID that receives the rows.
      overwrite: When True, existing rows are replaced by this run's rows. When
        False, this run's rows are appended.
  """

  enable: bool
  project_id: str
  dataset_id: str
  table_id: str
  overwrite: bool


@dataclasses.dataclass(frozen=True)
class RFDETRConfig:
  """RFDETR segmentation-model settings.

  Everything the model itself needs lives here: checkpoint, device, its
  own confidence gate, and the file extensions used to discover images
  the model will be run on.

  Attributes:
      checkpoint_path: Absolute path to the RFDETR .pth checkpoint (typically
        the CircularNet fine-tuned weights).
      device: Target torch device string, e.g. 'cuda'. Falls back to 'cpu'
        automatically when CUDA is unavailable.
      image_file_extensions: Glob patterns used to discover input images that
        will be fed to the model (e.g. ``["*.png", "*.jpg"]``).
      predict_threshold: Confidence threshold passed to ``RFDETR.predict``.
        Detections below this never reach the pipeline.
  """

  checkpoint_path: str
  device: str
  image_file_extensions: list[str]
  predict_threshold: float


@dataclasses.dataclass(frozen=True)
class DINOv3Config:
  """DINOv3 classification-model settings.

  Attributes:
      repo_dir: Absolute path to the DINOv3 repository directory.
      checkpoint_path: Absolute path to the DINOv3 checkpoint.
      model_name: Name of the DINOv3 model variant to load.
      inference_image_size: Target image size for DINOv3 inference (assumed
        square).
      classification_batch_size: Batch size for DINOv3 classification.
      image_mean: Mean RGB values for image normalization.
      image_std: Standard deviation RGB values for image normalization.
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
  """Container for all model configurations.

  Attributes:
      rfdetr: Configuration for the RFDETR segmentation model.
      dinov3: Configuration for the DINOv3 classification model.
  """

  rfdetr: RFDETRConfig
  dinov3: DINOv3Config


@dataclasses.dataclass(frozen=True)
class PreprocessingConfig:
  """Image preprocessing applied before any model sees the image.

  Attributes:
      max_short_side: Maximum length of the shorter image side. Images larger
        than this are downscaled preserving aspect ratio.
  """

  max_short_side: int


@dataclasses.dataclass(frozen=True)
class PostProcessingConfig:
  """Filters applied to RFDETR's raw output before tracking.

  Applied in this order per image:

    1. containment_threshold       -- mask-in-mask filter.
    2. merge_containment_threshold -- box-in-box merge (unconditional).
    3. score_threshold             -- final cutoff before tracking.

  Attributes:
      containment_threshold: Mask-in-mask cutoff; the smaller mask is dropped
        when its intersection with a larger mask exceeds this fraction of its
        own area.
      merge_containment_threshold: Box-in-box cutoff for the merge step; the
        smaller detection is merged into the larger one when their box
        intersection exceeds this fraction of the smaller box's area.
      score_threshold: Final score cutoff applied when converting the state dict
        to ``supervision.Detections``.
  """

  containment_threshold: float
  merge_containment_threshold: float
  score_threshold: float


@dataclasses.dataclass(frozen=True)
class CroppingConfig:
  """Per-track crop geometry (crops that feed DINOv3).

  Attributes:
      crop_size: Output letterbox size as a ``(height, width)`` tuple.
      crop_buffer_pixels: Pixel buffer added around each detection box before
        cropping (guards against edge clipping).
  """

  crop_size: tuple[int, int]
  crop_buffer_pixels: int


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


@dataclasses.dataclass(frozen=True)
class VisualizationConfig:
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

  When enabled, every class in the pipeline's `classes` list must be
  assigned to exactly one category. The category for a given class is
  looked up via `get_category_for_class`. When disabled, the mapping is
  empty and no per-category reporting is performed.

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
        The matching category name when the feature is enabled, or None
        when the feature is disabled.

    Raises:
        ConfigurationError: If the feature is enabled but the class is
            not present in any category (this indicates the validation
            in `from_yaml` was bypassed).
    """
    if not self.enable:
      return None
    for category_name, class_list in self.mapping.items():
      if class_name in class_list:
        return category_name
    raise ConfigurationError(
        f"Class '{class_name}' is not assigned to any collapsed category."
    )

  def get_category_names(self) -> list[str]:
    """Returns the configured category names in declaration order.

    Returns:
        List of category names. Empty list when the feature is disabled.
    """
    if not self.enable:
      return []
    return list(self.mapping.keys())


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
  """Root configuration object representing the entire pipeline state."""

  paths: PathsConfig
  bigquery: BigQueryConfig
  models: ModelsConfig
  classes: list[str]
  preprocessing: PreprocessingConfig
  post_processing: PostProcessingConfig
  cropping: CroppingConfig
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
        ConfigurationError: If the YAML file cannot be found, if exactly
            one path source is not enabled, if BigQuery is enabled with a
            missing ID, or if the collapsed_categories section is enabled
            but invalid.
    """
    try:
      with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    except FileNotFoundError as err:
      raise ConfigurationError(f"Config file not found: {yaml_path}") from err

    classes = data["classes"]
    collapsed_categories = cls._build_collapsed_categories_config(
        raw_section=data.get("collapsed_categories"),
        classes=classes,
    )
    return cls(
        paths=cls._build_paths_config(data["paths"]),
        bigquery=cls._build_bigquery_config(data.get("bigquery")),
        models=ModelsConfig(
            rfdetr=cls._build_rfdetr_config(data["models"]["rfdetr"]),
            dinov3=DINOv3Config(**{
                **data["models"]["dinov3"],
                "image_mean": tuple(data["models"]["dinov3"]["image_mean"]),
                "image_std": tuple(data["models"]["dinov3"]["image_std"]),
            }),
        ),
        classes=classes,
        preprocessing=cls._build_preprocessing_config(data["preprocessing"]),
        post_processing=cls._build_post_processing_config(
            data["post_processing"]
        ),
        cropping=cls._build_cropping_config(data["cropping"]),
        tracking=TrackingConfig(**data["tracking"]),
        visualization=VisualizationConfig(**{
            **data["visualization"],
            "background_blend_color_rgb": tuple(
                data["visualization"]["background_blend_color_rgb"]
            ),
        }),
        collapsed_categories=collapsed_categories,
    )

  @staticmethod
  def _build_paths_config(raw_section: dict[str, Any]) -> PathsConfig:
    """Builds and validates the PathsConfig from raw YAML.

    Exactly one of the local or GCS sources must be enabled. Enabling
    both, or enabling neither, is a configuration error.

    Args:
        raw_section: The raw `paths` dict from YAML.

    Returns:
        A validated PathsConfig instance.

    Raises:
        ConfigurationError: If both sources are enabled or both are
            disabled.
    """
    local_section = raw_section["local"]
    gcs_section = raw_section["gcs"]
    local_enabled = local_section["enable"]
    gcs_enabled = gcs_section["enable"]

    if local_enabled and gcs_enabled:
      raise ConfigurationError(
          "Both paths.local.enable and paths.gcs.enable are true. "
          "Exactly one source must be enabled."
      )
    if not local_enabled and not gcs_enabled:
      raise ConfigurationError(
          "Both paths.local.enable and paths.gcs.enable are false. "
          "Exactly one source must be enabled."
      )

    return PathsConfig(
        local=LocalPathsConfig(**local_section),
        gcs=GCSPathsConfig(**gcs_section),
        output_frame_subfolder=raw_section["output_frame_subfolder"],
        output_video_filename=raw_section["output_video_filename"],
        track_grid_subfolder=raw_section["track_grid_subfolder"],
    )

  @staticmethod
  def _build_bigquery_config(
      raw_section: dict[str, Any] | None,
  ) -> BigQueryConfig:
    """Builds and validates the BigQueryConfig from raw YAML.

    The section is optional: when it is missing or disabled, a disabled
    config is returned. When enabled, the project, dataset, and table
    IDs must all be non-empty.

    Args:
        raw_section: The raw `bigquery` dict from YAML, or None if the section
          was omitted entirely.

    Returns:
        A validated BigQueryConfig instance.

    Raises:
        ConfigurationError: If enabled but any required ID is empty.
    """
    if raw_section is None or not raw_section.get("enable", False):
      return BigQueryConfig(
          enable=False,
          project_id="",
          dataset_id="",
          table_id="",
          overwrite=False,
      )

    required_fields = ("project_id", "dataset_id", "table_id")
    missing_fields = [
        field_name
        for field_name in required_fields
        if not raw_section.get(field_name)
    ]
    if missing_fields:
      raise ConfigurationError(
          "bigquery.enable is true but these required fields are empty: "
          f"{missing_fields}"
      )

    return BigQueryConfig(
        enable=True,
        project_id=raw_section["project_id"],
        dataset_id=raw_section["dataset_id"],
        table_id=raw_section["table_id"],
        overwrite=raw_section.get("overwrite", False),
    )

  @staticmethod
  def _build_rfdetr_config(raw_section: dict[str, Any]) -> RFDETRConfig:
    """Builds and validates the RFDETRConfig from raw YAML.

    Args:
        raw_section: The raw `models.rfdetr` dict from YAML.

    Returns:
        A validated RFDETRConfig instance.
    """
    return RFDETRConfig(
        checkpoint_path=raw_section["checkpoint_path"],
        device=raw_section["device"],
        image_file_extensions=list(raw_section["image_file_extensions"]),
        predict_threshold=float(raw_section["predict_threshold"]),
    )

  @staticmethod
  def _build_preprocessing_config(
      raw_section: dict[str, Any],
  ) -> PreprocessingConfig:
    """Builds and validates the PreprocessingConfig from raw YAML.

    Args:
        raw_section: The raw `preprocessing` dict from YAML.

    Returns:
        A validated PreprocessingConfig instance.
    """
    return PreprocessingConfig(
        max_short_side=int(raw_section["max_short_side"]),
    )

  @staticmethod
  def _build_post_processing_config(
      raw_section: dict[str, Any],
  ) -> PostProcessingConfig:
    """Builds and validates the PostProcessingConfig from raw YAML.

    Args:
        raw_section: The raw `post_processing` dict from YAML.

    Returns:
        A validated PostProcessingConfig instance.
    """
    return PostProcessingConfig(
        containment_threshold=float(raw_section["containment_threshold"]),
        merge_containment_threshold=float(
            raw_section["merge_containment_threshold"]
        ),
        score_threshold=float(raw_section["score_threshold"]),
    )

  @staticmethod
  def _build_cropping_config(raw_section: dict[str, Any]) -> CroppingConfig:
    """Builds and validates the CroppingConfig from raw YAML.

    Args:
        raw_section: The raw `cropping` dict from YAML.

    Returns:
        A validated CroppingConfig instance.
    """
    return CroppingConfig(
        crop_size=tuple(raw_section["crop_size"]),
        crop_buffer_pixels=int(raw_section["crop_buffer_pixels"]),
    )

  @staticmethod
  def _build_collapsed_categories_config(
      raw_section: dict[str, Any] | None, classes: list[str]
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
        raw_section: The raw `collapsed_categories` dict from YAML, or None if
          the section was omitted entirely.
        classes: The full list of class names from the config.

    Returns:
        A validated CollapsedCategoriesConfig instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    if raw_section is None or not raw_section.get("enable", False):
      return CollapsedCategoriesConfig(enable=False, mapping={})

    raw_mapping = raw_section.get("mapping") or {}
    if not isinstance(raw_mapping, dict) or not raw_mapping:
      raise ConfigurationError(
          "collapsed_categories.enable is true but 'mapping' is empty or"
          " missing."
      )

    seen_classes: dict[str, str] = {}
    for category_name, class_list in raw_mapping.items():
      if not isinstance(class_list, list) or not class_list:
        raise ConfigurationError(
            f"Category '{category_name}' must map to a non-empty list of class"
            " names."
        )
      for class_name in class_list:
        if class_name not in classes:
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

    unmapped_classes = [
        class_name for class_name in classes if class_name not in seen_classes
    ]
    if unmapped_classes:
      raise ConfigurationError(
          "The following classes are not assigned to any collapsed "
          f"category: {unmapped_classes}"
      )

    return CollapsedCategoriesConfig(enable=True, mapping=dict(raw_mapping))
