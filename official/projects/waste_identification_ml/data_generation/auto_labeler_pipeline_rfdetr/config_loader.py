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

"""Loads, validates, and exposes the pipeline configuration.

This module is the single entry point for pipeline configuration. Every
stage script calls :func:`load_config` at the top of its ``main`` and then
reads typed attributes off the returned :class:`PipelineConfig` instead of
reaching into module globals or re-parsing YAML.

The design splits configuration into two kinds of value:

  * Operator knobs live in ``config.yaml`` and may be edited by hand by
    anyone, technical or not. Paths, split ratios, detection thresholds,
    and the augmentation list all live there.

  * Invariants live here in code as module constants: the canonical
    augmentation order, the set of allowed augmentation names, the set of
    allowed crop variants, the allowed folder-name values, and the numeric
    ranges that thresholds must fall in. These are not knobs; they define
    what the pipeline is capable of, and the YAML is validated against them.

Because ``config.yaml`` is hand-edited, validation is strict and eager. A bad
value produces a :class:`ConfigError` at load time, before any dataset walk or
GPU work begins, with a message naming the offending field and the allowed
values. The goal is that a typo never survives long enough to waste a GPU run.
"""

import dataclasses
import os
from typing import Any

import yaml

# ── Invariants (fixed in code, never operator-editable) ─────────────────────

# Canonical order in which augmentations are emitted to disk. Output filenames
# follow this order regardless of how the YAML lists them, so a run is
# deterministic. This is the authoritative list of every augmentation the
# pipeline knows how to produce.
CANONICAL_AUGMENTATION_ORDER = (
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

# Every crop variant segmentation.py knows how to write.
ALLOWED_CROP_VARIANTS = (
    "raw",
    "black_background",
    "imagenet_mean_background",
)

# The pipeline's on-disk folder names are fixed in code. Each YAML field
# below is validated to be exactly the single allowed value. They are exposed
# through the config so scripts do not embed the string literal themselves.
_ALLOWED_FOLDER_NAMES = {
    "input_images_folder_name": ("images",),
    "train_val_folder_name": ("train_val_images",),
    "train_split_name": ("train",),
    "val_split_name": ("val",),
}

# Top-level keys required in config.yaml.
_REQUIRED_TOP_LEVEL_KEYS = (
    "root_dir",
    "rfdetr_checkpoint_path",
    "cuda_visible_devices",
    "input_images_folder_name",
    "train_val_folder_name",
    "train_split_name",
    "val_split_name",
    "keep_every_nth",
    "train_ratio",
    "min_detections",
    "crop_variants",
    "max_cpu_workers",
    "queue_maxsize",
    "rotation_fill_color",
    "predict_threshold",
    "score_threshold",
    "containment_threshold",
    "merge_containment_threshold",
    "max_short_side",
    "crop_size",
    "augmentations",
)

# RGB channel bounds for rotation_fill_color.
_MIN_RGB_VALUE = 0
_MAX_RGB_VALUE = 255


class ConfigError(Exception):
  """Raised when config.yaml is missing, malformed, or out of range.

  The message is written for a human editing the YAML by hand: it names the
  offending field and, where relevant, the allowed values.
  """


# ── Typed configuration objects ─────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
  """Fully validated configuration for one pipeline run.

  Attributes:
      root_dir: Parent directory containing one subfolder per dataset.
      classifier_dir: Sibling directory for the classifier-ready dataset,
        derived from ``root_dir`` by appending ``_classifier`` to its final
        path component.
      rejected_dir: Sibling directory that receives sparse images, derived
        from ``root_dir`` by appending ``_empty`` to its final path component.
      rfdetr_checkpoint_path: Absolute path to the RFDETR checkpoint.
      cuda_visible_devices: Value assigned to CUDA_VISIBLE_DEVICES.
      input_images_folder_name: Subfolder inside each dataset holding raw
        input images.
      train_val_folder_name: Subfolder inside each dataset written by the
        train/val split stage.
      train_split_name: Name of the train split folder.
      val_split_name: Name of the val split folder.
      keep_every_nth: Subsampling interval for the train/val split.
      train_ratio: Fraction of kept images assigned to the val split (see
        note in ``split_train_val.py``).
      min_detections: Minimum post-filter detections to keep an image.
      crop_variants: Crop variants to write, in canonical allowed order.
      max_cpu_workers: Size of the CPU thread pool used for crop saving.
      queue_maxsize: Maximum in-flight CPU jobs before the GPU loop blocks.
      rotation_fill_color: RGB fill color used to pad rotated images.
      predict_threshold: Minimum confidence passed to ``RFDETR.predict``.
        Detections below this are dropped by RFDETR before any pipeline
        post-processing runs.
      score_threshold: Minimum score for a detection to be saved as a crop.
      containment_threshold: Ratio above which a smaller mask is treated as
        contained by a larger one and removed by the mask filter.
      merge_containment_threshold: Ratio above which a smaller box is
        treated as contained by a larger one and merged.
      max_short_side: Maximum length of the shorter image side at inference.
      crop_size: Output letterbox size as a ``(height, width)`` tuple.
      augmentations: Augmentation names to apply, already reordered into
        canonical order so downstream output is deterministic.
  """

  root_dir: str
  classifier_dir: str
  rejected_dir: str
  rfdetr_checkpoint_path: str
  cuda_visible_devices: str
  input_images_folder_name: str
  train_val_folder_name: str
  train_split_name: str
  val_split_name: str
  keep_every_nth: int
  train_ratio: float
  min_detections: int
  crop_variants: tuple[str, ...]
  max_cpu_workers: int
  queue_maxsize: int
  rotation_fill_color: tuple[int, int, int]
  predict_threshold: float
  score_threshold: float
  containment_threshold: float
  merge_containment_threshold: float
  max_short_side: int
  crop_size: tuple[int, int]
  augmentations: tuple[str, ...]


# ── Path derivation ─────────────────────────────────────────────────────────


def _derive_sibling_dir(root_dir: str, suffix: str) -> str:
  """Appends a suffix to the final component of a directory path.

  Using the final component rather than naive string concatenation keeps the
  result correct whether or not ``root_dir`` ends in a separator. For
  ``/data/run`` and suffix ``_empty`` the result is ``/data/run_empty``.

  Args:
      root_dir: Source directory path.
      suffix: String appended to the final path component.

  Returns:
      The sibling directory path as a string.
  """
  normalized = root_dir.rstrip(os.sep)
  parent = os.path.dirname(normalized)
  name = os.path.basename(normalized)
  return os.path.join(parent, name + suffix)


# ── Validation helpers ──────────────────────────────────────────────────────


def _require_keys(
    mapping: dict[str, Any],
    required_keys: tuple[str, ...],
    context: str,
) -> None:
  """Raises if any required key is absent from a mapping.

  Args:
      mapping: The mapping to inspect.
      required_keys: Keys that must be present.
      context: Human-readable description of where the mapping came from,
        used in the error message.

  Raises:
      ConfigError: If any required key is missing.
  """
  missing = [key for key in required_keys if key not in mapping]
  if missing:
    raise ConfigError(
        f"{context} is missing required field(s): {', '.join(missing)}."
    )


def _require_number_in_range(
    value: Any,
    field_name: str,
    minimum: float,
    maximum: float,
    allow_int: bool = True,
) -> float:
  """Validates that a value is a number within an inclusive range.

  Booleans are rejected explicitly because ``bool`` is a subclass of ``int``
  in Python and would otherwise slip through numeric checks.

  Args:
      value: The value to validate.
      field_name: Field name used in the error message.
      minimum: Inclusive lower bound.
      maximum: Inclusive upper bound.
      allow_int: Whether integer values are acceptable.

  Returns:
      The validated value as a float.

  Raises:
      ConfigError: If the value is not a number or is out of range.
  """
  allowed_types = (int, float) if allow_int else (float,)
  if isinstance(value, bool) or not isinstance(value, allowed_types):
    raise ConfigError(f"{field_name} must be a number, got {value!r}.")
  if not minimum <= value <= maximum:
    raise ConfigError(
        f"{field_name} must be between {minimum} and {maximum}, "
        f"got {value!r}."
    )
  return float(value)


def _require_positive_int(value: Any, field_name: str) -> int:
  """Validates that a value is a positive (non-zero) integer.

  Args:
      value: The value to validate.
      field_name: Field name used in the error message.

  Returns:
      The validated integer.

  Raises:
      ConfigError: If the value is not a positive integer.
  """
  if isinstance(value, bool) or not isinstance(value, int):
    raise ConfigError(f"{field_name} must be an integer, got {value!r}.")
  if value < 1:
    raise ConfigError(f"{field_name} must be at least 1, got {value!r}.")
  return int(value)


def _require_non_empty_string(value: Any, field_name: str) -> str:
  """Validates that a value is a non-empty string.

  Args:
      value: The value to validate.
      field_name: Field name used in the error message.

  Returns:
      The validated string, unchanged.

  Raises:
      ConfigError: If the value is not a non-empty string.
  """
  if not isinstance(value, str) or not value.strip():
    raise ConfigError(f"{field_name} must be a non-empty string.")
  return value


def _require_allowed_folder_name(value: Any, field_name: str) -> str:
  """Validates a folder-name field against its single allowed value.

  Each folder-name knob has exactly one allowed value declared in
  ``_ALLOWED_FOLDER_NAMES``. Any other value is rejected with a message
  listing the allowed set.

  Args:
      value: The value to validate.
      field_name: The top-level YAML field name (also the key into
        ``_ALLOWED_FOLDER_NAMES``).

  Returns:
      The validated string.

  Raises:
      ConfigError: If the value is not in the allowed set for that field.
  """
  allowed_values = _ALLOWED_FOLDER_NAMES[field_name]
  if value not in allowed_values:
    raise ConfigError(
        f"{field_name} must be one of {list(allowed_values)}, "
        f"got {value!r}."
    )
  return value


def _validate_crop_size(raw_crop_size: Any) -> tuple[int, int]:
  """Validates and normalizes a crop_size entry into a tuple.

  Args:
      raw_crop_size: The value read from YAML; expected to be a two-element
        sequence of positive integers.

  Returns:
      The crop size as a ``(height, width)`` tuple of ints.

  Raises:
      ConfigError: If the value is not two positive integers.
  """
  context = "crop_size"
  if not isinstance(raw_crop_size, (list, tuple)) or len(raw_crop_size) != 2:
    raise ConfigError(
        f"{context} must be a list of exactly two integers, "
        f"got {raw_crop_size!r}."
    )
  height, width = raw_crop_size
  _require_positive_int(height, f"{context}[0]")
  _require_positive_int(width, f"{context}[1]")
  return (int(height), int(width))


def _validate_crop_variants(raw_variants: Any) -> tuple[str, ...]:
  """Validates configured crop variants against the allowed set.

  Args:
      raw_variants: The value read from YAML; expected to be a non-empty
        sequence of allowed variant names.

  Returns:
      The variants reordered to match ``ALLOWED_CROP_VARIANTS``, so on-disk
      layout is deterministic regardless of YAML ordering.

  Raises:
      ConfigError: If the sequence is empty, not a list, or contains an
          unknown or duplicate variant name.
  """
  if not isinstance(raw_variants, (list, tuple)) or not raw_variants:
    raise ConfigError(
        "crop_variants must be a non-empty list. "
        f"Allowed values: {list(ALLOWED_CROP_VARIANTS)}."
    )
  seen = set()
  for variant in raw_variants:
    if variant not in ALLOWED_CROP_VARIANTS:
      raise ConfigError(
          f"Unknown crop variant {variant!r}. "
          f"Allowed values: {list(ALLOWED_CROP_VARIANTS)}."
      )
    if variant in seen:
      raise ConfigError(f"Duplicate crop variant {variant!r}.")
    seen.add(variant)
  return tuple(variant for variant in ALLOWED_CROP_VARIANTS if variant in seen)


def _validate_rotation_fill_color(
    raw_color: Any,
) -> tuple[int, int, int]:
  """Validates the rotation fill color entry into an RGB tuple.

  Args:
      raw_color: The value read from YAML; expected to be a three-element
        sequence of integers in the range ``[0, 255]``.

  Returns:
      The color as an ``(r, g, b)`` tuple of ints.

  Raises:
      ConfigError: If the value is not three integers in the allowed range.
  """
  context = "rotation_fill_color"
  if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
    raise ConfigError(
        f"{context} must be a list of exactly three integers "
        f"in [{_MIN_RGB_VALUE}, {_MAX_RGB_VALUE}], got {raw_color!r}."
    )
  channels = []
  for index, channel_value in enumerate(raw_color):
    if isinstance(channel_value, bool) or not isinstance(channel_value, int):
      raise ConfigError(
          f"{context}[{index}] must be an integer, got {channel_value!r}."
      )
    if not _MIN_RGB_VALUE <= channel_value <= _MAX_RGB_VALUE:
      raise ConfigError(
          f"{context}[{index}] must be in "
          f"[{_MIN_RGB_VALUE}, {_MAX_RGB_VALUE}], got {channel_value!r}."
      )
    channels.append(int(channel_value))
  return (channels[0], channels[1], channels[2])


def _validate_augmentations(raw_augmentations: Any) -> tuple[str, ...]:
  """Validates the augmentation list against the canonical set.

  Args:
      raw_augmentations: The value read from YAML; expected to be a non-empty
        sequence of canonical augmentation names.

  Returns:
      The augmentations reordered to match ``CANONICAL_AUGMENTATION_ORDER``,
      so output filenames are deterministic regardless of YAML ordering.

  Raises:
      ConfigError: If the sequence is empty, not a list, or contains an
          unknown or duplicate augmentation name.
  """
  context = "augmentations"
  if not isinstance(raw_augmentations, (list, tuple)) or not raw_augmentations:
    raise ConfigError(
        f"{context} must be a non-empty list. "
        f"Allowed values: {list(CANONICAL_AUGMENTATION_ORDER)}."
    )
  seen = set()
  for augmentation in raw_augmentations:
    if augmentation not in CANONICAL_AUGMENTATION_ORDER:
      raise ConfigError(
          f"Unknown augmentation {augmentation!r} in {context}. "
          f"Allowed values: {list(CANONICAL_AUGMENTATION_ORDER)}."
      )
    if augmentation in seen:
      raise ConfigError(
          f"Duplicate augmentation {augmentation!r} in {context}."
      )
    seen.add(augmentation)
  return tuple(
      augmentation
      for augmentation in CANONICAL_AUGMENTATION_ORDER
      if augmentation in seen
  )


def _validate_cuda_visible_devices(raw_value: Any) -> str:
  """Validates cuda_visible_devices, allowing ints and coercing to string.

  Args:
      raw_value: The value read from YAML.

  Returns:
      The value as a string suitable for CUDA_VISIBLE_DEVICES.

  Raises:
      ConfigError: If the value is neither a string nor an integer.
  """
  if isinstance(raw_value, bool):
    raise ConfigError(
        f"cuda_visible_devices must be a string or integer, got {raw_value!r}."
    )
  if isinstance(raw_value, int):
    return str(raw_value)
  if isinstance(raw_value, str):
    return raw_value
  raise ConfigError(
      "cuda_visible_devices must be a string (quote it in YAML) or "
      f"integer, got {raw_value!r}."
  )


# ── Public loader ───────────────────────────────────────────────────────────


def load_config(config_path: str) -> PipelineConfig:
  """Reads, validates, and returns the pipeline configuration.

  All validation happens here so that every stage fails at the same gate,
  before any dataset walk or GPU work. On success the returned object is
  fully typed and internally consistent.

  Args:
      config_path: Path to the YAML configuration file.

  Returns:
      A validated :class:`PipelineConfig`.

  Raises:
      ConfigError: If the file is missing, is not valid YAML, is missing
          required fields, or contains out-of-range values.
  """
  if not os.path.isfile(config_path):
    raise ConfigError(f"Config file does not exist: {config_path}")

  try:
    with open(config_path, "r", encoding="utf-8") as config_file:
      raw_config = yaml.safe_load(config_file)
  except OSError as error:
    raise ConfigError(f"Cannot read config file: {error}") from error
  except yaml.YAMLError as error:
    raise ConfigError(f"Config file is not valid YAML: {error}") from error

  if not isinstance(raw_config, dict):
    raise ConfigError(
        "Config file must contain a top-level mapping of settings."
    )

  _require_keys(raw_config, _REQUIRED_TOP_LEVEL_KEYS, "config.yaml")

  root_dir = _require_non_empty_string(raw_config["root_dir"], "root_dir")
  rfdetr_checkpoint_path = _require_non_empty_string(
      raw_config["rfdetr_checkpoint_path"], "rfdetr_checkpoint_path"
  )
  cuda_visible_devices = _validate_cuda_visible_devices(
      raw_config["cuda_visible_devices"]
  )

  input_images_folder_name = _require_allowed_folder_name(
      raw_config["input_images_folder_name"], "input_images_folder_name"
  )
  train_val_folder_name = _require_allowed_folder_name(
      raw_config["train_val_folder_name"], "train_val_folder_name"
  )
  train_split_name = _require_allowed_folder_name(
      raw_config["train_split_name"], "train_split_name"
  )
  val_split_name = _require_allowed_folder_name(
      raw_config["val_split_name"], "val_split_name"
  )

  keep_every_nth = _require_positive_int(
      raw_config["keep_every_nth"], "keep_every_nth"
  )
  train_ratio = _require_number_in_range(
      raw_config["train_ratio"], "train_ratio", 0.0, 1.0
  )
  min_detections = _require_positive_int(
      raw_config["min_detections"], "min_detections"
  )
  crop_variants = _validate_crop_variants(raw_config["crop_variants"])
  max_cpu_workers = _require_positive_int(
      raw_config["max_cpu_workers"], "max_cpu_workers"
  )
  queue_maxsize = _require_positive_int(
      raw_config["queue_maxsize"], "queue_maxsize"
  )
  rotation_fill_color = _validate_rotation_fill_color(
      raw_config["rotation_fill_color"]
  )

  predict_threshold = _require_number_in_range(
      raw_config["predict_threshold"], "predict_threshold", 0.0, 1.0
  )
  score_threshold = _require_number_in_range(
      raw_config["score_threshold"], "score_threshold", 0.0, 1.0
  )
  containment_threshold = _require_number_in_range(
      raw_config["containment_threshold"], "containment_threshold", 0.0, 1.0
  )
  merge_containment_threshold = _require_number_in_range(
      raw_config["merge_containment_threshold"],
      "merge_containment_threshold",
      0.0,
      1.0,
  )
  max_short_side = _require_positive_int(
      raw_config["max_short_side"], "max_short_side"
  )
  crop_size = _validate_crop_size(raw_config["crop_size"])
  augmentations = _validate_augmentations(raw_config["augmentations"])

  return PipelineConfig(
      root_dir=root_dir,
      classifier_dir=_derive_sibling_dir(root_dir, "_classifier"),
      rejected_dir=_derive_sibling_dir(root_dir, "_empty"),
      rfdetr_checkpoint_path=rfdetr_checkpoint_path,
      cuda_visible_devices=cuda_visible_devices,
      input_images_folder_name=input_images_folder_name,
      train_val_folder_name=train_val_folder_name,
      train_split_name=train_split_name,
      val_split_name=val_split_name,
      keep_every_nth=keep_every_nth,
      train_ratio=train_ratio,
      min_detections=min_detections,
      crop_variants=crop_variants,
      max_cpu_workers=max_cpu_workers,
      queue_maxsize=queue_maxsize,
      rotation_fill_color=rotation_fill_color,
      predict_threshold=predict_threshold,
      score_threshold=score_threshold,
      containment_threshold=containment_threshold,
      merge_containment_threshold=merge_containment_threshold,
      max_short_side=max_short_side,
      crop_size=crop_size,
      augmentations=augmentations,
  )
