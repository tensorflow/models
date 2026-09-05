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

"""RFDETR sparse-image filter: move images with fewer than ``min_detections``.

For each dataset under ``config.root_dir``, walks its input images folder
recursively, runs RFDETR on every supported image, and moves any image whose
post-filter detection count is below ``config.min_detections`` into the
sibling rejected directory at ``config.rejected_dir``, preserving each
image's full relative path from the pipeline root.

The same contained-sub-mask, contained-box-merge, and edge-visibility filters
as the batch segmentation pipeline are applied. The score threshold is NOT
applied here -- this stage is intended to discard images that contain too
few objects to be useful for downstream training, regardless of confidence
scores.

Logging is per-dataset only: header, a per-dataset summary at the end, and a
final overall summary. Per-image chatter is intentionally suppressed. Skips
and inference failures still surface as warnings so nothing is silently
dropped.

No crops are saved. Images that pass the filter are left in place; only
rejected images are moved.
"""

import gc
import logging
import os
import shutil
import sys
import time
from typing import Any
import warnings

import natsort
from PIL import Image
import torch
import tqdm

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import config_loader
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import detection_utils

# ── Warning suppression ─────────────────────────────────────────────────────
# Silence noise that we've reviewed and know is harmless for our setup:
#   * torch.jit TracerWarning: raised by RFDETRSegMedium.optimize_for_inference
#     when tracing the model. Only relevant when the traced model must handle
#     different input shapes than the trace saw; not our case.
#   * "rf-detr" logger warnings: informational lines confirming that we're
#     fine-tuning a checkpoint (patch size, DINOv2 weights, detection head
#     reinit). Errors from the same logger are still shown.
#   * "transformers" logger warnings: the "loss_type=None" config notice.
# Everything else (real errors, other libraries' warnings) is left alone.
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
  # pylint: disable=g-import-not-at-top
  from rfdetr import RFDETRSegMedium  # type: ignore[import-error]
  # pylint: enable=g-import-not-at-top
except ImportError:
  RFDETRSegMedium = None


# Resolve config.yaml relative to this script file so the script runs
# correctly regardless of the caller's current working directory.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)
IMAGE_EXTENSIONS = frozenset([".jpg", ".jpeg", ".png"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level logger. Configuration is applied once in ``main`` so that
# importing this module from other code does not force a logging setup.
logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────


def configure_logging() -> None:
  """Configures the module logger to write to stdout.

  Uses a plain, tqdm-friendly format so log lines interleave cleanly with
  the progress bar. Attaches a handler only once even if the function is
  called multiple times.
  """
  if logger.handlers:
    return

  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(
      logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  )
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)
  logger.propagate = False


# ── Model setup ───────────────────────────────────────────────────────────────


def build_rfdetr_model(checkpoint_path: str) -> Any:
  """Builds the RFDETR segmentation model and optimizes it for inference.

  Args:
      checkpoint_path: Absolute path to the RFDETR checkpoint.

  Returns:
      An RFDETRSegMedium instance ready for ``.predict`` calls.

  Raises:
      ImportError: If the ``rfdetr`` package is not installed or available on
          the Python path.
  """
  if RFDETRSegMedium is None:
    raise ImportError(
        "The 'rfdetr' package is not installed or available in python "
        "path. Cannot build the RFDETR model."
    )
  model = RFDETRSegMedium(pretrain_weights=checkpoint_path)
  model.optimize_for_inference()
  return model


# ── Dataset discovery and validation ──────────────────────────────────────────


def discover_dataset_directories(root_dir: str) -> list[tuple[str, str]]:
  """Returns the list of dataset subdirectories directly under ``root_dir``.

  Args:
      root_dir: Path to the root directory containing dataset subfolders.

  Returns:
      A sorted list of ``(dataset_name, dataset_path)`` tuples.

  Raises:
      FileNotFoundError: If ``root_dir`` does not exist.
      ValueError: If ``root_dir`` contains no subdirectories.
  """
  if not os.path.isdir(root_dir):
    raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

  dataset_entries = sorted(
      [entry for entry in os.scandir(root_dir) if entry.is_dir()],
      key=lambda entry: entry.name,
  )

  if not dataset_entries:
    raise ValueError(f"No dataset subfolders found under: {root_dir}")

  return [(entry.name, entry.path) for entry in dataset_entries]


def validate_dataset_paths(
    dataset_directories: list[tuple[str, str]], input_images_folder_name: str
) -> list[tuple[str, str]]:
  """Validates that each dataset has the expected input images folder.

  Args:
      dataset_directories: List of ``(dataset_name, dataset_path)`` tuples.
      input_images_folder_name: Name of the input images subfolder.

  Returns:
      A list of ``(dataset_name, images_dir)`` tuples ready for processing.

  Raises:
      FileNotFoundError: If any dataset is missing its images folder.
  """
  validated = []
  for dataset_name, dataset_path in dataset_directories:
    images_dir = os.path.join(dataset_path, input_images_folder_name)

    if not os.path.isdir(images_dir):
      raise FileNotFoundError(
          f"Dataset {dataset_name!r} is missing required images folder: "
          f"{images_dir}"
      )

    validated.append((dataset_name, images_dir))

  return validated


def validate_rejected_dir(rejected_dir: str) -> None:
  """Ensures the rejected directory does not already exist.

  Args:
      rejected_dir: Path to the rejected directory.

  Raises:
      FileExistsError: If ``rejected_dir`` already exists.
  """
  if os.path.exists(rejected_dir):
    raise FileExistsError(
        f"Rejected directory already exists: {rejected_dir}. "
        "Remove or rename it before re-running."
    )


# ── Filesystem helpers ────────────────────────────────────────────────────────


def gather_image_paths(root_directory: str) -> list[str]:
  """Recursively collects image file paths under a directory.

  Args:
      root_directory: Directory to walk.

  Returns:
      A naturally sorted list of absolute image file paths whose extensions
      (lower-cased) are in ``IMAGE_EXTENSIONS``.
  """
  image_paths = []
  for current_directory, _, filenames in os.walk(root_directory):
    for filename in filenames:
      extension = os.path.splitext(filename)[1].lower()
      if extension in IMAGE_EXTENSIONS:
        image_paths.append(os.path.join(current_directory, filename))
  return natsort.natsorted(image_paths)


def move_to_rejected(
    image_path: str, source_root: str, rejected_root: str
) -> None:
  """Moves an image into the rejected directory preserving relative path.

  Args:
      image_path: Absolute path to the image to move.
      source_root: Absolute path used as the base for relative-path calculation.
        The image's path relative to this root is mirrored under
        ``rejected_root``.
      rejected_root: Absolute path to the rejected root directory.
  """
  relative_path = os.path.relpath(image_path, source_root)
  destination_path = os.path.join(rejected_root, relative_path)
  os.makedirs(os.path.dirname(destination_path), exist_ok=True)
  shutil.move(image_path, destination_path)


# ── Per-image detection count ─────────────────────────────────────────────────


def count_detections(
    image: Image.Image,
    model: Any,
    config: config_loader.PipelineConfig,
) -> int:
  """Runs RFDETR on a single image and returns the post-filter count.

  Applies the contained-sub-mask filter, contained-box merge, and
  edge-visibility filter. The score threshold is NOT applied here.

  Args:
      image: PIL RGB image (already resized).
      model: RFDETR model instance.
      config: Validated pipeline configuration.

  Returns:
      The number of detections remaining after all post-processing filters.
      ``0`` when RFDETR returns no detections.
  """
  image_width, image_height = image.size
  detections = model.predict(image, threshold=config.predict_threshold)
  state = detection_utils.convert_rfdetr_detections_to_state(
      detections, image_height=image_height, image_width=image_width
  )

  if state["scores"].shape[0] == 0:
    return 0

  state = detection_utils.filter_contained_sub_masks(
      state, containment_threshold=config.containment_threshold
  )
  state = detection_utils.merge_contained_boxes(
      state, containment_threshold=config.merge_containment_threshold
  )
  #   state = detection_utils.get_valid_bottle_indices(state)

  return int(state["scores"].shape[0])


# ── Per-dataset filter pass ───────────────────────────────────────────────────


def filter_dataset_images(
    dataset_name: str,
    images_dir: str,
    config: config_loader.PipelineConfig,
    model: Any,
) -> tuple[int, int, int]:
  """Walks a dataset's images folder and moves sparse images to rejected.

  Each rejected image is mirrored under ``config.rejected_dir`` keeping its
  full path relative to the pipeline root, e.g.
  ``<root_dir>/<dataset>/images/foo.jpg`` ->
  ``<rejected_dir>/<dataset>/images/foo.jpg``.

  Args:
      dataset_name: Name of the dataset (used for log labels).
      images_dir: Path to the dataset's input images folder.
      config: Validated pipeline configuration.
      model: RFDETR model instance.

  Returns:
      A tuple of ``(rejected_count, skipped_count, total_count)``.
  """
  image_paths = gather_image_paths(images_dir)
  logger.info("[%s] Found %d images", dataset_name, len(image_paths))

  rejected_count = 0
  skipped_count = 0

  for image_path in tqdm.tqdm(image_paths, desc=dataset_name):
    try:
      with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    except Exception as error:  # pylint: disable=broad-exception-caught
      logger.warning(
          "[SKIP] %s: could not open image \u2014 %s", image_path, error
      )
      skipped_count += 1
      continue

    image = detection_utils.resize_image_for_inference(
        image, max_short_side=config.max_short_side
    )

    try:
      final_count = count_detections(image, model, config)
    except Exception as error:  # pylint: disable=broad-exception-caught
      logger.warning("[SKIP] %s: inference failed \u2014 %s", image_path, error)
      skipped_count += 1
      del image
      gc.collect()
      torch.cuda.empty_cache()
      continue

    if final_count < config.min_detections:
      move_to_rejected(image_path, config.root_dir, config.rejected_dir)
      rejected_count += 1

    del image
    gc.collect()
    torch.cuda.empty_cache()

  kept_count = len(image_paths) - rejected_count - skipped_count
  logger.info(
      "[%s] Done. Rejected: %d, Skipped: %d, Kept: %d",
      dataset_name,
      rejected_count,
      skipped_count,
      kept_count,
  )

  return (rejected_count, skipped_count, len(image_paths))


# ── Main ──────────────────────────────────────────────────────────────────────


def format_elapsed_time(elapsed_seconds: float) -> str:
  """Formats elapsed seconds into a human-readable string.

  Args:
      elapsed_seconds: Total elapsed time in seconds.

  Returns:
      A formatted string like ``'2h 15m 30s'``.
  """
  hours = int(elapsed_seconds // 3600)
  minutes = int((elapsed_seconds % 3600) // 60)
  seconds = int(elapsed_seconds % 60)
  return f"{hours}h {minutes}m {seconds}s"


def main() -> None:
  """Entry point: filters sparse images out of every dataset's input folder."""
  configure_logging()

  config = config_loader.load_config(CONFIG_PATH)
  os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

  validate_rejected_dir(config.rejected_dir)
  dataset_directories = discover_dataset_directories(config.root_dir)
  validated_datasets = validate_dataset_paths(
      dataset_directories, config.input_images_folder_name
  )

  dataset_names = [name for name, _ in validated_datasets]
  logger.info("Root directory:    %s", config.root_dir)
  logger.info("Rejected output:   %s", config.rejected_dir)
  logger.info("Found %d dataset(s): %s", len(validated_datasets), dataset_names)
  logger.info("Min detections:    %d", config.min_detections)
  logger.info("Predict threshold: %.3f", config.predict_threshold)

  os.makedirs(config.rejected_dir, exist_ok=True)

  model = build_rfdetr_model(
      os.path.join(os.getcwd(), config.rfdetr_checkpoint_path)
  )

  overall_rejected = 0
  overall_skipped = 0
  overall_total = 0
  wall_start = time.perf_counter()

  for dataset_name, images_dir in validated_datasets:
    rejected, skipped, total = filter_dataset_images(
        dataset_name, images_dir, config, model
    )
    overall_rejected += rejected
    overall_skipped += skipped
    overall_total += total

  elapsed = time.perf_counter() - wall_start
  overall_kept = overall_total - overall_rejected - overall_skipped
  logger.info(
      "All datasets filtered in %s. Total: %d, Kept: %d, Rejected: %d, "
      "Skipped: %d",
      format_elapsed_time(elapsed),
      overall_total,
      overall_kept,
      overall_rejected,
      overall_skipped,
  )


if __name__ == "__main__":
  main()
