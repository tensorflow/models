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

"""SAM3 sparse-image filter: move images with fewer than ``min_detections``.

For each dataset under ``config.root_dir``, walks its input images folder
recursively, runs SAM3 on every supported image, and moves any image whose
post-filter detection count is below ``config.min_detections`` into the
sibling rejected directory at ``config.rejected_dir``, preserving each
image's full relative path from the pipeline root.

The same contained-sub-mask and edge-visibility filters as the batch
segmentation pipeline are applied. The score threshold is NOT applied here
-- this stage is intended to discard images that contain too few objects to
be useful for downstream training, regardless of confidence scores.

No crops are saved. Images that pass the filter are left in place; only
rejected images are moved.

Expected input layout under ``config.root_dir``::

    root_dir/
    ├── dataset_a/
    │   └── images/                    ← walked recursively
    │       ├── foo.jpg
    │       ├── bar.jpg
    │       └── nested_subfolder/
    │           └── baz.jpg
    └── dataset_b/
        └── images/
            └── qux.jpg

The subfolder name (``images``) is set by
``config.input_images_folder_name``. Any subdirectory structure inside it
is walked recursively; only files with extensions in ``IMAGE_EXTENSIONS``
are considered.

Output layout after the stage runs::

    root_dir/                          ← unchanged, minus rejected files
    ├── dataset_a/
    │   └── images/
    │       ├── foo.jpg                ← kept (had >= min_detections)
    │       └── nested_subfolder/
    │           └── baz.jpg            ← kept
    └── dataset_b/
        └── images/
            └── qux.jpg                ← kept

    root_dir_empty/                    ← created; mirrors relative paths
    └── dataset_a/
        └── images/
            └── bar.jpg                ← moved here (below threshold)

The sibling ``<root_dir>_empty`` directory is derived by ``config_loader``
and exposed as ``config.rejected_dir``. This stage refuses to run if that
directory already exists, so a re-run cannot silently merge into a
previous run's output.
"""

import gc
import os
import shutil
import time
from typing import Any

import natsort
from PIL import Image
import torch
import tqdm

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import sam3_inference_utils

try:
  # pylint: disable=g-import-not-at-top
  from sam3 import model_builder as sam3_model_builder  # type: ignore[import-error]
  from sam3.model import sam3_image_processor  # type: ignore[import-error]
  # pylint: enable=g-import-not-at-top
except ImportError:
  sam3_model_builder = None
  sam3_image_processor = None


# Resolve config.yaml relative to this script file so the script runs
# correctly regardless of the caller's current working directory.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)
IMAGE_EXTENSIONS = frozenset([".jpg", ".jpeg", ".png"])
PACKETS_PROMPT_NAME = "packets"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model setup ───────────────────────────────────────────────────────────────


def build_sam3_processor(
    detection_config: config_loader.DetectionConfig,
    sam3_checkpoint_path: str,
) -> tuple[Any, Any]:
  """Builds the SAM3 model and its processor.

  Args:
      detection_config: Validated detection thresholds for the active prompt.
      sam3_checkpoint_path: Absolute path to the SAM3 checkpoint.

  Returns:
      A tuple of ``(sam3_model, sam3_processor)``.

  Raises:
      ImportError: If the ``sam3`` package is not installed or available on
          the Python path.
  """
  if sam3_model_builder is None or sam3_image_processor is None:
    raise ImportError(
        "The 'sam3' package is not installed or available in python path. "
        "Cannot build the SAM3 processor."
    )
  sam3_model = sam3_model_builder.build_sam3_image_model(
      checkpoint_path=sam3_checkpoint_path
  )
  sam3_model.to(device=DEVICE)
  sam3_processor = sam3_image_processor.Sam3Processor(
      sam3_model,
      confidence_threshold=detection_config.confidence_threshold,
  )
  return sam3_model, sam3_processor


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
    processor: Any,
    detection_config: config_loader.DetectionConfig,
    prompt: str,
) -> int:
  """Runs SAM3 on a single image and returns its post-filter detection count.

  Applies the same contained-sub-mask and edge-visibility filters as the
  batch pipeline. The score threshold is NOT applied here.

  Args:
      image: PIL RGB image (already resized).
      processor: SAM3 processor instance.
      detection_config: Validated detection thresholds for this prompt.
      prompt: Text prompt for detection.

  Returns:
      Integer count of detections after contained/edge filtering.
  """
  state = sam3_inference_utils.run_inference(processor, image, prompt)

  if not state["scores"].tolist():
    return 0

  state = sam3_inference_utils.filter_contained_sub_masks(
      state, containment_threshold=detection_config.containment_threshold
  )
  if prompt == PACKETS_PROMPT_NAME:
    state = sam3_inference_utils.merge_contained_boxes(state)
  state = sam3_inference_utils.get_valid_bottle_indices(state)

  return int(state["scores"].shape[0])


# ── Per-dataset filter pass ───────────────────────────────────────────────────


def filter_dataset_images(
    dataset_name: str,
    images_dir: str,
    root_dir: str,
    rejected_dir: str,
    processor: Any,
    detection_config: config_loader.DetectionConfig,
    prompt: str,
    min_detections: int,
) -> tuple[int, int, int]:
  """Walks a dataset's images folder and moves sparse images to rejected.

  Each rejected image is mirrored under ``rejected_dir`` keeping its full
  path relative to the pipeline root, e.g.
  ``<root_dir>/<dataset>/images/foo.jpg`` ->
  ``<rejected_dir>/<dataset>/images/foo.jpg``.

  Args:
      dataset_name: Name of the dataset (used for log labels).
      images_dir: Path to the dataset's input images folder.
      root_dir: Pipeline root directory (used to compute relative paths).
      rejected_dir: Path to the rejected root directory.
      processor: SAM3 processor instance.
      detection_config: Validated detection thresholds for this prompt.
      prompt: Text prompt for detection.
      min_detections: Threshold below which images are moved to rejected.

  Returns:
      A tuple of ``(rejected_count, skipped_count, total_count)``.
  """
  image_paths = gather_image_paths(images_dir)
  print(f"\n[{dataset_name}] Found {len(image_paths)} images")

  rejected_count = 0
  skipped_count = 0

  for image_path in tqdm.tqdm(image_paths, desc=dataset_name):
    try:
      with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    except Exception as error:  # pylint: disable=broad-exception-caught
      print(f"  [SKIP] {image_path}: could not open image — {error}")
      skipped_count += 1
      continue

    image = sam3_inference_utils.resize_image_for_inference(
        image, max_short_side=detection_config.max_short_side
    )

    try:
      detection_count = count_detections(
          image, processor, detection_config, prompt
      )
    except Exception as error:  # pylint: disable=broad-exception-caught
      print(f"  [SKIP] {image_path}: inference failed — {error}")
      skipped_count += 1
      del image
      gc.collect()
      torch.cuda.empty_cache()
      continue

    if detection_count < min_detections:
      move_to_rejected(image_path, root_dir, rejected_dir)
      rejected_count += 1

    del image
    gc.collect()
    torch.cuda.empty_cache()

  kept_count = len(image_paths) - rejected_count - skipped_count
  print(
      f"[{dataset_name}] Done. "
      f"Rejected: {rejected_count}, Skipped: {skipped_count}, "
      f"Kept: {kept_count}"
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
  config = config_loader.load_config(CONFIG_PATH)
  os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

  validate_rejected_dir(config.rejected_dir)
  dataset_directories = discover_dataset_directories(config.root_dir)
  validated_datasets = validate_dataset_paths(
      dataset_directories, config.input_images_folder_name
  )

  dataset_names = [name for name, _ in validated_datasets]
  print(f"Root directory:    {config.root_dir}")
  print(f"Rejected output:   {config.rejected_dir}")
  print(f"Found {len(validated_datasets)} dataset(s): {dataset_names}")
  print(f"Min detections:    {config.min_detections}")
  print(f"Prompt:            {config.prompt_to_detect!r}")

  os.makedirs(config.rejected_dir, exist_ok=True)

  detection_config = config.active_detection
  _, processor = build_sam3_processor(
      detection_config, config.sam3_checkpoint_path
  )

  overall_rejected = 0
  overall_skipped = 0
  overall_total = 0
  wall_start = time.perf_counter()

  for dataset_name, images_dir in validated_datasets:
    rejected, skipped, total = filter_dataset_images(
        dataset_name,
        images_dir,
        config.root_dir,
        config.rejected_dir,
        processor,
        detection_config,
        config.prompt_to_detect,
        config.min_detections,
    )
    overall_rejected += rejected
    overall_skipped += skipped
    overall_total += total

  elapsed = time.perf_counter() - wall_start
  overall_kept = overall_total - overall_rejected - overall_skipped
  print(
      f"\nAll datasets filtered in {format_elapsed_time(elapsed)}.\n"
      f"  Total:    {overall_total}\n"
      f"  Kept:     {overall_kept}\n"
      f"  Rejected: {overall_rejected}\n"
      f"  Skipped:  {overall_skipped}"
  )


if __name__ == "__main__":
  main()
