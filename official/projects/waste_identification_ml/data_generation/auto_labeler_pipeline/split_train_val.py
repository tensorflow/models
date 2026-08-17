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

"""Split images into train and validation sets across multiple datasets.

Discovers dataset subfolders under a single root directory. For each
dataset, reads its input images folder, sorts images naturally, filters to
keep every Nth image, and splits the result into ``train/`` and ``val/``
under a sibling ``train_val_images/`` folder.

If a dataset's input images folder contains subfolders, each subfolder is
processed independently and its images are copied flat into the dataset's
shared ``train/`` and ``val/`` folders. If the input folder contains loose
files, they are processed directly.

Expected layout under ``config.root_dir``::

    root_dir/
    ├── dataset_a/
    │   └── images/
    └── dataset_b/
        └── images/

Produces, for each dataset::

    root_dir/
    └── dataset_a/
        ├── images/                ← unchanged input
        └── train_val_images/      ← created
            ├── train/
            └── val/
"""

import os
import shutil

import natsort

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader


# Resolve config.yaml relative to this script file so the script runs
# correctly regardless of the caller's current working directory.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)


# ── Dataset discovery and validation ────────────────────────────────────────


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
    dataset_directories: list[tuple[str, str]],
    input_images_folder_name: str,
    train_val_folder_name: str,
) -> list[tuple[str, str, str]]:
  """Validates that each dataset has the expected input/output layout.

  Performs all checks upfront before any copying starts, so misconfigured
  datasets are reported immediately.

  Args:
      dataset_directories: List of ``(dataset_name, dataset_path)`` tuples.
      input_images_folder_name: Name of the input images subfolder.
      train_val_folder_name: Name of the train/val output subfolder.

  Returns:
      A list of ``(dataset_name, source_folder, output_folder)`` tuples ready
      for processing.

  Raises:
      FileNotFoundError: If a dataset is missing its input folder.
      FileExistsError: If a dataset's output folder already exists.
  """
  validated = []
  for dataset_name, dataset_path in dataset_directories:
    source_folder = os.path.join(dataset_path, input_images_folder_name)
    output_folder = os.path.join(dataset_path, train_val_folder_name)

    if not os.path.isdir(source_folder):
      raise FileNotFoundError(
          f"Dataset {dataset_name!r} is missing required input folder: "
          f"{source_folder}"
      )

    if os.path.exists(output_folder):
      raise FileExistsError(
          f"Dataset {dataset_name!r} already has an output folder: "
          f"{output_folder}. Remove or rename it before re-running."
      )

    validated.append((dataset_name, source_folder, output_folder))

  return validated


# ── Image discovery and filtering ───────────────────────────────────────────


def get_subfolder_names(source_folder: str) -> list[str]:
  """Returns a sorted list of subfolder names in the source folder.

  Args:
      source_folder: Path to the root source folder.

  Returns:
      A sorted list of subfolder names. Empty list if none found.

  Raises:
      FileNotFoundError: If the source folder does not exist.
  """
  if not os.path.isdir(source_folder):
    raise FileNotFoundError(f"Source folder not found: {source_folder}")

  return [
      name
      for name in sorted(os.listdir(source_folder))
      if os.path.isdir(os.path.join(source_folder, name))
  ]


def get_sorted_image_names(folder_path: str) -> list[str]:
  """Returns a naturally sorted list of image file names from a folder.

  Args:
      folder_path: Path to the folder containing images.

  Returns:
      A naturally sorted list of image file names.
  """
  image_names = [
      file_name
      for file_name in os.listdir(folder_path)
      if os.path.isfile(os.path.join(folder_path, file_name))
      and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
  ]
  return natsort.natsorted(image_names)


def filter_every_nth_image(
    sorted_image_names: list[str], keep_every_nth: int
) -> list[str]:
  """Keeps every Nth image starting from index 0.

  For ``keep_every_nth=3``, keeps indices 0, 3, 6, 9, ... and skips the rest.

  Args:
      sorted_image_names: Naturally sorted list of image file names.
      keep_every_nth: Interval for keeping images (e.g. 3 means keep every
        3rd image).

  Returns:
      A filtered list of image file names.
  """
  return [
      file_name
      for index, file_name in enumerate(sorted_image_names)
      if index % keep_every_nth == 0
  ]


# ── Copy helpers ────────────────────────────────────────────────────────────


def check_for_duplicates(
    file_names: list[str], destination_folder: str
) -> None:
  """Checks if any files already exist in the destination folder.

  Args:
      file_names: List of file names to check.
      destination_folder: Path to the destination folder.

  Raises:
      FileExistsError: If any file names conflict with existing files.
  """
  conflicting_files = [
      name
      for name in file_names
      if os.path.exists(os.path.join(destination_folder, name))
  ]

  if conflicting_files:
    conflict_list = "\n  ".join(conflicting_files)
    raise FileExistsError(
        f"Duplicate files found in '{destination_folder}':\n  {conflict_list}"
    )


def copy_files(
    file_names: list[str],
    source_folder: str,
    destination_folder: str,
) -> None:
  """Copies files from source to destination folder.

  Args:
      file_names: List of file names to copy.
      source_folder: Path to the source folder.
      destination_folder: Path to the destination folder.
  """
  for file_name in file_names:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)
    shutil.copy2(source_path, destination_path)


# ── Per-folder pipeline ─────────────────────────────────────────────────────


def process_folder(
    folder_path: str,
    folder_label: str,
    train_folder: str,
    val_folder: str,
    keep_every_nth: int,
    train_ratio: float,
) -> tuple[str, str, list[str], list[str]] | None:
  """Sorts, filters, splits, and checks duplicates for a single folder.

  ``train_ratio`` is treated as the fraction going to the VAL split; train
  receives the remaining majority. The name is kept for backward
  compatibility with existing configs.

  Args:
      folder_path: Path to the folder containing images.
      folder_label: Display name for logging.
      train_folder: Path to the train output folder.
      val_folder: Path to the val output folder.
      keep_every_nth: Interval for keeping images.
      train_ratio: Fraction assigned to the val split (see note above).

  Returns:
      A tuple of ``(folder_label, folder_path, train_image_names,
      val_image_names)``, or ``None`` if no images were found.
  """
  sorted_image_names = get_sorted_image_names(folder_path)

  if not sorted_image_names:
    print(f"\n[{folder_label}] No images found, skipping.")
    return None

  filtered_image_names = filter_every_nth_image(
      sorted_image_names, keep_every_nth
  )

  print(
      f"[{folder_label}] {len(sorted_image_names)} total, "
      f"{len(filtered_image_names)} after keeping every "
      f"{keep_every_nth}rd image"
  )

  # train_ratio is treated as the val fraction; train gets the majority.
  val_size = int(len(filtered_image_names) * train_ratio)
  val_image_names = filtered_image_names[:val_size]
  train_image_names = filtered_image_names[val_size:]

  check_for_duplicates(train_image_names, train_folder)
  check_for_duplicates(val_image_names, val_folder)

  return (folder_label, folder_path, train_image_names, val_image_names)


# ── Per-dataset pipeline ────────────────────────────────────────────────────


def process_dataset(
    dataset_name: str,
    source_folder: str,
    output_folder: str,
    train_split_name: str,
    val_split_name: str,
    keep_every_nth: int,
    train_ratio: float,
) -> tuple[int, int]:
  """Splits one dataset's images into train and val.

  Handles both flat source folders and source folders with subfolders.

  Args:
      dataset_name: Name of the dataset (used for log prefixes).
      source_folder: Path to the dataset's input folder.
      output_folder: Path to the dataset's train/val output folder.
      train_split_name: Name of the train split subfolder.
      val_split_name: Name of the val split subfolder.
      keep_every_nth: Interval for keeping images.
      train_ratio: Fraction assigned to the val split.

  Returns:
      A tuple of ``(train_count, val_count)`` for this dataset.

  Raises:
      ValueError: If no images are found in any folder.
  """
  print(f"\n=== Dataset: {dataset_name} ===")
  print(f"Source: {source_folder}")
  print(f"Output: {output_folder}")

  train_folder = os.path.join(output_folder, train_split_name)
  val_folder = os.path.join(output_folder, val_split_name)
  os.makedirs(train_folder, exist_ok=True)
  os.makedirs(val_folder, exist_ok=True)

  subfolder_names = get_subfolder_names(source_folder)

  # Build list of folders to process within this dataset.
  if subfolder_names:
    print(f"Subfolders found: {len(subfolder_names)}")
    folders_to_process = [
        (os.path.join(source_folder, name), name) for name in subfolder_names
    ]
  else:
    print("No subfolders found. Processing source folder directly.")
    folder_label = os.path.basename(source_folder.rstrip(os.sep))
    folders_to_process = [(source_folder, folder_label)]

  # First pass: check all duplicates before copying anything.
  all_splits = []
  for folder_path, folder_label in folders_to_process:
    result = process_folder(
        folder_path,
        folder_label,
        train_folder,
        val_folder,
        keep_every_nth,
        train_ratio,
    )
    if result is not None:
      all_splits.append(result)

  if not all_splits:
    raise ValueError(
        f"No images found in any folder for dataset {dataset_name!r}."
    )

  # Second pass: copy files.
  dataset_train_count = 0
  dataset_val_count = 0

  for split in all_splits:
    folder_label, folder_path, train_image_names, val_image_names = split
    print(
        f"[{folder_label}] {len(train_image_names)} train, "
        f"{len(val_image_names)} val"
    )

    copy_files(train_image_names, folder_path, train_folder)
    copy_files(val_image_names, folder_path, val_folder)

    dataset_train_count += len(train_image_names)
    dataset_val_count += len(val_image_names)

  print(
      f"\n=== Dataset {dataset_name} done. "
      f"Train: {dataset_train_count}, Val: {dataset_val_count} ==="
  )

  return (dataset_train_count, dataset_val_count)


# ── Main ────────────────────────────────────────────────────────────────────


def main(config_path: str = CONFIG_PATH) -> None:
  """Entry point: discovers datasets under root_dir and splits each."""
  config = config_loader.load_config(config_path)

  dataset_directories = discover_dataset_directories(config.root_dir)
  validated_datasets = validate_dataset_paths(
      dataset_directories,
      config.input_images_folder_name,
      config.train_val_folder_name,
  )

  dataset_names = [name for name, _, _ in validated_datasets]
  print(f"Root directory: {config.root_dir}")
  print(f"Found {len(validated_datasets)} dataset(s): {dataset_names}")
  print(f"Val ratio (train_ratio in YAML): {config.train_ratio}")
  print(f"Keep every Nth image: {config.keep_every_nth}")

  overall_train_count = 0
  overall_val_count = 0

  for dataset_name, source_folder, output_folder in validated_datasets:
    dataset_train_count, dataset_val_count = process_dataset(
        dataset_name,
        source_folder,
        output_folder,
        config.train_split_name,
        config.val_split_name,
        config.keep_every_nth,
        config.train_ratio,
    )
    overall_train_count += dataset_train_count
    overall_val_count += dataset_val_count

  print("\n" + "=" * 60)
  print("All datasets processed.")
  print(f"Overall train: {overall_train_count}")
  print(f"Overall val:   {overall_val_count}")


if __name__ == "__main__":
  main()
