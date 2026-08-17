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

"""Apply augmentations to images in the train split of a classifier dataset.

Augmentation is applied only to the ``train/`` split. The ``val/`` split is
intentionally skipped.

Expected folder structure (output of ``segmentation.py``)::

    config.classifier_dir/
    ├── train/
    │   ├── class_a/   ← processed
    │   ├── class_b/   ← processed
    │   └── ...
    └── val/           ← not processed

Augmented images are saved alongside the originals inside each class
folder. Originals are preserved.

Which augmentations are applied is controlled by
``config.active_augmentations``. Output filenames always follow the
canonical augmentation order defined in ``config_loader``, so runs are
deterministic regardless of YAML ordering.
"""

import argparse
import os

import PIL.Image
import PIL.ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import tqdm

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
OUTPUT_EXTENSION = ".jpg"


# ── Individual augmentation functions ────────────────────────────────────────


def apply_fixed_rotation(
    image: PIL.Image.Image,
    degrees: float,
    fill_color: tuple[int, int, int],
) -> PIL.Image.Image:
  """Rotates an image by a fixed angle with the given fill color.

  Uses torchvision's functional rotate to match the rotation convention of
  ``T.RandomRotation``. Rotation angle convention: positive = clockwise.

  Args:
      image: PIL Image to rotate.
      degrees: Fixed rotation angle in degrees.
      fill_color: RGB tuple used to fill empty corners after rotation.

  Returns:
      The rotated PIL Image.
  """
  fill_color_as_list = list(fill_color)
  return TF.rotate(image, angle=degrees, fill=fill_color_as_list)


def apply_vertical_flip(image: PIL.Image.Image) -> PIL.Image.Image:
  """Flips an image vertically.

  Args:
      image: PIL Image to flip.

  Returns:
      The vertically flipped PIL Image.
  """
  transform = T.RandomVerticalFlip(p=1.0)
  return transform(image)


def apply_horizontal_flip(image: PIL.Image.Image) -> PIL.Image.Image:
  """Flips an image horizontally.

  Args:
      image: PIL Image to flip.

  Returns:
      The horizontally flipped PIL Image.
  """
  transform = T.RandomHorizontalFlip(p=1.0)
  return transform(image)


def apply_gaussian_blur(image: PIL.Image.Image) -> PIL.Image.Image:
  """Applies gaussian blur to an image.

  Args:
      image: PIL Image to blur.

  Returns:
      The blurred PIL Image.
  """
  blur_transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(2, 20))
  return blur_transform(image)


def apply_add_noise(
    image: PIL.Image.Image, noise_factor: float
) -> PIL.Image.Image:
  """Adds uniform noise to an image.

  Args:
      image: PIL Image to add noise to.
      noise_factor: Scalar controlling the magnitude of noise.

  Returns:
      The noisy PIL Image.
  """
  image_tensor = T.ToTensor()(image)
  noisy_tensor = image_tensor + torch.rand_like(image_tensor) * noise_factor
  noisy_tensor = torch.clip(noisy_tensor, 0.0, 1.0)
  return T.ToPILImage()(noisy_tensor)


def apply_color_jitter(image: PIL.Image.Image) -> PIL.Image.Image:
  """Applies brightness color jitter to an image.

  Args:
      image: PIL Image to apply color jitter to.

  Returns:
      The color-jittered PIL Image.
  """
  jitter_transform = T.ColorJitter(brightness=(0.1, 1.8))
  return jitter_transform(image)


def build_single_augmentation(
    image: PIL.Image.Image,
    augmentation_name: str,
    rotation_fill_color: tuple[int, int, int],
) -> PIL.Image.Image:
  """Builds one augmented image by name.

  Args:
      image: Original PIL Image.
      augmentation_name: Name of the augmentation to apply. Must be one of
        the entries in ``config_loader.CANONICAL_AUGMENTATION_ORDER``.
      rotation_fill_color: RGB fill color used to pad rotated images.

  Returns:
      The augmented PIL Image.

  Raises:
      ValueError: If ``augmentation_name`` is not recognised.
  """
  if augmentation_name == "vflip":
    return apply_vertical_flip(image)
  if augmentation_name == "hflip":
    return apply_horizontal_flip(image)
  if augmentation_name == "rot45":
    return apply_fixed_rotation(image, 45, rotation_fill_color)
  if augmentation_name == "rot65":
    return apply_fixed_rotation(image, 65, rotation_fill_color)
  if augmentation_name == "rot90":
    return apply_fixed_rotation(image, 90, rotation_fill_color)
  if augmentation_name == "blur":
    return apply_gaussian_blur(image)
  if augmentation_name == "noise03":
    return apply_add_noise(image, 0.3)
  if augmentation_name == "noise06":
    return apply_add_noise(image, 0.6)
  if augmentation_name == "cjitter":
    return apply_color_jitter(image)
  raise ValueError(f"Unknown augmentation name: {augmentation_name!r}")


def build_augmented_images(
    image: PIL.Image.Image,
    augmentations_to_apply: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
) -> dict[str, PIL.Image.Image]:
  """Creates a dictionary of augmented images keyed by suffix.

  The loader has already reordered ``augmentations_to_apply`` into
  canonical order, so iterating over it directly is enough to make the
  on-disk output deterministic.

  Args:
      image: Original PIL Image.
      augmentations_to_apply: Sequence of augmentation names to apply,
        already in canonical order.
      rotation_fill_color: RGB fill color used to pad rotated images.

  Returns:
      A dictionary mapping suffix string to augmented PIL Image.
  """
  augmented_images = {}
  for augmentation_name in augmentations_to_apply:
    augmented_images[augmentation_name] = build_single_augmentation(
        image, augmentation_name, rotation_fill_color
    )
  return augmented_images


# ── Saving helpers ───────────────────────────────────────────────────────────


def save_augmented_images(
    augmented_images: dict[str, PIL.Image.Image],
    folder_path: str,
    original_base_name: str,
) -> None:
  """Saves all augmented images to disk.

  Args:
      augmented_images: Dictionary mapping suffix to PIL Image.
      folder_path: Destination folder path.
      original_base_name: Base filename (without extension) of the original.
  """
  for suffix, augmented_image in augmented_images.items():
    output_file_name = f"{original_base_name}_{suffix}{OUTPUT_EXTENSION}"
    output_path = os.path.join(folder_path, output_file_name)

    if augmented_image.mode != "RGB":
      augmented_image = augmented_image.convert("RGB")

    augmented_image.save(output_path, "JPEG", quality=95)


# ── Validation ───────────────────────────────────────────────────────────────


def validate_train_split_exists(
    classifier_dir: str, train_split_name: str
) -> str:
  """Ensures the train split exists under the classifier directory.

  Args:
      classifier_dir: Path to the classifier dataset root.
      train_split_name: Name of the train split folder (e.g. ``'train'``).

  Returns:
      The path to the train split folder.

  Raises:
      FileNotFoundError: If the classifier dir or train split is missing.
  """
  if not os.path.isdir(classifier_dir):
    raise FileNotFoundError(
        f"Classifier directory does not exist: {classifier_dir}"
    )

  train_dir = os.path.join(classifier_dir, train_split_name)
  if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Train split folder is missing: {train_dir}")

  return train_dir


def is_augmented_filename(file_name: str) -> bool:
  """Checks if a filename already corresponds to an augmented image.

  Scans for every suffix in ``config_loader.CANONICAL_AUGMENTATION_ORDER``,
  not just the currently active set, so leftover augmentations from a
  previous run with different settings are still detected.

  Args:
      file_name: Image file name.

  Returns:
      ``True`` if the filename ends with any known augmentation suffix.
  """
  base_name = os.path.splitext(file_name)[0]
  for augmentation_name in config_loader.CANONICAL_AUGMENTATION_ORDER:
    if base_name.endswith(f"_{augmentation_name}"):
      return True
  return False


def discover_class_folders(train_dir: str) -> list[tuple[str, str]]:
  """Returns the sorted list of class subfolders under the train split.

  Args:
      train_dir: Path to the train split folder.

  Returns:
      A sorted list of ``(class_name, class_path)`` tuples.

  Raises:
      ValueError: If no class subfolders are found.
  """
  class_entries = sorted(
      [entry for entry in os.scandir(train_dir) if entry.is_dir()],
      key=lambda entry: entry.name,
  )
  if not class_entries:
    raise ValueError(f"No class subfolders found under: {train_dir}")
  return [(entry.name, entry.path) for entry in class_entries]


def discover_target_folders(
    train_dir: str, crop_variants: tuple[str, ...]
) -> list[tuple[str, str]]:
  """Returns the sorted list of target subfolders containing images to augment.

  When ``crop_variants`` has a single entry, images are stored directly in
  each class folder under ``train_dir``. When ``crop_variants`` has multiple
  entries, each class folder contains one subdirectory per variant.

  Args:
      train_dir: Path to the train split folder.
      crop_variants: Tuple of active crop variants from the config.

  Returns:
      A sorted list of ``(target_name, target_path)`` tuples.

  Raises:
      ValueError: If no class subfolders are found.
  """
  class_entries = sorted(
      [entry for entry in os.scandir(train_dir) if entry.is_dir()],
      key=lambda entry: entry.name,
  )
  if not class_entries:
    raise ValueError(f"No class subfolders found under: {train_dir}")

  target_folders = []
  for entry in class_entries:
    if len(crop_variants) > 1:
      has_variant_subdirs = False
      for variant in crop_variants:
        variant_path = os.path.join(entry.path, variant)
        if os.path.isdir(variant_path):
          target_folders.append((f"{entry.name}/{variant}", variant_path))
          has_variant_subdirs = True
      if not has_variant_subdirs:
        target_folders.append((entry.name, entry.path))
    else:
      target_folders.append((entry.name, entry.path))

  return target_folders


def find_pre_existing_augmentations(class_folder: str) -> list[str]:
  """Returns names of any pre-existing augmented files in a class folder.

  Args:
      class_folder: Path to a class folder under the train split.

  Returns:
      A sorted list of augmented file names found in the folder.
  """
  return sorted(
      file_name
      for file_name in os.listdir(class_folder)
      if file_name.lower().endswith(IMAGE_EXTENSIONS)
      and is_augmented_filename(file_name)
  )


def validate_no_pre_existing_augmentations(
    class_folders: list[tuple[str, str]],
) -> None:
  """Stops execution if any class folder already contains augmented files.

  Performs the check across all class folders upfront so misconfigured runs
  are detected before any augmentation starts.

  Args:
      class_folders: List of ``(class_name, class_path)`` tuples.

  Raises:
      FileExistsError: If any class folder contains augmented files.
  """
  folders_with_augmentations = []
  for class_name, class_path in class_folders:
    existing_augmented_files = find_pre_existing_augmentations(class_path)
    if existing_augmented_files:
      sample = existing_augmented_files[:5]
      extra = len(existing_augmented_files) - len(sample)
      sample_text = ", ".join(sample)
      if extra > 0:
        sample_text += f", ... (+{extra} more)"
      folders_with_augmentations.append(
          f"  - {class_name}: {len(existing_augmented_files)} files "
          f"({sample_text})"
      )

  if folders_with_augmentations:
    details = "\n".join(folders_with_augmentations)
    raise FileExistsError(
        "Augmented files already exist in the following class folders:\n"
        f"{details}\n"
        "Remove them before re-running, or run on a fresh classifier dataset."
    )


# ── Per-folder pipeline ──────────────────────────────────────────────────────


def process_class_folder(
    class_name: str,
    class_folder: str,
    augmentations_to_apply: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
) -> None:
  """Applies augmentations to all images in a class folder.

  Args:
      class_name: Name of the class (used for log labels).
      class_folder: Path to the class folder.
      augmentations_to_apply: Sequence of augmentation names to apply.
      rotation_fill_color: RGB fill color used to pad rotated images.
  """
  original_image_names = sorted(
      file_name
      for file_name in os.listdir(class_folder)
      if file_name.lower().endswith(IMAGE_EXTENSIONS)
      and not is_augmented_filename(file_name)
  )

  total_images = len(original_image_names)
  print(f"\n[{class_name}] processing {total_images} image(s)")

  if total_images == 0:
    return

  progress_bar = tqdm.tqdm(
      original_image_names,
      total=total_images,
      desc=class_name,
      unit="img",
  )

  for image_name in progress_bar:
    image_path = os.path.join(class_folder, image_name)
    original_base_name = os.path.splitext(image_name)[0]

    with PIL.Image.open(image_path) as image:
      image = PIL.ImageOps.exif_transpose(image)
      image = image.convert("RGB")
      augmented_images = build_augmented_images(
          image, augmentations_to_apply, rotation_fill_color
      )
      save_augmented_images(
          augmented_images, class_folder, original_base_name
      )


# ── Main ─────────────────────────────────────────────────────────────────────


def main(config_path: str = _DEFAULT_CONFIG_PATH) -> None:
  """Entry point: validates inputs and augments every train class folder.

  Args:
      config_path: Path to the YAML configuration file.
  """
  config = config_loader.load_config(config_path)

  train_dir = validate_train_split_exists(
      config.classifier_dir, config.train_split_name
  )
  target_folders = discover_target_folders(train_dir, config.crop_variants)
  validate_no_pre_existing_augmentations(target_folders)

  target_names = [name for name, _ in target_folders]
  active_augmentations = config.active_augmentations
  print(f"Classifier directory: {config.classifier_dir}")
  print(f"Train split:          {train_dir}")
  print(f"Found {len(target_folders)} target folder(s): {target_names}")
  print(f"Active augmentations: {list(active_augmentations)}")

  for folder_name, folder_path in target_folders:
    process_class_folder(
        folder_name,
        folder_path,
        active_augmentations,
        config.rotation_fill_color,
    )

  print("\nDone.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Apply augmentations to images in the train split."
  )
  parser.add_argument(
      "--config",
      type=str,
      default=_DEFAULT_CONFIG_PATH,
      help="Path to the config.yaml file.",
  )
  args = parser.parse_args()
  main(args.config)
