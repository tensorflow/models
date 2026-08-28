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

"""Apply foreground-only augmentations to the train split of a classifier dataset.

Augmentation is applied only to the ``train/`` split. The ``val/`` split is
intentionally skipped.

Expected folder structure (output of ``segmentation.py``)::

    config.classifier_dir/
    ├── train/
    │   ├── class_a/
    │   │   ├── image_001_0.jpg
    │   │   ├── image_001_0_mask.png     ← sidecar written by segmentation
    │   │   └── ...
    │   ├── class_b/
    │   └── ...
    └── val/                              ← not processed, no masks written

Every augmentation is applied to the foreground object only. Background
pixels in the output are a solid color, determined by the crop variant of
the image being augmented:

  * ``raw``                       -> black.
  * ``black_background``          -> black.
  * ``imagenet_mean_background``  -> ``config.rotation_fill_color``.

The variant of each image is inferred from the containing directory name
when multiple crop variants are configured, and from the single configured
variant otherwise.

For geometric augmentations (``vflip``, ``hflip``, ``rot45``, ``rot65``,
``rot90``) the image and its mask are transformed together (the mask uses
nearest-neighbor interpolation to stay strictly binary), then the object
pixels are composited onto a fresh solid background.

For non-geometric augmentations (``blur``, ``noise03``, ``noise06``,
``cjitter``) the transform is applied to the whole image, then only the
pixels inside the mask are kept and composited onto a fresh solid
background. This yields clean object edges (blur samples true neighbor
pixels before the mask is applied) and a completely uniform background.

Each augmented image is written alongside a matching augmented mask (e.g.
``image_001_0_vflip.jpg`` + ``image_001_0_vflip_mask.png``) so the pairing
survives any future re-augmentation. After all augmentation finishes
(whether successfully or with an exception), every ``_mask.png`` sidecar
under the train split is deleted.

Which augmentations are applied is controlled by
``config.active_augmentations`` (selected by the active prompt). Output
filenames always follow the canonical augmentation order defined in
``config_loader``, so runs are deterministic regardless of YAML ordering.

JPEG save settings match ``segmentation.py`` (quality=95, subsampling=0,
optimize=True) so an augmented copy has the same fidelity as the original
crop it was derived from.
"""

import argparse
import os

import numpy as np
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

# Suffix used by segmentation.py for the mask sidecar next to every crop.
# Kept identical here so the two stages stay in sync.
MASK_SIDECAR_SUFFIX = "_mask.png"

# Crop-variant names, matching config_loader.ALLOWED_CROP_VARIANTS.
_RAW_VARIANT = "raw"
_BLACK_BACKGROUND_VARIANT = "black_background"
_IMAGENET_MEAN_BACKGROUND_VARIANT = "imagenet_mean_background"

# JPEG encoder settings for saved augmentations. Matches segmentation.py so
# augmented crops have the same fidelity as the originals they were derived
# from.
_JPEG_QUALITY = 95
_JPEG_SUBSAMPLING = 0
_JPEG_OPTIMIZE = True

# Augmentations that transform the geometry of the image (position/rotation).
# For these, the mask must be transformed together with the image.
_GEOMETRIC_AUGMENTATIONS = frozenset(
    ["vflip", "hflip", "rot45", "rot65", "rot90"]
)


# ── Background color per variant ─────────────────────────────────────────────


def get_background_color_for_variant(
    variant: str,
    rotation_fill_color: tuple[int, int, int],
) -> tuple[int, int, int]:
  """Returns the RGB background color used by the given crop variant.

  Args:
      variant: One of ``'raw'``, ``'black_background'``,
        ``'imagenet_mean_background'``.
      rotation_fill_color: The configured background color used by the
        ``imagenet_mean_background`` variant.

  Returns:
      The RGB background color used by that variant's saved crop.

  Raises:
      ValueError: If ``variant`` is not one of the allowed values.
  """
  if variant == _RAW_VARIANT:
    return (0, 0, 0)
  if variant == _BLACK_BACKGROUND_VARIANT:
    return (0, 0, 0)
  if variant == _IMAGENET_MEAN_BACKGROUND_VARIANT:
    return rotation_fill_color
  raise ValueError(f"Unknown crop variant: {variant!r}")


# ── Mask I/O and compositing ─────────────────────────────────────────────────


def build_mask_sidecar_path(image_path: str) -> str:
  """Returns the mask sidecar path for a given image path.

  Args:
      image_path: Path to a crop image such as ``.../image_001_0.jpg``.

  Returns:
      Path to the matching mask sidecar such as
      ``.../image_001_0_mask.png``.
  """
  base_name = os.path.splitext(image_path)[0]
  return f"{base_name}{MASK_SIDECAR_SUFFIX}"


def load_mask_as_pil(mask_path: str) -> PIL.Image.Image:
  """Loads a mask sidecar as a single-channel PIL image.

  Args:
      mask_path: Absolute path to the mask sidecar PNG.

  Returns:
      A single-channel PIL image in mode ``'L'`` with values in
      ``{0, 255}``.
  """
  with PIL.Image.open(mask_path) as opened_mask:
    return opened_mask.convert("L")


def composite_foreground_on_background(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    background_color: tuple[int, int, int],
) -> PIL.Image.Image:
  """Composites the object pixels of an image onto a solid background.

  Pixels where the mask is non-zero come from ``image``; all other pixels
  come from a solid canvas filled with ``background_color``. The mask is
  binarized at ``> 0`` so that any interpolation artifacts introduced by an
  upstream transform do not leak background pixels through anti-aliased
  edges.

  Args:
      image: RGB PIL image, same size as ``mask``.
      mask: Single-channel PIL image in mode ``'L'``, same size as ``image``.
      background_color: RGB tuple used for pixels outside the mask.

  Returns:
      An RGB PIL image with the object on the solid background.
  """
  image_array = np.array(image, dtype=np.uint8)
  mask_array = np.array(mask, dtype=np.uint8)

  binary_mask = mask_array > 0
  background_array = np.full_like(image_array, 0)
  background_array[..., 0] = background_color[0]
  background_array[..., 1] = background_color[1]
  background_array[..., 2] = background_color[2]

  composited = np.where(
      binary_mask[..., np.newaxis], image_array, background_array
  )
  return PIL.Image.fromarray(composited)


# ── Geometric augmentations (mask must follow) ───────────────────────────────


def apply_fixed_rotation_to_image_and_mask(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    degrees: float,
    background_color: tuple[int, int, int],
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Rotates an image and its mask together by a fixed angle.

  The image is rotated with bilinear interpolation and its newly exposed
  corners are filled with ``background_color`` so the fill matches the
  final background. The mask is rotated with nearest-neighbor interpolation
  and newly exposed corners are filled with ``0`` (background) so the mask
  stays strictly binary.

  Rotation angle convention: positive = clockwise.

  Args:
      image: PIL RGB image to rotate.
      mask: Single-channel PIL image in mode ``'L'``, same size as ``image``.
      degrees: Fixed rotation angle in degrees.
      background_color: RGB tuple used to fill the image's empty corners.

  Returns:
      A tuple ``(rotated_image, rotated_mask)``.
  """
  rotated_image = TF.rotate(
      image,
      angle=degrees,
      interpolation=TF.InterpolationMode.BILINEAR,
      fill=list(background_color),
  )
  rotated_mask = TF.rotate(
      mask,
      angle=degrees,
      interpolation=TF.InterpolationMode.NEAREST,
      fill=[0],
  )
  return rotated_image, rotated_mask


def apply_vertical_flip_to_image_and_mask(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Flips an image and its mask vertically.

  Args:
      image: PIL RGB image to flip.
      mask: Single-channel PIL image in mode ``'L'``, same size as ``image``.

  Returns:
      A tuple ``(flipped_image, flipped_mask)``.
  """
  return TF.vflip(image), TF.vflip(mask)


def apply_horizontal_flip_to_image_and_mask(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Flips an image and its mask horizontally.

  Args:
      image: PIL RGB image to flip.
      mask: Single-channel PIL image in mode ``'L'``, same size as ``image``.

  Returns:
      A tuple ``(flipped_image, flipped_mask)``.
  """
  return TF.hflip(image), TF.hflip(mask)


def build_geometric_augmentation(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    augmentation_name: str,
    background_color: tuple[int, int, int],
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Builds one geometric augmentation of image and mask together.

  Args:
      image: PIL RGB image.
      mask: Single-channel PIL image in mode ``'L'``, same size as ``image``.
      augmentation_name: One of the entries in ``_GEOMETRIC_AUGMENTATIONS``.
      background_color: RGB tuple used to fill the image's empty corners after a
        rotation.

  Returns:
      A tuple ``(transformed_image, transformed_mask)``.

  Raises:
      ValueError: If ``augmentation_name`` is not a geometric augmentation.
  """
  if augmentation_name == "vflip":
    return apply_vertical_flip_to_image_and_mask(image, mask)
  if augmentation_name == "hflip":
    return apply_horizontal_flip_to_image_and_mask(image, mask)
  if augmentation_name == "rot45":
    return apply_fixed_rotation_to_image_and_mask(
        image, mask, 45, background_color
    )
  if augmentation_name == "rot65":
    return apply_fixed_rotation_to_image_and_mask(
        image, mask, 65, background_color
    )
  if augmentation_name == "rot90":
    return apply_fixed_rotation_to_image_and_mask(
        image, mask, 90, background_color
    )
  raise ValueError(
      f"Unknown geometric augmentation name: {augmentation_name!r}"
  )


# ── Non-geometric augmentations (mask is unchanged) ──────────────────────────


def apply_gaussian_blur(image: PIL.Image.Image) -> PIL.Image.Image:
  """Applies gaussian blur to an image.

  Args:
      image: PIL RGB image to blur.

  Returns:
      The blurred PIL image.
  """
  blur_transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(2, 20))
  return blur_transform(image)


def apply_add_noise(
    image: PIL.Image.Image, noise_factor: float
) -> PIL.Image.Image:
  """Adds uniform noise to an image.

  Args:
      image: PIL RGB image to add noise to.
      noise_factor: Scalar controlling the magnitude of noise.

  Returns:
      The noisy PIL image.
  """
  image_tensor = T.ToTensor()(image)
  noisy_tensor = image_tensor + torch.rand_like(image_tensor) * noise_factor
  noisy_tensor = torch.clip(noisy_tensor, 0.0, 1.0)
  return T.ToPILImage()(noisy_tensor)


def apply_color_jitter(image: PIL.Image.Image) -> PIL.Image.Image:
  """Applies brightness color jitter to an image.

  Args:
      image: PIL RGB image.

  Returns:
      The color-jittered PIL image.
  """
  jitter_transform = T.ColorJitter(brightness=(0.1, 1.8))
  return jitter_transform(image)


def build_non_geometric_augmentation(
    image: PIL.Image.Image,
    augmentation_name: str,
) -> PIL.Image.Image:
  """Builds one non-geometric augmentation of an image.

  Args:
      image: PIL RGB image.
      augmentation_name: One of ``'blur'``, ``'noise03'``, ``'noise06'``,
        ``'cjitter'``.

  Returns:
      The transformed PIL image, same size as the input.

  Raises:
      ValueError: If ``augmentation_name`` is not a non-geometric
          augmentation.
  """
  if augmentation_name == "blur":
    return apply_gaussian_blur(image)
  if augmentation_name == "noise03":
    return apply_add_noise(image, 0.3)
  if augmentation_name == "noise06":
    return apply_add_noise(image, 0.6)
  if augmentation_name == "cjitter":
    return apply_color_jitter(image)
  raise ValueError(
      f"Unknown non-geometric augmentation name: {augmentation_name!r}"
  )


# ── Full augmentation pipeline ───────────────────────────────────────────────


def build_single_augmentation_with_mask(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    augmentation_name: str,
    background_color: tuple[int, int, int],
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
  """Builds one augmented image and its matching mask, foreground-only.

  Geometric augmentations transform the image and mask together, then
  composite the object onto a solid background.

  Non-geometric augmentations transform only the image (on the whole
  image, so edge pixels are computed against true neighbors), then keep
  only pixels inside the mask and composite them onto a solid background.
  The mask itself is unchanged.

  Args:
      image: Original PIL RGB image.
      mask: Original single-channel PIL image in mode ``'L'``.
      augmentation_name: Name of the augmentation to apply. Must be one of the
        entries in ``config_loader.CANONICAL_AUGMENTATION_ORDER``.
      background_color: RGB tuple used as the solid background.

  Returns:
      A tuple ``(augmented_image, augmented_mask)``.

  Raises:
      ValueError: If ``augmentation_name`` is not recognised.
  """
  if augmentation_name in _GEOMETRIC_AUGMENTATIONS:
    transformed_image, transformed_mask = build_geometric_augmentation(
        image, mask, augmentation_name, background_color
    )
    composited_image = composite_foreground_on_background(
        transformed_image, transformed_mask, background_color
    )
    return composited_image, transformed_mask

  transformed_image = build_non_geometric_augmentation(image, augmentation_name)
  composited_image = composite_foreground_on_background(
      transformed_image, mask, background_color
  )
  return composited_image, mask


def build_augmented_images_with_masks(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    augmentations_to_apply: tuple[str, ...],
    background_color: tuple[int, int, int],
) -> dict[str, tuple[PIL.Image.Image, PIL.Image.Image]]:
  """Creates augmented image and mask pairs keyed by augmentation name.

  The loader has already reordered ``augmentations_to_apply`` into
  canonical order, so iterating over it directly is enough to make the
  on-disk output deterministic.

  Args:
      image: Original PIL RGB image.
      mask: Original single-channel PIL image in mode ``'L'``.
      augmentations_to_apply: Sequence of augmentation names to apply, already
        in canonical order.
      background_color: RGB tuple used as the solid background.

  Returns:
      A dict mapping augmentation name to a
      ``(augmented_image, augmented_mask)`` tuple.
  """
  augmented_outputs = {}
  for augmentation_name in augmentations_to_apply:
    augmented_outputs[augmentation_name] = build_single_augmentation_with_mask(
        image, mask, augmentation_name, background_color
    )
  return augmented_outputs


# ── Saving helpers ───────────────────────────────────────────────────────────


def save_augmented_outputs(
    augmented_outputs: dict[str, tuple[PIL.Image.Image, PIL.Image.Image]],
    folder_path: str,
    original_base_name: str,
) -> None:
  """Saves all augmented image + mask pairs to disk.

  Uses the same JPEG settings as ``segmentation.py`` so an augmented copy
  has the same fidelity as the original crop it was derived from. Masks
  are written as PNGs with the same ``_mask.png`` suffix used by
  ``segmentation.py``.

  Args:
      augmented_outputs: Dict mapping augmentation name to a ``(image, mask)``
        tuple.
      folder_path: Destination folder path.
      original_base_name: Base filename (without extension) of the original.
  """
  for augmentation_name, (
      augmented_image,
      augmented_mask,
  ) in augmented_outputs.items():
    image_file_name = (
        f"{original_base_name}_{augmentation_name}{OUTPUT_EXTENSION}"
    )
    image_output_path = os.path.join(folder_path, image_file_name)

    mask_file_name = (
        f"{original_base_name}_{augmentation_name}{MASK_SIDECAR_SUFFIX}"
    )
    mask_output_path = os.path.join(folder_path, mask_file_name)

    if augmented_image.mode != "RGB":
      augmented_image = augmented_image.convert("RGB")

    augmented_image.save(
        image_output_path,
        "JPEG",
        quality=_JPEG_QUALITY,
        subsampling=_JPEG_SUBSAMPLING,
        optimize=_JPEG_OPTIMIZE,
    )

    if augmented_mask.mode != "L":
      augmented_mask = augmented_mask.convert("L")
    augmented_mask.save(mask_output_path, format="PNG", optimize=True)


# ── Validation and discovery ─────────────────────────────────────────────────


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


def is_mask_sidecar_filename(file_name: str) -> bool:
  """Checks if a filename is a mask sidecar produced by segmentation.py.

  Args:
      file_name: Image file name (no directory component).

  Returns:
      ``True`` if the filename ends with ``_mask.png``.
  """
  return file_name.lower().endswith(MASK_SIDECAR_SUFFIX)


def is_augmented_filename(file_name: str) -> bool:
  """Checks if a filename already corresponds to an augmented image.

  Scans for every suffix in ``config_loader.CANONICAL_AUGMENTATION_ORDER``,
  not just the currently active set, so leftover augmentations from a
  previous run with different settings are still detected. Mask sidecars
  are stripped of their mask suffix first so an augmented mask like
  ``foo_vflip_mask.png`` is also detected.

  Args:
      file_name: Image or mask file name (no directory component).

  Returns:
      ``True`` if the filename ends with any known augmentation suffix.
  """
  base_name = os.path.splitext(file_name)[0]
  if base_name.lower().endswith("_mask"):
    base_name = base_name[: -len("_mask")]
  for augmentation_name in config_loader.CANONICAL_AUGMENTATION_ORDER:
    if base_name.endswith(f"_{augmentation_name}"):
      return True
  return False


def discover_target_folders(
    train_dir: str, crop_variants: tuple[str, ...]
) -> list[tuple[str, str, str]]:
  """Returns the sorted list of target subfolders containing images to augment.

  When ``crop_variants`` has a single entry, images are stored directly in
  each class folder under ``train_dir`` and the variant is that single
  configured entry. When ``crop_variants`` has multiple entries, each class
  folder contains one subdirectory per variant and the variant is taken
  from the subdirectory name.

  Args:
      train_dir: Path to the train split folder.
      crop_variants: Tuple of active crop variants from the config.

  Returns:
      A sorted list of ``(target_label, target_path, variant_name)``
      tuples. ``target_label`` is human-readable for logging.

  Raises:
      ValueError: If no class subfolders are found.
  """
  class_entries = sorted(
      [entry for entry in os.scandir(train_dir) if entry.is_dir()],
      key=lambda entry: entry.name,
  )
  if not class_entries:
    raise ValueError(f"No class subfolders found under: {train_dir}")

  if len(crop_variants) == 1:
    only_variant = crop_variants[0]
    return [(entry.name, entry.path, only_variant) for entry in class_entries]

  target_folders = []
  for entry in class_entries:
    for variant in crop_variants:
      variant_path = os.path.join(entry.path, variant)
      if os.path.isdir(variant_path):
        target_folders.append(
            (f"{entry.name}/{variant}", variant_path, variant)
        )
  return target_folders


def find_pre_existing_augmentations(class_folder: str) -> list[str]:
  """Returns names of any pre-existing augmented files in a class folder.

  Args:
      class_folder: Path to a class folder under the train split.

  Returns:
      A sorted list of augmented file names found in the folder. Includes
      both augmented images (``*.jpg``) and augmented mask sidecars
      (``*_mask.png``).
  """
  return sorted(
      file_name
      for file_name in os.listdir(class_folder)
      if file_name.lower().endswith(IMAGE_EXTENSIONS)
      and is_augmented_filename(file_name)
  )


def validate_no_pre_existing_augmentations(
    target_folders: list[tuple[str, str, str]],
) -> None:
  """Stops execution if any target folder already contains augmented files.

  Performs the check across all target folders upfront so misconfigured
  runs are detected before any augmentation starts.

  Args:
      target_folders: List of ``(target_label, target_path, variant_name)``
        tuples.

  Raises:
      FileExistsError: If any folder contains augmented files.
  """
  folders_with_augmentations = []
  for target_label, target_path, _ in target_folders:
    existing_augmented_files = find_pre_existing_augmentations(target_path)
    if existing_augmented_files:
      sample = existing_augmented_files[:5]
      extra = len(existing_augmented_files) - len(sample)
      sample_text = ", ".join(sample)
      if extra > 0:
        sample_text += f", ... (+{extra} more)"
      folders_with_augmentations.append(
          f"  - {target_label}: {len(existing_augmented_files)} files "
          f"({sample_text})"
      )

  if folders_with_augmentations:
    details = "\n".join(folders_with_augmentations)
    raise FileExistsError(
        "Augmented files already exist in the following folders:\n"
        f"{details}\n"
        "Remove them before re-running, or run on a fresh classifier dataset."
    )


# ── Per-folder pipeline ──────────────────────────────────────────────────────


def list_original_image_names(folder_path: str) -> list[str]:
  """Returns sorted names of original crop images in a folder.

  Excludes mask sidecars and any files that already look augmented.

  Args:
      folder_path: Path to a class or variant folder.

  Returns:
      A naturally sorted list of image file names.
  """
  return sorted(
      file_name
      for file_name in os.listdir(folder_path)
      if file_name.lower().endswith(IMAGE_EXTENSIONS)
      and not is_mask_sidecar_filename(file_name)
      and not is_augmented_filename(file_name)
  )


def process_target_folder(
    target_label: str,
    target_path: str,
    variant_name: str,
    augmentations_to_apply: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
) -> None:
  """Applies augmentations to all images in one target folder.

  Args:
      target_label: Human-readable label used in progress logs.
      target_path: Path to the folder containing original crops and masks.
      variant_name: The crop variant this folder holds. Determines the
        background color used when compositing.
      augmentations_to_apply: Sequence of augmentation names to apply, in
        canonical order.
      rotation_fill_color: The configured background color used by the
        ``imagenet_mean_background`` variant.

  Raises:
      ValueError: If a mask size does not match its image size.
      FileNotFoundError: If any mask sidecars are missing.
  """
  background_color = get_background_color_for_variant(
      variant_name, rotation_fill_color
  )

  original_image_names = list_original_image_names(target_path)
  total_images = len(original_image_names)
  print(
      f"\n[{target_label}] variant={variant_name} "
      f"background={list(background_color)} "
      f"processing {total_images} image(s)"
  )

  if total_images == 0:
    return

  progress_bar = tqdm.tqdm(
      original_image_names,
      total=total_images,
      desc=target_label,
      unit="img",
  )

  missing_masks = []

  for image_name in progress_bar:
    image_path = os.path.join(target_path, image_name)
    mask_path = build_mask_sidecar_path(image_path)
    original_base_name = os.path.splitext(image_name)[0]

    if not os.path.isfile(mask_path):
      missing_masks.append(image_name)
      continue

    with PIL.Image.open(image_path) as opened_image:
      image = PIL.ImageOps.exif_transpose(opened_image)
      image = image.convert("RGB")

    mask = load_mask_as_pil(mask_path)
    if mask.size != image.size:
      raise ValueError(
          f"Mask size {mask.size} does not match image size {image.size} "
          f"for {image_path}"
      )

    augmented_outputs = build_augmented_images_with_masks(
        image,
        mask,
        augmentations_to_apply,
        background_color,
    )
    save_augmented_outputs(augmented_outputs, target_path, original_base_name)

  if missing_masks:
    sample = missing_masks[:5]
    extra = len(missing_masks) - len(sample)
    sample_text = ", ".join(sample)
    if extra > 0:
      sample_text += f", ... (+{extra} more)"
    raise FileNotFoundError(
        f"[{target_label}] Missing mask sidecar for {len(missing_masks)} "
        f"image(s): {sample_text}. Regenerate crops with segmentation.py "
        "so each image has a matching '_mask.png' sidecar."
    )


# ── Mask cleanup ─────────────────────────────────────────────────────────────


def delete_mask_sidecars_under(root_dir: str) -> tuple[int, list[str]]:
  """Recursively deletes every ``_mask.png`` file under ``root_dir``.

  The masks are only needed by the augmentation stage. Once augmentation
  has run (successfully or not), they can be removed so downstream trainers
  aren't confused by non-image files in the class folders. Nothing else is
  touched: only files whose lowercase name ends in the mask sidecar suffix
  are deleted, and no directories are removed.

  Errors on individual files are collected rather than raised so that a
  single un-deletable file does not stop cleanup of the rest.

  Args:
      root_dir: Directory to walk. All descendants are considered.

  Returns:
      A tuple ``(deleted_count, error_messages)`` where ``error_messages``
      is a list of ``'<path>: <error>'`` strings, one per failed deletion.
  """
  deleted_count = 0
  error_messages = []

  for current_directory, _, file_names in os.walk(root_dir):
    for file_name in file_names:
      if not is_mask_sidecar_filename(file_name):
        continue
      mask_path = os.path.join(current_directory, file_name)
      try:
        os.remove(mask_path)
        deleted_count += 1
      except OSError as error:
        error_messages.append(f"{mask_path}: {error}")

  return deleted_count, error_messages


def cleanup_mask_sidecars_in_directory(directory: str, label: str) -> None:
  """Deletes every ``_mask.png`` under a directory and prints a summary.

  Args:
      directory: Directory to walk. Skipped with a message if it does not exist.
      label: Short human-readable label used in the summary line (e.g. ``'train
        split'``).
  """
  if not os.path.isdir(directory):
    print(f"Skipping cleanup for {label} (missing directory): {directory}")
    return

  print(f"\nCleaning up mask sidecars under {label}: {directory}")
  deleted_count, error_messages = delete_mask_sidecars_under(directory)
  print(f"Deleted {deleted_count} mask sidecar(s) from {label}.")
  if error_messages:
    print(f"{len(error_messages)} deletion(s) failed in {label}:")
    for message in error_messages:
      print(f"  {message}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(config_path: str = _DEFAULT_CONFIG_PATH) -> None:
  """Entry point: validates inputs and augments every train target folder.

  After augmentation finishes (whether successfully or with an exception),
  every ``_mask.png`` sidecar under the train split is deleted. The val
  split is never touched here and, by design in ``segmentation.py``, has
  no mask sidecars to clean up.

  Args:
      config_path: Path to the YAML configuration file.
  """
  config = config_loader.load_config(config_path)

  train_dir = validate_train_split_exists(
      config.classifier_dir, config.train_split_name
  )
  target_folders = discover_target_folders(train_dir, config.crop_variants)
  validate_no_pre_existing_augmentations(target_folders)

  target_labels = [label for label, _, _ in target_folders]
  augmentations_to_apply = config.active_augmentations
  print(f"Classifier directory: {config.classifier_dir}")
  print(f"Train split:          {train_dir}")
  print(f"Prompt:               {config.prompt_to_detect!r}")
  print(f"Crop variants:        {list(config.crop_variants)}")
  print(f"Rotation fill color:  {list(config.rotation_fill_color)}")
  print(f"Found {len(target_folders)} target folder(s): {target_labels}")
  print(f"Active augmentations: {list(augmentations_to_apply)}")

  try:
    for target_label, target_path, variant_name in target_folders:
      process_target_folder(
          target_label,
          target_path,
          variant_name,
          augmentations_to_apply,
          config.rotation_fill_color,
      )
    print("\nAugmentation done.")
  finally:
    cleanup_mask_sidecars_in_directory(train_dir, "train split")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Apply foreground-only augmentations to the train split."
  )
  parser.add_argument(
      "--config",
      type=str,
      default=_DEFAULT_CONFIG_PATH,
      help="Path to the config.yaml file.",
  )
  args = parser.parse_args()
  main(args.config)
