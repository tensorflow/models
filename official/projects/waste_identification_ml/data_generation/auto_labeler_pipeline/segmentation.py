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

"""Batch SAM3 segmentation pipeline that writes a classifier-ready dataset.

Discovers dataset subfolders under a single root directory, runs SAM3
inference on each image, and writes the resulting crops directly into a
sibling classifier dataset. Each dataset folder name becomes a class label
under ``train/`` and ``val/``.

For every saved crop on the train split, a matching binary mask is written
as a sibling PNG with a ``_mask.png`` suffix (e.g. ``image_001_0.jpg`` +
``image_001_0_mask.png``). The mask is aligned pixel-for-pixel with its
crop and is consumed by ``augment_train_split.py`` so augmentations can be
restricted to the foreground object. The val split does not receive mask
sidecars because nothing downstream consumes them.

Backgrounds per variant:
  * ``raw``                       -> black (unchanged).
  * ``black_background``          -> black (unchanged).
  * ``imagenet_mean_background``  -> ``config.rotation_fill_color`` (the
    variant name is kept for backward compatibility with existing configs
    and on-disk layouts; the actual color now comes from the config so it
    matches the augmentation background exactly).

Expected layout under ``config.root_dir``::

    root_dir/
    ├── dataset_a/
    │   └── train_val_images/
    │       ├── train/
    │       └── val/
    └── dataset_b/
        └── train_val_images/
            ├── train/
            └── val/

Produces the sibling directory ``config.classifier_dir`` with::

    classifier_dir/
    ├── train/
    │   ├── dataset_a/
    │   │   ├── image_001_0.jpg
    │   │   ├── image_001_0_mask.png
    │   │   └── ...
    │   └── dataset_b/
    │       └── ...
    └── val/
        ├── dataset_a/
        └── dataset_b/

GPU inference runs sequentially on the main thread, while CPU
post-processing (crop saving) is submitted to a ThreadPoolExecutor with
manual future-based backpressure.

The set of crop variants to save is controlled by ``config.crop_variants``.
When exactly one variant is selected, crops are written flat under each
class folder. When more than one variant is selected, crops are organized
into per-variant subdirectories under each class folder.
"""

from concurrent import futures
import gc
import glob
import logging
import os
import time
from typing import Any, Optional
import warnings

import natsort
import numpy as np
from PIL import Image
import torch
import tqdm

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import sam3_inference_utils

# ── Warning suppression ─────────────────────────────────────────────────────
# NO_ALBUMENTATIONS_UPDATE must be set BEFORE the albumentations package is
# imported (some third-party detectors import it transitively), otherwise
# the update-check UserWarning has already fired by the time we could
# filter it.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# torch.jit TracerWarning: raised by any traced/scripted model path some
# third-party detectors take. Only relevant when the traced model must
# handle different input shapes than the trace saw; not our case.
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

try:
  # pylint: disable=g-import-not-at-top
  from sam3 import model_builder as sam3_model_builder  # type: ignore[import-error]
  from sam3.model import sam3_image_processor  # type: ignore[import-error]
  # pylint: enable=g-import-not-at-top
except ImportError:
  sam3_model_builder = None
  sam3_image_processor = None


def _silence_third_party_logger(logger_name: str) -> None:
  """Raises a third-party logger and every attached handler to ERROR.

  Setting the logger level alone is not enough for libraries that add
  their own StreamHandler with an independent level. We lift both so
  nothing below ERROR gets through, regardless of which side of the
  logging plumbing is doing the filtering.

  Args:
      logger_name: Name of the third-party logger, e.g. ``'transformers'``.
  """
  target_logger = logging.getLogger(logger_name)
  target_logger.setLevel(logging.ERROR)
  for attached_handler in target_logger.handlers:
    attached_handler.setLevel(logging.ERROR)


# ── Warning suppression (part 3: after third-party imports) ─────────────────
# Silence the "loss_type=None" config notice and any other WARNING-level
# lines from the ``transformers`` logger. Errors from the same logger are
# still shown.
_silence_third_party_logger("transformers")


# Resolve config.yaml relative to this script file so the script runs
# correctly regardless of the caller's current working directory.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)
PACKETS_PROMPT_NAME = "packets"

# Crop-variant names, matching config_loader.ALLOWED_CROP_VARIANTS.
_RAW_VARIANT = "raw"
_BLACK_BACKGROUND_VARIANT = "black_background"
_IMAGENET_MEAN_BACKGROUND_VARIANT = "imagenet_mean_background"

# Suffix for the mask sidecar written next to every crop on the train
# split. The augmentation stage looks for this exact suffix.
_MASK_SIDECAR_SUFFIX = "_mask.png"

# JPEG encoder settings for saved crops. quality=95 with subsampling=0
# (no chroma downsampling) gives visually near-lossless output at roughly
# 2x the file size of PIL's defaults; optimize=True runs a second pass
# that shaves a few percent off the file size at no visual cost.
_JPEG_QUALITY = 95
_JPEG_SUBSAMPLING = 0
_JPEG_OPTIMIZE = True

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
    dataset_directories: list[tuple[str, str]], train_val_folder_name: str
) -> list[tuple[str, str]]:
  """Validates that each dataset has the expected input layout.

  Args:
      dataset_directories: List of ``(dataset_name, dataset_path)`` tuples.
      train_val_folder_name: Name of the train/val input subfolder.

  Returns:
      A list of ``(dataset_name, input_dir)`` tuples ready for processing.

  Raises:
      FileNotFoundError: If a dataset is missing its input folder.
  """
  validated = []
  for dataset_name, dataset_path in dataset_directories:
    input_dir = os.path.join(dataset_path, train_val_folder_name)

    if not os.path.isdir(input_dir):
      raise FileNotFoundError(
          f"Dataset {dataset_name!r} is missing required input folder: "
          f"{input_dir}"
      )

    validated.append((dataset_name, input_dir))

  return validated


def validate_classifier_output_dir(classifier_output_dir: str) -> None:
  """Ensures the classifier output directory does not already exist.

  Args:
      classifier_output_dir: Path to the classifier dataset directory.

  Raises:
      FileExistsError: If the classifier output directory already exists.
  """
  if os.path.exists(classifier_output_dir):
    raise FileExistsError(
        "Classifier output directory already exists: "
        f"{classifier_output_dir}. Remove or rename it before re-running."
    )


# ── Variant helpers ───────────────────────────────────────────────────────────


def build_variant_directories(
    class_folder: str, variants: tuple[str, ...]
) -> dict[str, str]:
  """Creates output directories for each selected variant under a class folder.

  When only one variant is selected, the output directory is the class
  folder itself (flat layout). When multiple variants are selected, each
  variant gets its own subdirectory under the class folder.

  Args:
      class_folder: Path to the per-class folder (e.g.
        ``.../classifier/train/dataset_a``).
      variants: Sequence of variant names to save.

  Returns:
      A dict mapping variant name to its output directory path.
  """
  if len(variants) == 1:
    variant_directories = {variants[0]: class_folder}
  else:
    variant_directories = {
        variant: os.path.join(class_folder, variant) for variant in variants
    }

  for directory in variant_directories.values():
    os.makedirs(directory, exist_ok=True)

  return variant_directories


def build_variant_crop(
    image_array: np.ndarray,
    mask: np.ndarray,
    box: list[float],
    crop_size: tuple[int, int],
    variant: str,
    rotation_fill_color: tuple[int, int, int],
) -> Any:
  """Builds a single crop variant from an image and mask.

  Args:
      image_array: RGB image as a numpy array of shape ``(H, W, 3)``.
      mask: Binary mask of shape ``(H, W)``.
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.
      crop_size: Target letterbox size ``(height, width)``.
      variant: One of ``'raw'``, ``'black_background'``,
        ``'imagenet_mean_background'``.
      rotation_fill_color: Background color used by the
        ``imagenet_mean_background`` variant. Ignored by the other variants.

  Returns:
      A PIL image for the requested variant, or ``None`` for degenerate
      boxes in the ``'raw'`` variant.

  Raises:
      ValueError: If ``variant`` is not one of the allowed values.
  """
  if variant == _RAW_VARIANT:
    return sam3_inference_utils.crop_raw_masked_image(image_array, mask, box)
  if variant == _BLACK_BACKGROUND_VARIANT:
    return sam3_inference_utils.crop_masked_image(
        image_array, mask, box, size=crop_size
    )
  if variant == _IMAGENET_MEAN_BACKGROUND_VARIANT:
    return sam3_inference_utils.crop_with_mean_background_blend(
        image_array,
        mask,
        box,
        size=crop_size,
        background_color=rotation_fill_color,
    )
  raise ValueError(f"Unknown crop variant: {variant!r}")


def build_variant_mask(
    mask: np.ndarray,
    box: list[float],
    crop_size: tuple[int, int],
    variant: str,
) -> Optional[np.ndarray]:
  """Builds the mask aligned to a single crop variant's geometry.

  The returned mask has the same shape as the saved crop image for that
  variant, so consumers can composite the two without any re-alignment.

  Args:
      mask: Binary mask of shape ``(H, W)``.
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.
      crop_size: Target letterbox size ``(height, width)`` used by the
        letterboxed variants.
      variant: One of ``'raw'``, ``'black_background'``,
        ``'imagenet_mean_background'``.

  Returns:
      A ``uint8`` binary mask (values in ``{0, 255}``) matching the saved
      crop's shape. Returns ``None`` for degenerate boxes in the ``'raw'``
      variant, matching :func:`build_variant_crop`.

  Raises:
      ValueError: If ``variant`` is not one of the allowed values.
  """
  if variant == _RAW_VARIANT:
    return sam3_inference_utils.build_raw_variant_mask(mask, box)
  if variant in (_BLACK_BACKGROUND_VARIANT, _IMAGENET_MEAN_BACKGROUND_VARIANT):
    return sam3_inference_utils.build_letterboxed_variant_mask(
        mask, box, size=crop_size
    )
  raise ValueError(f"Unknown crop variant: {variant!r}")


def generate_selected_crops(
    image: Image.Image,
    state: dict[str, Any],
    score_threshold: float,
    crop_size: tuple[int, int],
    variants: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
    build_masks: bool = True,
) -> list[tuple[int, dict[str, Any], dict[str, Optional[np.ndarray]]]]:
  """Generates crop variants and optionally their geometry-aligned masks.

  Skips the work of building unused crop variants. Mask hole-filling is
  performed once per detection and reused across variants.

  Args:
      image: Input RGB PIL image.
      state: SAM output dict with ``'masks'``, ``'boxes'``, ``'scores'`` keys.
      score_threshold: Minimum confidence score to include a detection.
      crop_size: Target letterbox size for letterboxed variants.
      variants: Sequence of variant names to generate.
      rotation_fill_color: Background color used by the
        ``imagenet_mean_background`` variant.
      build_masks: When ``True``, also produce a geometry-aligned mask for every
        variant. When ``False``, the mask entry for every variant is ``None``
        (the corresponding sidecar is skipped downstream).

  Returns:
      A list of ``(detection_index, variant_to_crop, variant_to_mask)``
      tuples. ``variant_to_crop`` maps each requested variant name to its
      PIL image (or ``None`` for degenerate boxes). ``variant_to_mask``
      maps each variant name to its ``uint8`` mask array aligned with the
      crop, or ``None`` when ``build_masks`` is False or the box is
      degenerate.
  """
  image_array = np.array(image)
  crop_records = []

  num_detections = len(state["masks"])
  for detection_index in range(num_detections):
    score = state["scores"][detection_index].item()
    if score < score_threshold:
      continue

    mask = np.squeeze(state["masks"][detection_index])
    mask = sam3_inference_utils.fill_mask_holes(mask)
    box = state["boxes"][detection_index].tolist()

    variant_to_crop = {
        variant: build_variant_crop(
            image_array,
            mask,
            box,
            crop_size,
            variant,
            rotation_fill_color,
        )
        for variant in variants
    }
    if build_masks:
      variant_to_mask = {
          variant: build_variant_mask(mask, box, crop_size, variant)
          for variant in variants
      }
    else:
      variant_to_mask = {variant: None for variant in variants}
    crop_records.append((detection_index, variant_to_crop, variant_to_mask))

  return crop_records


# ── CPU worker functions ──────────────────────────────────────────────────────


def save_crop_image(crop: Image.Image, output_path: str) -> None:
  """Saves a single crop as a JPEG using the pipeline's encoder settings.

  Args:
      crop: PIL image to save.
      output_path: Absolute path to write to.
  """
  crop.save(
      output_path,
      quality=_JPEG_QUALITY,
      subsampling=_JPEG_SUBSAMPLING,
      optimize=_JPEG_OPTIMIZE,
  )


def save_mask_sidecar(mask: np.ndarray, output_path: str) -> None:
  """Saves a binary mask as a single-channel PNG.

  Args:
      mask: ``uint8`` mask array with values in ``{0, 255}``.
      output_path: Absolute path to write to.
  """
  mask_image = Image.fromarray(mask, mode="L")
  mask_image.save(output_path, format="PNG", optimize=True)


def save_one_detection(
    detection_index: int,
    variant_to_crop: dict[str, Any],
    variant_to_mask: dict[str, Optional[np.ndarray]],
    filename: str,
    variant_directories: dict[str, str],
    write_masks: bool,
) -> None:
  """Saves all selected variants of a single detection in parallel.

  For each variant, writes the crop JPEG and, when ``write_masks`` is
  ``True``, the aligned mask PNG.

  Args:
      detection_index: Index of this detection in the image.
      variant_to_crop: Dict mapping variant name to its PIL image (or None).
      variant_to_mask: Dict mapping variant name to its uint8 mask (or None).
      filename: Base filename without extension.
      variant_directories: Dict mapping variant name to output directory.
      write_masks: If ``True``, write ``<name>_mask.png`` sidecars alongside
        each crop. If ``False``, only the crop JPEGs are written.
  """
  crop_filename = f"{filename}_{detection_index}.jpg"
  mask_filename = f"{filename}_{detection_index}{_MASK_SIDECAR_SUFFIX}"

  save_tasks = []
  for variant, crop in variant_to_crop.items():
    if crop is None:
      continue
    variant_directory = variant_directories[variant]
    crop_path = os.path.join(variant_directory, crop_filename)
    save_tasks.append(("crop", crop, crop_path))

    if not write_masks:
      continue
    mask = variant_to_mask.get(variant)
    if mask is None:
      continue
    mask_path = os.path.join(variant_directory, mask_filename)
    save_tasks.append(("mask", mask, mask_path))

  if not save_tasks:
    return

  with futures.ThreadPoolExecutor(max_workers=len(save_tasks)) as nested_pool:
    save_futures = []
    for task_kind, payload, path in save_tasks:
      if task_kind == "crop":
        save_futures.append(nested_pool.submit(save_crop_image, payload, path))
      else:
        save_futures.append(
            nested_pool.submit(save_mask_sidecar, payload, path)
        )
    for save_future in futures.as_completed(save_futures):
      save_future.result()


def process_one_image_cpu(
    crop_records: list[
        tuple[int, dict[str, Any], dict[str, Optional[np.ndarray]]]
    ],
    filename: str,
    variant_directories: dict[str, str],
    write_masks: bool,
) -> None:
  """CPU post-processing for one image: saves all selected crop variants.

  Args:
      crop_records: List of ``(detection_index, variant_to_crop,
        variant_to_mask)`` tuples.
      filename: Base filename without extension.
      variant_directories: Dict mapping variant name to output directory.
      write_masks: Whether to write mask sidecars alongside each crop.
  """
  for detection_index, variant_to_crop, variant_to_mask in crop_records:
    save_one_detection(
        detection_index,
        variant_to_crop,
        variant_to_mask,
        filename,
        variant_directories,
        write_masks=write_masks,
    )


def _drain_one_completed_future(
    pending_futures: dict[futures.Future[Any], Any],
) -> None:
  """Waits for one pending CPU future to complete and reports errors.

  Args:
      pending_futures: Dict mapping in-flight futures to their filename. The
        completed entry is removed from this dict in place.
  """
  done_future = next(futures.as_completed(pending_futures))
  done_name = pending_futures.pop(done_future)
  try:
    done_future.result()
  except Exception as error:  # pylint: disable=broad-exception-caught
    print(f"  [ERROR] {done_name}: {error}")


def _drain_remaining_futures(
    pending_futures: dict[futures.Future[Any], Any],
) -> None:
  """Waits for all remaining CPU futures and reports errors.

  Args:
      pending_futures: Dict mapping in-flight futures to their filename.
  """
  for pending_future in futures.as_completed(pending_futures):
    future_filename = pending_futures[pending_future]
    try:
      pending_future.result()
    except Exception as error:  # pylint: disable=broad-exception-caught
      print(f"  [ERROR] {future_filename}: {error}")


# ── Per-split pipeline ────────────────────────────────────────────────────────


def _postprocess_detections(
    state: dict[str, Any],
    detection_config: config_loader.DetectionConfig,
    prompt: str,
) -> dict[str, Any]:
  """Applies the standard post-inference filters to a SAM state.

  Args:
      state: Raw SAM output dict from ``run_inference``.
      detection_config: Validated detection thresholds for this prompt.
      prompt: Text prompt for detection.

  Returns:
      The filtered SAM state dict.
  """
  state = sam3_inference_utils.filter_contained_sub_masks(
      state, containment_threshold=detection_config.containment_threshold
  )
  if prompt == PACKETS_PROMPT_NAME:
    state = sam3_inference_utils.merge_contained_boxes(state)
    # state = sam3_inference_utils.get_valid_bottle_indices(state)
  return state


def process_split(
    split_input_dir: str,
    class_folder: str,
    log_label: str,
    processor: Any,
    detection_config: config_loader.DetectionConfig,
    prompt: str,
    crop_variants: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
    max_cpu_workers: int,
    queue_maxsize: int,
    write_masks: bool,
) -> None:
  """Processes all images in one split (train or val) of one dataset.

  GPU inference runs on the main thread. After each image's crops are
  generated, the save work is submitted to a ThreadPoolExecutor. Manual
  backpressure drains one completed future when pending futures exceed
  ``queue_maxsize``.

  Args:
      split_input_dir: Path to the split folder (e.g.
        ``.../dataset_a/train_val_images/train``).
      class_folder: Path to the per-class output folder (e.g.
        ``.../classifier/train/dataset_a``).
      log_label: Label used in console logs (e.g. ``"dataset_a/train"``).
      processor: SAM3 processor instance.
      detection_config: Validated detection thresholds for this prompt.
      prompt: Text prompt for detection.
      crop_variants: Sequence of crop variant names to save.
      rotation_fill_color: Background color used by the
        ``imagenet_mean_background`` variant.
      max_cpu_workers: Size of the CPU thread pool.
      queue_maxsize: Maximum in-flight CPU jobs before the GPU loop blocks.
      write_masks: Whether to compute and write ``_mask.png`` sidecars for this
        split. Should be ``True`` for the train split (the augmentation stage
        needs them) and ``False`` for the val split (nothing downstream consumes
        them).
  """
  variant_directories = build_variant_directories(class_folder, crop_variants)

  image_paths = glob.glob(os.path.join(split_input_dir, "*"))
  image_paths = natsort.natsorted(image_paths)
  print(
      f"\n[{log_label}] Total images to process: {len(image_paths)} "
      f"(write_masks={write_masks})"
  )

  pending_futures = {}
  wall_start = time.perf_counter()

  with futures.ThreadPoolExecutor(max_workers=max_cpu_workers) as cpu_pool:
    for image_path in tqdm.tqdm(image_paths, desc=log_label):
      filename = os.path.splitext(os.path.basename(image_path))[0]

      try:
        with Image.open(image_path) as opened_image:
          image = opened_image.convert("RGB")
      except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"  [SKIP] {filename}: could not open image — {error}")
        continue

      image = sam3_inference_utils.resize_image_for_inference(
          image, max_short_side=detection_config.max_short_side
      )

      state = sam3_inference_utils.run_inference(processor, image, prompt)

      if not state["scores"].tolist():
        del image, state
        gc.collect()
        torch.cuda.empty_cache()
        continue

      state = _postprocess_detections(state, detection_config, prompt)

      crop_records = generate_selected_crops(
          image,
          state,
          detection_config.score_threshold,
          detection_config.crop_size,
          crop_variants,
          rotation_fill_color,
          build_masks=write_masks,
      )

      submitted_future = cpu_pool.submit(
          process_one_image_cpu,
          crop_records,
          filename,
          variant_directories,
          write_masks,
      )
      pending_futures[submitted_future] = filename

      if len(pending_futures) >= queue_maxsize + 1:
        _drain_one_completed_future(pending_futures)

      del image, state, crop_records
      gc.collect()
      torch.cuda.empty_cache()

    print(f"[{log_label}] GPU done — waiting for remaining CPU jobs...")
    _drain_remaining_futures(pending_futures)

  elapsed = time.perf_counter() - wall_start
  print(f"[{log_label}] Done in {format_elapsed_time(elapsed)}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────


def process_dataset(
    dataset_name: str,
    input_dir: str,
    classifier_output_dir: str,
    split_names: tuple[str, ...],
    train_split_name: str,
    processor: Any,
    detection_config: config_loader.DetectionConfig,
    prompt: str,
    crop_variants: tuple[str, ...],
    rotation_fill_color: tuple[int, int, int],
    max_cpu_workers: int,
    queue_maxsize: int,
) -> None:
  """Processes every split (train, val) of a single dataset.

  Mask sidecars are written only for the train split, since only the
  augmentation stage consumes them and the augmentation stage never
  touches the val split.

  Args:
      dataset_name: Name of the dataset, used as the class label.
      input_dir: Path to the dataset's train/val input folder.
      classifier_output_dir: Path to the classifier dataset root.
      split_names: Split subfolder names to iterate, e.g. ``('train', 'val')``.
      train_split_name: Name of the split that should have mask sidecars written
        (typically ``config.train_split_name``).
      processor: SAM3 processor instance.
      detection_config: Validated detection thresholds for this prompt.
      prompt: Text prompt for detection.
      crop_variants: Sequence of crop variant names to save.
      rotation_fill_color: Background color used by the
        ``imagenet_mean_background`` variant.
      max_cpu_workers: Size of the CPU thread pool.
      queue_maxsize: Maximum in-flight CPU jobs before the GPU loop blocks.

  Raises:
      FileNotFoundError: If a configured split is missing.
  """
  print(f"\n=== Dataset: {dataset_name} ===")
  dataset_start = time.perf_counter()

  for split_name in split_names:
    split_input_dir = os.path.join(input_dir, split_name)
    if not os.path.isdir(split_input_dir):
      raise FileNotFoundError(
          f"Dataset {dataset_name!r} is missing split folder: {split_input_dir}"
      )

    class_folder = os.path.join(classifier_output_dir, split_name, dataset_name)
    log_label = f"{dataset_name}/{split_name}"
    write_masks = split_name == train_split_name

    process_split(
        split_input_dir,
        class_folder,
        log_label,
        processor,
        detection_config,
        prompt,
        crop_variants,
        rotation_fill_color,
        max_cpu_workers,
        queue_maxsize,
        write_masks=write_masks,
    )

  dataset_elapsed = time.perf_counter() - dataset_start
  print(
      f"=== Dataset {dataset_name} finished in "
      f"{format_elapsed_time(dataset_elapsed)} ==="
  )


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
  """Entry point: discovers datasets and writes a classifier-ready dataset."""
  config = config_loader.load_config(CONFIG_PATH)
  os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

  validate_classifier_output_dir(config.classifier_dir)

  dataset_directories = discover_dataset_directories(config.root_dir)
  validated_datasets = validate_dataset_paths(
      dataset_directories, config.train_val_folder_name
  )

  split_names = (config.train_split_name, config.val_split_name)
  dataset_names = [name for name, _ in validated_datasets]
  print(f"Root directory:       {config.root_dir}")
  print(f"Classifier output:    {config.classifier_dir}")
  print(
      f"Found {len(validated_datasets)} dataset(s) (class labels): "
      f"{dataset_names}"
  )
  print(f"Splits:               {list(split_names)}")
  print(f"Saving crop variants: {list(config.crop_variants)}")
  print(f"Prompt:               {config.prompt_to_detect!r}")
  print(f"Rotation fill color:  {list(config.rotation_fill_color)}")

  detection_config = config.active_detection
  _, processor = build_sam3_processor(
      detection_config, config.sam3_checkpoint_path
  )

  total_start = time.perf_counter()

  for dataset_name, input_dir in validated_datasets:
    process_dataset(
        dataset_name,
        input_dir,
        config.classifier_dir,
        split_names,
        config.train_split_name,
        processor,
        detection_config,
        config.prompt_to_detect,
        config.crop_variants,
        config.rotation_fill_color,
        config.max_cpu_workers,
        config.queue_maxsize,
    )

  total_elapsed = time.perf_counter() - total_start
  print(f"\nAll datasets processed in {format_elapsed_time(total_elapsed)}")
  print(f"Classifier dataset written to: {config.classifier_dir}")


if __name__ == "__main__":
  main()
