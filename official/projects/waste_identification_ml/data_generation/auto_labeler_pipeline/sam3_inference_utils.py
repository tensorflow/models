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

"""Utility functions for preprocessing, SAM3 inference, and postprocessing.

Pure helpers with no dependence on ``config.yaml`` -- callers pass the
relevant thresholds and sizes in explicitly. Grouped roughly into:

  * Image resize / inference-state hygiene
  * Detection-state filters (contained-mask filter, edge-visibility filter,
    contained-box merge)
  * Cropping (raw, black background, ImageNet-mean blended background)
  * Mask hole filling
  * Convenience iterator that yields all three crop variants per detection
  * A matplotlib-based thumbnail viewer for interactive debugging
"""

import math
from typing import Any, Iterator, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Intermediate state entries dropped after inference to reduce memory
# footprint. They are set by the SAM3 processor but not needed downstream.
_INFERENCE_KEYS_TO_DROP = frozenset(
    ["backbone_out", "geometric_prompt", "image_embeddings"]
)

# State entries that are per-detection arrays; kept in lockstep after any
# filtering step.
_STATE_ARRAY_KEYS = ("masks", "masks_logits", "boxes", "scores")

# State entries preserved unchanged by the edge-visibility filter.
_SAM_META_KEYS = ("original_height", "original_width")

# ImageNet mean RGB, used as the default blended-crop background so training
# crops sit on the same neutral colour the classifier will see at inference.
_IMAGENET_MEAN_RGB = (124, 116, 104)

# Pixel buffer added to every bounding box before cropping so a small
# rounding error doesn't clip an object right at its edge.
_CROP_BUFFER = 5


# ── Image resize and state hygiene ───────────────────────────────────────────


def resize_image_for_inference(
    image: Image.Image,
    max_short_side: int,
) -> Image.Image:
  """Resizes an image so its short side does not exceed a maximum length.

  Maintains the original aspect ratio. If the short side is already within
  the limit, the image is returned unchanged.

  Args:
      image: A PIL RGB image to resize.
      max_short_side: Maximum allowed length for the shorter dimension.

  Returns:
      The resized PIL image, or the original if no resize was needed.
  """
  original_width, original_height = image.size
  short_side = min(original_width, original_height)

  if short_side <= max_short_side:
    return image

  scale = max_short_side / short_side
  new_width = int(original_width * scale)
  new_height = int(original_height * scale)

  return image.resize((new_width, new_height), Image.LANCZOS)


def move_inference_state_to_cpu(
    inference_state: dict[str, Any],
) -> dict[str, Any]:
  """Moves all tensors in an inference state dictionary to CPU.

  Recursively traverses nested dictionaries and moves any ``torch.Tensor``
  values to CPU in place.

  Args:
      inference_state: Dictionary potentially containing tensors and nested
        dictionaries of tensors.

  Returns:
      The same dictionary with all tensors moved to CPU.
  """
  for key, value in inference_state.items():
    if isinstance(value, torch.Tensor):
      inference_state[key] = value.cpu()
    elif isinstance(value, dict):
      move_inference_state_to_cpu(value)
  return inference_state


def run_inference(
    processor,
    image: Image.Image,
    label: str,
) -> dict[str, Any]:
  """Runs SAM grounded inference on a single image.

  Performs inference with mixed precision, drops large intermediate tensors
  to free GPU memory, and moves the remaining state to CPU.

  Args:
      processor: SAM processor instance with ``set_image`` and
        ``set_text_prompt`` methods.
      image: Input RGB image.
      label: Text prompt for grounded segmentation.

  Returns:
      An inference state dictionary with all tensors on CPU.
  """
  with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=label)

  for key in _INFERENCE_KEYS_TO_DROP:
    state.pop(key, None)

  return move_inference_state_to_cpu(state)


# ── Detection-state filters ──────────────────────────────────────────────────


def filter_contained_sub_masks(
    state: dict[str, Any], containment_threshold: float
) -> dict[str, Any]:
  """Removes smaller masks that are contained within larger masks.

  For each pair of masks, computes the containment ratio
  ``intersection / smaller_mask_area``. If the ratio exceeds the threshold,
  the smaller mask is discarded. All parallel arrays in ``state`` are
  filtered in lockstep.

  Args:
      state: Dict with keys ``'masks'``, ``'masks_logits'``, ``'boxes'``,
        ``'scores'``. ``masks`` is a bool tensor of shape ``[N, H, W]``.
      containment_threshold: Ratio above which a smaller mask is considered
        contained and will be removed.

  Returns:
      The filtered state dict with contained masks removed.
  """
  masks = state["masks"]
  num_masks = masks.shape[0]
  if num_masks == 0:
    return state

  flat_masks = masks.view(num_masks, -1).float()
  areas = flat_masks.sum(dim=1)

  pairwise_intersection = flat_masks @ flat_masks.T

  indices_to_remove = set()
  for outer_index in range(num_masks):
    if outer_index in indices_to_remove:
      continue
    for inner_index in range(outer_index + 1, num_masks):
      if inner_index in indices_to_remove:
        continue

      intersection = pairwise_intersection[outer_index, inner_index].item()
      area_outer = areas[outer_index].item()
      area_inner = areas[inner_index].item()

      if area_outer <= area_inner:
        smaller_index = outer_index
        smaller_area = area_outer
      else:
        smaller_index = inner_index
        smaller_area = area_inner

      if smaller_area == 0:
        indices_to_remove.add(smaller_index)
        continue

      containment_ratio = intersection / smaller_area
      if containment_ratio > containment_threshold:
        indices_to_remove.add(smaller_index)

  keep_indices = sorted(set(range(num_masks)) - indices_to_remove)
  keep_tensor = torch.tensor(keep_indices, dtype=torch.long)

  for key in _STATE_ARRAY_KEYS:
    state[key] = state[key][keep_tensor]

  return state


def get_valid_bottle_indices(
    sam_output: dict[str, Any],
    margin: int = 5,
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
  """Filters SAM output to remove edge bottles less than 50% visible.

  Bottles fully inside the image are always kept. Bottles touching the
  image edge are kept only if their mask area is at least
  ``visibility_threshold * median_area`` of the inner bottles.

  Args:
      sam_output: SAM output dict with keys ``'boxes'``, ``'masks'``,
        ``'masks_logits'``, ``'scores'``, ``'original_height'``,
        ``'original_width'``.
      margin: Pixel margin from the image border to consider as edge.
      visibility_threshold: Minimum fraction of the median inner-bottle area
        required for an edge bottle to be kept.

  Returns:
      A filtered SAM output dict with partially visible edge bottles removed.
  """
  boxes = sam_output["boxes"].numpy()
  masks = sam_output["masks"].numpy()
  if masks.ndim == 4:
    masks = masks.squeeze(1)

  image_height = sam_output["original_height"]
  image_width = sam_output["original_width"]

  inner_indices = []
  edge_indices = []
  for detection_index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
    touches_edge = (
        x_min <= margin
        or y_min <= margin
        or x_max >= image_width - margin
        or y_max >= image_height - margin
    )
    if touches_edge:
      edge_indices.append(detection_index)
    else:
      inner_indices.append(detection_index)

  if not inner_indices:
    return sam_output

  inner_areas = [np.sum(masks[i]) for i in inner_indices]
  median_area = np.median(inner_areas)
  minimum_valid_area = visibility_threshold * median_area

  valid_edge_indices = [
      i for i in edge_indices if np.sum(masks[i]) >= minimum_valid_area
  ]

  valid_indices = sorted(inner_indices + valid_edge_indices)

  filtered_output = {}
  for key in _SAM_META_KEYS:
    filtered_output[key] = sam_output[key]
  for key in _STATE_ARRAY_KEYS:
    filtered_output[key] = sam_output[key][valid_indices]

  return filtered_output


def merge_contained_boxes(
    state: dict[str, Any], containment_threshold: float = 0.7
) -> dict[str, Any]:
  """Merges detections where a smaller box is largely contained in a larger.

  Uses containment ratio (``intersection_area / smaller_box_area``) instead
  of IoU to avoid merging adjacent objects whose boxes partially overlap.

  Args:
      state: SAM output dict with ``'masks'``, ``'boxes'``, ``'scores'`` keys.
      containment_threshold: Minimum fraction of the smaller box's area that
        must overlap with the larger box to trigger a merge.

  Returns:
      A state dict with merged detections.
  """
  masks = state["masks"]
  boxes = state["boxes"]
  scores = state["scores"]

  if len(scores) == 0:
    return state

  num_detections = len(masks)
  box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

  is_absorbed = torch.zeros(num_detections, dtype=torch.bool)
  absorb_target = list(range(num_detections))

  for outer_index in range(num_detections):
    if is_absorbed[outer_index]:
      continue
    for inner_index in range(outer_index + 1, num_detections):
      if is_absorbed[inner_index]:
        continue

      intersection_x_min = torch.max(
          boxes[outer_index, 0], boxes[inner_index, 0]
      )
      intersection_y_min = torch.max(
          boxes[outer_index, 1], boxes[inner_index, 1]
      )
      intersection_x_max = torch.min(
          boxes[outer_index, 2], boxes[inner_index, 2]
      )
      intersection_y_max = torch.min(
          boxes[outer_index, 3], boxes[inner_index, 3]
      )

      intersection_area = torch.clamp(
          intersection_x_max - intersection_x_min, min=0
      ) * torch.clamp(intersection_y_max - intersection_y_min, min=0)

      if box_areas[outer_index] <= box_areas[inner_index]:
        smaller_index = outer_index
        larger_index = inner_index
        smaller_area = box_areas[outer_index]
      else:
        smaller_index = inner_index
        larger_index = outer_index
        smaller_area = box_areas[inner_index]

      if smaller_area == 0:
        is_absorbed[smaller_index] = True
        continue

      containment_ratio = intersection_area / smaller_area
      if containment_ratio > containment_threshold:
        is_absorbed[smaller_index] = True
        absorb_target[smaller_index] = larger_index

  # Group absorbed detections with their targets.
  groups = {}
  for detection_index in range(num_detections):
    if is_absorbed[detection_index]:
      target = absorb_target[detection_index]
      if target not in groups:
        groups[target] = [target]
      groups[target].append(detection_index)
    elif detection_index not in groups:
      groups[detection_index] = [detection_index]

  merged_masks = []
  merged_boxes = []
  merged_scores = []

  for member_indices in groups.values():
    member_tensor = torch.tensor(member_indices, dtype=torch.long)

    union_mask = masks[member_tensor].squeeze(1).any(dim=0)

    group_boxes = boxes[member_tensor]
    enclosing_box = torch.stack([
        group_boxes[:, 0].min(),
        group_boxes[:, 1].min(),
        group_boxes[:, 2].max(),
        group_boxes[:, 3].max(),
    ])

    combined_score = torch.tensor(min(scores[member_tensor].sum().item(), 1.0))

    merged_masks.append(union_mask)
    merged_boxes.append(enclosing_box)
    merged_scores.append(combined_score)

  state["masks"] = torch.stack(merged_masks).unsqueeze(1)
  state["boxes"] = torch.stack(merged_boxes)
  state["scores"] = torch.stack(merged_scores)

  return state


# ── Cropping helpers ─────────────────────────────────────────────────────────


def letterbox_image(
    image: np.ndarray,
    size: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
  """Resizes an image onto a fixed canvas without distortion.

  Scales the image to fit within the target size while preserving aspect
  ratio, then centers it on a filled canvas.

  Args:
      image: Input image as a numpy array of shape ``(H, W, 3)``.
      size: Target canvas size as ``(height, width)``.
      color: RGB fill color for the canvas padding.

  Returns:
      A letterboxed image as a numpy array of shape ``(size[0], size[1], 3)``.
  """
  image_height, image_width = image.shape[:2]
  target_height, target_width = size

  scale = min(target_width / image_width, target_height / image_height)
  new_width = int(image_width * scale)
  new_height = int(image_height * scale)

  resized = cv2.resize(
      image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
  )

  canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
  offset_x = (target_width - new_width) // 2
  offset_y = (target_height - new_height) // 2
  canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = (
      resized
  )

  return canvas


def get_padded_box(
    box: list[float],
    mask_shape: tuple[int, ...],
    buffer: int = _CROP_BUFFER,
) -> tuple[int, int, int, int]:
  """Expands a bounding box by a buffer, clamped to mask boundaries.

  Args:
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.
      mask_shape: Shape of the mask array, at least ``(H, W)``.
      buffer: Pixel buffer to expand on each side.

  Returns:
      A tuple ``(x_min, y_min, x_max, y_max)`` clamped to valid bounds.
  """
  mask_height, mask_width = mask_shape[:2]
  x_min, y_min, x_max, y_max = [int(round(v)) for v in box]

  x_min = max(0, x_min - buffer)
  y_min = max(0, y_min - buffer)
  x_max = min(mask_width, x_max + buffer)
  y_max = min(mask_height, y_max + buffer)

  return x_min, y_min, x_max, y_max


def crop_with_mean_background_blend(
    image_array: np.ndarray,
    mask: np.ndarray,
    box: list[float],
    size: tuple[int, int],
    background_color: tuple[int, int, int] = _IMAGENET_MEAN_RGB,
) -> Image.Image:
  """Returns a soft-edged letterboxed crop with blended background.

  Operates only on the cropped ROI instead of the full image, then blends
  using vectorized numpy operations.

  Args:
      image_array: RGB image as a numpy array of shape ``(H, W, 3)``.
      mask: Binary mask of shape ``(H, W)``.
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.
      size: Output size after letterboxing.
      background_color: RGB tuple used for the blended background.

  Returns:
      A letterboxed PIL image with soft-edged mask blending.
  """
  x_min, y_min, x_max, y_max = get_padded_box(box, mask.shape)

  roi_image = image_array[y_min:y_max, x_min:x_max]
  roi_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255

  kernel = np.ones((5, 5), np.uint8)
  dilated_mask = cv2.dilate(roi_mask, kernel, iterations=1)
  blurred_mask = cv2.GaussianBlur(dilated_mask, (5, 5), 0)

  alpha = blurred_mask.astype(np.float32) / 255.0
  alpha_three_channel = alpha[:, :, np.newaxis]
  background = np.array(background_color, dtype=np.float32)

  blended = roi_image.astype(np.float32) * alpha_three_channel + background * (
      1.0 - alpha_three_channel
  )
  blended = blended.astype(np.uint8)

  letterboxed = letterbox_image(blended, size=size, color=background_color)
  return Image.fromarray(letterboxed)


def crop_masked_image(
    image_array: np.ndarray,
    mask: np.ndarray,
    box: list[float],
    size: tuple[int, int],
) -> Image.Image:
  """Returns a hard-masked letterboxed crop with black background.

  Args:
      image_array: RGB image as a numpy array of shape ``(H, W, 3)``.
      mask: Binary mask of shape ``(H, W)``.
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.
      size: Output size after letterboxing.

  Returns:
      A letterboxed PIL image with black background outside the mask.
  """
  x_min, y_min, x_max, y_max = get_padded_box(box, mask.shape)

  mask_three_channel = mask[:, :, None]
  masked_image = np.where(mask_three_channel, image_array, 0)
  crop = masked_image[y_min:y_max, x_min:x_max]

  letterboxed = letterbox_image(crop, size=size)
  return Image.fromarray(letterboxed)


def crop_raw_masked_image(
    image_array: np.ndarray,
    mask: np.ndarray,
    box: list[float],
) -> Optional[Image.Image]:
  """Returns a hard-masked crop at exact box size with no letterboxing.

  Args:
      image_array: RGB image as a numpy array of shape ``(H, W, 3)``.
      mask: Binary mask of shape ``(H, W)``.
      box: Bounding box as ``[x_min, y_min, x_max, y_max]``.

  Returns:
      A PIL image cropped to the bounding box with black background outside
      the mask, or ``None`` if the box is degenerate.
  """
  x_min, y_min, x_max, y_max = map(round, box)

  x_min = max(0, x_min)
  y_min = max(0, y_min)
  x_max = min(image_array.shape[1], x_max)
  y_max = min(image_array.shape[0], y_max)

  if x_max <= x_min or y_max <= y_min:
    return None

  mask_three_channel = mask[:, :, None]
  masked_image = np.where(mask_three_channel, image_array, 0)
  crop = masked_image[y_min:y_max, x_min:x_max]

  return Image.fromarray(crop)


# ── Mask hole filling ────────────────────────────────────────────────────────


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
  """Fills all interior holes in a binary mask using border flood-fill.

  More robust than morphological closing, which only fills holes smaller
  than the structuring element. This fills all holes regardless of size.

  Algorithm:
      1. Pad the mask and flood-fill background from the corner.
      2. Any zero-pixel not reached by flood fill is an interior hole.
      3. Union the original mask with the unreached region.

  Args:
      mask: Binary mask of shape ``(H, W)``, dtype ``bool`` or ``uint8``.

  Returns:
      A hole-filled binary mask of the same shape, dtype ``bool``.
  """
  mask_uint8 = np.asarray(mask).astype(np.uint8) * 255

  height, width = mask_uint8.shape
  padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
  padded[1 : height + 1, 1 : width + 1] = mask_uint8

  flood_filled = padded.copy()
  cv2.floodFill(flood_filled, mask=None, seedPoint=(0, 0), newVal=255)

  flood_filled = flood_filled[1 : height + 1, 1 : width + 1]
  interior_holes = cv2.bitwise_not(flood_filled)

  filled = cv2.bitwise_or(mask_uint8, interior_holes)
  return filled.astype(bool)


# ── Crop iterator and debug viewer ───────────────────────────────────────────


def process_detections(
    image: Image.Image,
    state: dict[str, Any],
    score_threshold: float,
    crop_size: tuple[int, int],
) -> Iterator[tuple[int, Optional[Image.Image], Image.Image, Image.Image]]:
  """Yields raw, masked, and blended crops for each valid detection.

  Args:
      image: Input RGB PIL image.
      state: SAM output dict with ``'masks'``, ``'boxes'``, ``'scores'`` keys.
      score_threshold: Minimum confidence score to include a detection.
      crop_size: Target letterbox size ``(height, width)`` for the letterboxed
        variants.

  Yields:
      A tuple ``(detection_index, raw_crop, masked_crop, blended_crop)`` for
      each detection above the score threshold.
  """
  image_array = np.array(image)

  for detection_index, mask_tensor in enumerate(state["masks"]):
    score = state["scores"][detection_index].item()
    if score < score_threshold:
      continue

    mask = np.squeeze(mask_tensor)
    mask = fill_mask_holes(mask)
    box = state["boxes"][detection_index].tolist()

    raw_crop = crop_raw_masked_image(image_array, mask, box)
    masked_crop = crop_masked_image(image_array, mask, box, size=crop_size)
    blended_crop = crop_with_mean_background_blend(
        image_array, mask, box, size=crop_size
    )

    yield detection_index, raw_crop, masked_crop, blended_crop


def display_crop_thumbnails(
    crop_pairs: list[Any],
    state: dict[str, Any],
    crop_type: str = "blended",
    columns_per_row: int = 5,
    thumbnail_size: int = 3,
) -> None:
  """Displays detection crops as a grid of labeled thumbnails.

  Args:
      crop_pairs: List of tuples from ``process_detections``, each containing
        ``(detection_index, raw_crop, masked_crop, blended_crop)``.
      state: SAM output dict containing ``'scores'``.
      crop_type: Which crop to display. One of ``'raw'``, ``'masked'``,
        ``'blended'``.
      columns_per_row: Maximum number of thumbnails per row.
      thumbnail_size: Size of each thumbnail in inches.

  Raises:
      ValueError: If ``crop_type`` is not one of the allowed values.
  """
  crop_type_index = {"raw": 1, "masked": 2, "blended": 3}
  if crop_type not in crop_type_index:
    raise ValueError(f"crop_type must be one of {list(crop_type_index.keys())}")

  idx = crop_type_index[crop_type]
  valid_pairs = [pair for pair in crop_pairs if pair[idx] is not None]

  total_crops = len(valid_pairs)
  if total_crops == 0:
    print("No valid crops to display.")
    return

  num_columns = min(total_crops, columns_per_row)
  num_rows = math.ceil(total_crops / num_columns)

  _, axes = plt.subplots(
      num_rows,
      num_columns,
      figsize=(num_columns * thumbnail_size, num_rows * thumbnail_size),
  )

  if total_crops == 1:
    axes = np.array([axes])
  axes = axes.flatten()

  for axis_index, pair in enumerate(valid_pairs):
    detection_index = pair[0]
    crop_image = pair[crop_type_index[crop_type]]
    score = state["scores"][detection_index].item()

    axes[axis_index].imshow(crop_image)
    axes[axis_index].set_title(f"#{detection_index}  score: {score:.2f}")
    axes[axis_index].axis("off")

  for axis_index in range(total_crops, len(axes)):
    axes[axis_index].axis("off")

  plt.tight_layout()
  plt.show()
