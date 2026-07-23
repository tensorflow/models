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

"""SAM3 detection and segmentation management."""

import collections
from collections.abc import Mapping
import logging
import pathlib
from typing import Any, Self

import numpy
from PIL import Image
import supervision
import torch

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader

try:
  from sam3 import model_builder as sam3_model_builder  # pylint: disable=g-import-not-at-top
  from sam3.model import sam3_image_processor  # pylint: disable=g-import-not-at-top
except ImportError:
  sam3_model_builder = None
  sam3_image_processor = None

_LOGGER = logging.getLogger(__name__)

# Prompt whose detections should have their contained boxes merged.
_MERGE_PROMPT = "packets"

# Intermediate state entries dropped before returning to callers to reduce
# memory footprint. These are set by the SAM3 processor during inference and
# are not needed downstream.
_INFERENCE_KEYS_TO_DROP = frozenset(
    ["backbone_out", "geometric_prompt", "image_embeddings"]
)

# State entries that are per-detection arrays, kept in lockstep after any
# filtering step.
_STATE_ARRAY_KEYS = frozenset({"masks", "masks_logits", "boxes", "scores"})


class SAM3Detector:
  """Runs SAM3 inference and post-processes detections.

  The detector loads the SAM3 model onto the resolved device and exposes
  `detect` for running inference on a single image with a text prompt.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      processor: Any,
      device: torch.device,
      prompt_config: config_loader.PromptConfig,
  ):
    """Initializes the SAM3 detector with pre-built dependencies.

    Args:
      model: The SAM3 PyTorch image model.
      processor: The SAM3 image processor instance (`Sam3Processor`).
      device: Resolved PyTorch device where the model resides (`torch.device`).
      prompt_config: Detection and post-processing thresholds (`PromptConfig`).
    """
    self._model = model
    self._processor = processor
    self._device = device
    self._config = prompt_config

  @classmethod
  def from_checkpoint(
      cls,
      checkpoint_path: pathlib.Path,
      device: str,
      prompt_config: config_loader.PromptConfig,
  ) -> Self:
    """Constructs a SAM3Detector by loading a model from a checkpoint.

    Args:
      checkpoint_path: Path to the SAM3 model checkpoint.
      device: Requested device string (e.g., 'cuda', 'cpu', 'mps').
      prompt_config: Detection and post-processing thresholds for the active
        prompt.

    Returns:
      An initialized `SAM3Detector` instance on the resolved device.

    Raises:
      ImportError: If the 'sam3' package is not installed or available.
    """
    if sam3_model_builder is None or sam3_image_processor is None:
      raise ImportError(
          "The 'sam3' package is not installed or available in python path. "
          "Cannot construct SAM3Detector from checkpoint."
      )

    resolved_device = _resolve_device(device)
    model = sam3_model_builder.build_sam3_image_model(
        checkpoint_path=str(checkpoint_path)
    )
    model.to(device=resolved_device)
    processor = sam3_image_processor.Sam3Processor(
        model,
        confidence_threshold=prompt_config.confidence_threshold,
    )
    return cls(
        model=model,
        processor=processor,
        device=resolved_device,
        prompt_config=prompt_config,
    )

  def detect(self, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
    """Runs SAM3 inference and returns a CPU-bound state dictionary.

    Post-processing (containment filtering, and for the merge prompt also box
    merging) is performed on the active device to keep tensor operations on
    the GPU where possible. The final state is moved to CPU before return.

    Args:
      image: PIL RGB image to detect on.
      prompt: Text prompt to condition the detector.

    Returns:
      A dict containing per-detection tensors on CPU. Guaranteed keys include
      those in _STATE_ARRAY_KEYS.
    """
    with torch.no_grad(), torch.autocast(
        self._device.type, dtype=torch.float16
    ):
      state = self._processor.set_image(image)
      state = self._processor.set_text_prompt(state=state, prompt=prompt)

    # Post-process on the active device before the CPU transfer to avoid
    # unnecessary device round-trips.
    state = self._filter_contained_sub_masks(state)

    if prompt == _MERGE_PROMPT:
      state = self._merge_contained_boxes(state)

    for key in _INFERENCE_KEYS_TO_DROP:
      state.pop(key, None)

    return self._move_state_to_cpu(state)

  def to_supervision_detections(
      self, state: Mapping[str, torch.Tensor]
  ) -> supervision.Detections:
    """Converts a SAM3 state dict into supervision.Detections.

    Boxes whose score falls below `prompt_config.score_threshold` are
    filtered out. Returns an empty Detections object when no box survives.

    Args:
      state: State dictionary as returned by `detect`.

    Returns:
      Detections containing the surviving boxes and their confidences.
    """
    boxes = state["boxes"].numpy().astype(numpy.float32)
    scores = state["scores"].numpy().astype(numpy.float32)

    # SAM3 uses a 1-D empty tensor to signal "no detections"; normalize to a
    # (0, 4) shape so the slice below stays valid.
    if boxes.ndim == 1:
      boxes = boxes.reshape(0, 4) if boxes.size == 0 else boxes.reshape(-1, 4)

    keep_mask = scores >= self._config.score_threshold
    kept_boxes = boxes[keep_mask]
    kept_scores = scores[keep_mask]

    if kept_boxes.shape[0] == 0:
      return supervision.Detections.empty()

    return supervision.Detections(
        xyxy=kept_boxes,
        confidence=kept_scores,
        class_id=numpy.zeros(kept_boxes.shape[0], dtype=int),
    )

  def _move_state_to_cpu(
      self, inference_state: dict[str, Any]
  ) -> dict[str, Any]:
    """Recursively moves every tensor in the state dict to CPU in place.

    Args:
      inference_state: State dict returned by the SAM3 processor. Mutated in
        place.

    Returns:
      The same dict reference, for call-site convenience.
    """
    for key, value in inference_state.items():
      if isinstance(value, torch.Tensor):
        inference_state[key] = value.cpu()
      elif isinstance(value, dict):
        self._move_state_to_cpu(value)
    return inference_state

  def _filter_contained_sub_masks(
      self, state: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Removes smaller masks that are largely contained in bigger masks.

    Args:
      state: State dict; mutated in place for the per-detection arrays.

    Returns:
      The same state dict reference.
    """
    masks = state["masks"]
    number_of_masks = masks.shape[0]
    if number_of_masks == 0:
      return state

    flat_masks = masks.view(number_of_masks, -1).float()
    pairwise_intersection = flat_masks @ flat_masks.T

    # TODO(umairsabir): Vectorize containment filtering using PyTorch's indexing
    # and boolean operations over pairwise_intersection if nested loops become a
    # bottleneck under large batches or many prompts.
    indices_to_remove: set[int] = set()
    for outer_index in range(number_of_masks):
      if outer_index in indices_to_remove:
        continue
      for inner_index in range(outer_index + 1, number_of_masks):
        if inner_index in indices_to_remove:
          continue

        outer_area = pairwise_intersection[outer_index, outer_index].item()
        inner_area = pairwise_intersection[inner_index, inner_index].item()
        if outer_area <= inner_area:
          smaller_index = outer_index
          smaller_area = outer_area
        else:
          smaller_index = inner_index
          smaller_area = inner_area

        if smaller_area == 0:
          indices_to_remove.add(smaller_index)
          if smaller_index == outer_index:
            break
          continue

        intersection = pairwise_intersection[outer_index, inner_index].item()
        if (intersection / smaller_area) > self._config.containment_threshold:
          indices_to_remove.add(smaller_index)
          if smaller_index == outer_index:
            break

    kept_indices = sorted(set(range(number_of_masks)) - indices_to_remove)
    keep_tensor = torch.tensor(
        kept_indices, dtype=torch.long, device=masks.device
    )

    for key in _STATE_ARRAY_KEYS:
      state[key] = state[key][keep_tensor]

    return state

  def _merge_contained_boxes(
      self, state: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Merges detections where a smaller box is contained in a larger one.

    Merging combines masks (element-wise OR), takes the axis-aligned union
    of boxes, and sums the scores (clamped to 1.0).

    Args:
      state: State dict; mutated in place.

    Returns:
      The same state dict reference.
    """
    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    if len(scores) == 0:
      return state

    number_of_detections = len(masks)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    is_absorbed = torch.zeros(
        number_of_detections, dtype=torch.bool, device=boxes.device
    )
    absorb_target = list(range(number_of_detections))

    for outer_index in range(number_of_detections):
      if is_absorbed[outer_index]:
        continue
      for inner_index in range(outer_index + 1, number_of_detections):
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
          absorb_target[smaller_index] = larger_index
          if smaller_index == outer_index:
            break
          continue

        containment_ratio = intersection_area / smaller_area
        if containment_ratio > self._config.merge_overlap_threshold:
          is_absorbed[smaller_index] = True
          absorb_target[smaller_index] = larger_index
          if smaller_index == outer_index:
            break

    groups: collections.defaultdict[int, list[int]] = collections.defaultdict(
        list
    )
    for detection_index in range(number_of_detections):
      if is_absorbed[detection_index]:
        target = absorb_target[detection_index]
      else:
        target = detection_index
      groups[target].append(detection_index)

    merged_masks: list[torch.Tensor] = []
    merged_boxes: list[torch.Tensor] = []
    merged_scores: list[torch.Tensor] = []
    for member_indices in groups.values():
      member_tensor = torch.tensor(
          member_indices, dtype=torch.long, device=boxes.device
      )
      merged_masks.append(masks[member_tensor].squeeze(1).any(dim=0))
      group_boxes = boxes[member_tensor]
      merged_boxes.append(
          torch.stack([
              group_boxes[:, 0].min(),
              group_boxes[:, 1].min(),
              group_boxes[:, 2].max(),
              group_boxes[:, 3].max(),
          ])
      )
      summed_score = scores[member_tensor].sum().item()
      merged_scores.append(
          torch.tensor(min(summed_score, 1.0), device=scores.device)
      )

    state["masks"] = torch.stack(merged_masks).unsqueeze(1)
    state["boxes"] = torch.stack(merged_boxes)
    state["scores"] = torch.stack(merged_scores)
    return state


def _resolve_device(requested_device: str) -> torch.device:
  """Resolves the requested device string to an available torch.device.

  Falls back to CPU with a warning if the requested device is not available.

  Args:
    requested_device: Device string requested by the caller.

  Returns:
    The resolved torch.device.
  """
  if requested_device.startswith("cuda") and not torch.cuda.is_available():
    _LOGGER.warning(
        "Requested device '%s' but CUDA is not available; falling back to CPU.",
        requested_device,
    )
    return torch.device("cpu")

  if requested_device == "mps" and not torch.backends.mps.is_available():
    _LOGGER.warning(
        "Requested device 'mps' but MPS is not available; falling back to CPU."
    )
    return torch.device("cpu")

  return torch.device(requested_device)
