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

"""RFDETR detection and segmentation management.

RFDETR is used purely as a "find and segment objects" engine: every
detection contributes a mask, box, and score, and the RFDETR class
prediction is intentionally discarded. Downstream tracking and DINOv3
grading are unchanged.

The returned state dict shape consumed by the rest of the pipeline
(TrackManager crop extraction, visualization, BigQuery ingestion):

  * ``masks``   torch.bool tensor of shape [N, 1, H, W]
  * ``boxes``   torch.float32 tensor of shape [N, 4], (x_min, y_min, x_max,
  y_max)
  * ``scores``  torch.float32 tensor of shape [N]

``merge_contained_boxes`` runs unconditionally because RFDETR has no
prompt to gate it on. ``optimize_for_inference`` is called once at model
build.
"""

import collections

import numpy as np
from PIL import Image
import supervision
import torch

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader

try:
  from rfdetr import RFDETRSegMedium  # pylint: disable=g-import-not-at-top
except ImportError:
  RFDETRSegMedium = None

_STATE_ARRAY_KEYS = ("masks", "boxes", "scores")


class RFDETRDetector:
  """Handles RFDETR inference and memory-safe post-processing.

  Public surface:

    * ``detect(image)`` returns the state dict.
    * ``to_supervision_detections(state)`` returns
      ``supervision.Detections`` with the score threshold applied.
  """

  def __init__(
      self,
      rfdetr_config: config_loader.RFDETRConfig,
      post_processing_config: config_loader.PostProcessingConfig,
  ):
    """Initializes the RFDETR model and applies inference optimization.

    Args:
      rfdetr_config: Validated RFDETR-model config (checkpoint, device, and the
        ``predict_threshold`` fed to ``RFDETR.predict``).
      post_processing_config: Validated post-processing thresholds (mask-in-mask
        filter, box-in-box merge, final score cutoff).

    Raises:
      RuntimeError: If the rfdetr package is not installed.
    """
    if RFDETRSegMedium is None:
      raise RuntimeError(
          "rfdetr is not installed. Please install it to use RFDETRDetector."
      )
    self._device = torch.device(
        rfdetr_config.device if torch.cuda.is_available() else "cpu"
    )
    self._predict_threshold = rfdetr_config.predict_threshold
    self._post_processing = post_processing_config
    self._model = RFDETRSegMedium(
        pretrain_weights=rfdetr_config.checkpoint_path
    )
    self._model.optimize_for_inference()

  def detect(self, image: Image.Image) -> dict[str, torch.Tensor]:
    """Runs RFDETR inference and returns a CPU-resident state dictionary.

    Args:
        image: PIL RGB image (already resized to the pipeline's target short
          side).

    Returns:
        State dict with keys ``masks``, ``boxes``, ``scores``. All
        tensors are on CPU.
    """
    image_width, image_height = image.size
    raw_detections = self._model.predict(
        image, threshold=self._predict_threshold
    )

    state = self._convert_detections_to_state(
        detections=raw_detections,
        image_height=image_height,
        image_width=image_width,
    )

    state = self._filter_contained_sub_masks(state)
    state = self._merge_contained_boxes(state)

    return state

  def to_supervision_detections(
      self, state: dict[str, torch.Tensor]
  ) -> supervision.Detections:
    """Converts a state dict to ``supervision.Detections``.

    Applies the configured ``score_threshold`` before returning. Every
    surviving detection is emitted with ``class_id=0`` so downstream
    tracking / voting does not depend on RFDETR's class predictions.

    Args:
        state: The state dict returned by ``detect``.

    Returns:
        A ``supervision.Detections`` instance.
    """
    boxes = state["boxes"].numpy().astype(np.float32)
    scores = state["scores"].numpy().astype(np.float32)

    if boxes.ndim == 1:
      boxes = boxes.reshape(0, 4)

    keep_mask = scores >= self._post_processing.score_threshold
    kept_boxes = boxes[keep_mask]
    kept_scores = scores[keep_mask]

    if kept_boxes.shape[0] == 0:
      return supervision.Detections.empty()

    return supervision.Detections(
        xyxy=kept_boxes,
        confidence=kept_scores,
        class_id=np.zeros(kept_boxes.shape[0], dtype=int),
    )

  def _convert_detections_to_state(
      self,
      detections: supervision.Detections,
      image_height: int,
      image_width: int,
  ) -> dict[str, torch.Tensor]:
    """Adapts a ``supervision.Detections`` into the pipeline state dict.

    RFDETR's ``class_id`` is intentionally discarded. Masks are
    reshaped to ``[N, 1, H, W]`` because the box-merge helper below
    calls ``masks.squeeze(1)``.

    Args:
        detections: The ``supervision.Detections`` returned by
          ``RFDETRSegMedium.predict``.
        image_height: Height of the image that produced the detections.
        image_width: Width of the image that produced the detections.

    Returns:
        A CPU-resident state dict.
    """
    if detections.mask is None or len(detections) == 0:
      masks = torch.zeros((0, 1, image_height, image_width), dtype=torch.bool)
      boxes = torch.zeros((0, 4), dtype=torch.float32)
      scores = torch.zeros((0,), dtype=torch.float32)
    else:
      masks = torch.from_numpy(detections.mask.astype(bool)).unsqueeze(1)
      boxes = torch.from_numpy(detections.xyxy.astype(np.float32))
      scores = torch.from_numpy(detections.confidence.astype(np.float32))

    return {
        "masks": masks,
        "boxes": boxes,
        "scores": scores,
    }

  def _filter_contained_sub_masks(
      self, state: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Removes smaller masks that are contained inside larger masks.

    Uses ``intersection / smaller_mask_area`` and drops the smaller
    of the two when the ratio exceeds
    ``post_processing.containment_threshold``.

    Args:
        state: State dict from ``_convert_detections_to_state``.

    Returns:
        The filtered state dict.
    """
    masks = state["masks"]
    num_masks = masks.shape[0]
    if num_masks == 0:
      return state

    flat_masks = masks.view(num_masks, -1).float()
    pairwise_intersection = flat_masks @ flat_masks.T

    indices_to_remove = set()
    for i in range(num_masks):
      if i in indices_to_remove:
        continue
      for j in range(i + 1, num_masks):
        if j in indices_to_remove:
          continue

        area_i = pairwise_intersection[i, i].item()
        area_j = pairwise_intersection[j, j].item()
        smaller_index, smaller_area = (
            (i, area_i) if area_i <= area_j else (j, area_j)
        )

        if smaller_area == 0:
          indices_to_remove.add(smaller_index)
          continue

        intersection = pairwise_intersection[i, j].item()
        if (
            intersection / smaller_area
        ) > self._post_processing.containment_threshold:
          indices_to_remove.add(smaller_index)

    keep_tensor = torch.tensor(
        sorted(set(range(num_masks)) - indices_to_remove),
        dtype=torch.long,
        device=masks.device,
    )

    for key in _STATE_ARRAY_KEYS:
      state[key] = state[key][keep_tensor]

    return state

  def _merge_contained_boxes(
      self, state: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Merges detections where a smaller box sits inside a larger one.

    Uses ``intersection_area / smaller_box_area`` and merges the
    smaller detection into the larger one when the ratio exceeds
    ``post_processing.merge_containment_threshold``. Runs
    unconditionally.

    The merged detection's mask is the union of the group's masks,
    its box is the enclosing box, and its score is the (clamped) sum
    of the group's scores.

    Args:
        state: State dict from ``_filter_contained_sub_masks``.

    Returns:
        The merged state dict.
    """
    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    if len(scores) == 0:
      return state

    num_detections = len(masks)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    is_absorbed = torch.zeros(
        num_detections, dtype=torch.bool, device=boxes.device
    )
    absorb_target = list(range(num_detections))

    for i in range(num_detections):
      if is_absorbed[i]:
        continue
      for j in range(i + 1, num_detections):
        if is_absorbed[j]:
          continue

        inter_x_min = torch.max(boxes[i, 0], boxes[j, 0])
        inter_y_min = torch.max(boxes[i, 1], boxes[j, 1])
        inter_x_max = torch.min(boxes[i, 2], boxes[j, 2])
        inter_y_max = torch.min(boxes[i, 3], boxes[j, 3])

        intersection_area = torch.clamp(
            inter_x_max - inter_x_min, min=0
        ) * torch.clamp(inter_y_max - inter_y_min, min=0)

        smaller_index, larger_index, smaller_area = (
            (i, j, box_areas[i])
            if box_areas[i] <= box_areas[j]
            else (j, i, box_areas[j])
        )

        if smaller_area == 0:
          is_absorbed[smaller_index] = True
          continue

        containment_ratio = intersection_area / smaller_area
        if (
            containment_ratio
            > self._post_processing.merge_containment_threshold
        ):
          is_absorbed[smaller_index] = True
          absorb_target[smaller_index] = larger_index

    groups: collections.defaultdict[int, list[int]] = collections.defaultdict(
        list
    )
    for i in range(num_detections):
      target = absorb_target[i] if is_absorbed[i] else i
      groups[target].append(i)

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
      merged_scores.append(
          torch.tensor(
              min(scores[member_tensor].sum().item(), 1.0),
              device=scores.device,
          )
      )

    state["masks"] = torch.stack(merged_masks).unsqueeze(1)
    state["boxes"] = torch.stack(merged_boxes)
    state["scores"] = torch.stack(merged_scores)
    return state
