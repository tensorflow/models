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

"""Tracking, image blending, and track-level classification voting logic."""

import collections
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import tqdm
import trackers

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader
from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import dinov3_classifier


class TrackManager:
  """Manages object trajectories, blended crop extraction, and final label voting."""

  def __init__(
      self,
      tracking_config: config_loader.TrackingConfig,
      cropping_config: config_loader.CroppingConfig,
      vis_config: config_loader.VisualizationConfig,
      tracker: Any | None = None,
  ) -> None:
    """Initializes TrackManager.

    Args:
      tracking_config: Configuration for tracking parameters.
      cropping_config: Configuration for crop size and padding buffer.
      vis_config: Configuration for visualization and background blend color.
      tracker: Optional tracker instance to inject. If None, an instance of
        ByteTrackTracker is constructed from tracking_config.
    """
    self._tracking_enabled = tracking_config.enable
    if tracker is not None:
      self._tracker = tracker
    else:
      self._tracker = trackers.ByteTrackTracker(
          minimum_iou_threshold=tracking_config.bytetrack_minimum_iou_threshold,
          minimum_consecutive_frames=tracking_config.bytetrack_minimum_consecutive_frames,
      )
    self._crop_size = cropping_config.crop_size
    self._buffer = cropping_config.crop_buffer_pixels
    self._bg_color = vis_config.background_blend_color_rgb
    self._crop_records: dict[int, list[dict[str, Any]]] = {}
    self._next_standalone_id = 1

  def reset(self) -> None:
    """Resets tracker ID counter and clears all collected crop records.

    Call this when switching to a new video or scene so that track IDs in
    the next session start from 1 and no crops carry over from the
    previous session.
    """
    self._tracker.reset()
    self._crop_records.clear()
    self._next_standalone_id = 1

  def update_and_extract_crops(
      self,
      detections: Any,
      state: dict[str, torch.Tensor],
      image: Image.Image,
      frame_name: str,
  ) -> tuple[Any, list[float]]:
    """Updates tracks and extracts image crops.

    When tracking is disabled in the config, ByteTrack is bypassed
    entirely and every detection in the current frame is assigned a
    fresh sequential ID. The returned `unassigned_scores` list is always
    empty in that mode because no detection is ever dropped by the
    tracker.

    Args:
      detections: Detection object (e.g. supervision.Detections) for the frame.
      state: Detector state mapping containing bounding boxes and masks.
      image: Full-frame PIL image.
      frame_name: Identifier or filename for the current frame.

    Returns:
      A tuple of (detections, unassigned_scores).
    """
    if self._tracking_enabled:
      return self._update_with_tracking(detections, state, image, frame_name)
    return self._update_without_tracking(detections, state, image, frame_name)

  def _update_with_tracking(
      self,
      detections: Any,
      state: dict[str, torch.Tensor],
      image: Image.Image,
      frame_name: str,
  ) -> tuple[Any, list[float]]:
    """Runs ByteTrack and extracts crops for every successfully assigned detection."""
    pre_update_scores = (
        detections.confidence.copy() if len(detections) > 0 else None
    )
    detections = self._tracker.update(detections)

    unassigned_scores = []
    if pre_update_scores is not None and detections.tracker_id is not None:
      unassigned_scores = [
          float(score)
          for score, tid in zip(pre_update_scores, detections.tracker_id)
          if int(tid) == -1
      ]

    if len(detections) == 0 or detections.tracker_id is None:
      return detections, unassigned_scores

    state_boxes = state["boxes"].numpy().astype(np.float32)
    det_boxes = detections.xyxy.astype(np.float32)

    for row_idx in range(len(detections)):
      tracker_id = int(detections.tracker_id[row_idx])
      if tracker_id == -1:
        continue

      matching_rows = np.where(
          np.all(np.isclose(state_boxes, det_boxes[row_idx], atol=1e-3), axis=1)
      )[0]

      if matching_rows.size == 0:
        continue

      crop = self._extract_blended_crop(image, state, int(matching_rows[0]))
      self._crop_records.setdefault(tracker_id, []).append({
          "frame_name": frame_name,
          "crop": crop,
      })

    return detections, unassigned_scores

  def _update_without_tracking(
      self,
      detections: Any,
      state: dict[str, torch.Tensor],
      image: Image.Image,
      frame_name: str,
  ) -> tuple[Any, list[float]]:
    """Assigns a fresh sequential ID to every detection and extracts its crop.

    ByteTrack is bypassed entirely. Detection box order is assumed to
    match the detector state row order (they originate from the same
    `to_supervision_detections` call upstream), so the mask for each
    detection is looked up positionally.

    Args:
      detections: Detection object (e.g. supervision.Detections) for the frame.
      state: Detector state mapping containing bounding boxes and masks.
      image: Full-frame PIL image.
      frame_name: Identifier or filename for the current frame.

    Returns:
      A tuple of (detections, unassigned_scores).
    """
    if len(detections) == 0:
      return detections, []

    num_detections = len(detections)
    assigned_ids = np.arange(
        self._next_standalone_id,
        self._next_standalone_id + num_detections,
        dtype=int,
    )
    detections.tracker_id = assigned_ids
    self._next_standalone_id += num_detections

    for row_idx in range(num_detections):
      tracker_id = int(assigned_ids[row_idx])
      crop = self._extract_blended_crop(image, state, row_idx)
      self._crop_records.setdefault(tracker_id, []).append({
          "frame_name": frame_name,
          "crop": crop,
      })

    return detections, []

  def classify_all_tracks(
      self, classifier: dinov3_classifier.DINOv3Classifier, batch_size: int
  ) -> dict[int, list[dict[str, Any]]]:
    """Runs batch classification for every collected track crop with a progress bar."""
    track_predictions = {}
    total_crops = sum(len(records) for records in self._crop_records.values())

    progress_bar = tqdm.tqdm(
        total=total_crops, desc="Classifying crops", unit="crop"
    )

    for tracker_id, records in self._crop_records.items():
      progress_bar.set_postfix_str(f"track {tracker_id:04d}")
      per_crop_preds = []

      chunks = [
          records[i : i + batch_size]
          for i in range(0, len(records), batch_size)
      ]
      for chunk in chunks:
        pil_images = [r["crop"] for r in chunk]
        chunk_preds = classifier.predict_batch(pil_images)

        for record, pred in zip(chunk, chunk_preds):
          per_crop_preds.append({
              "frame_name": record["frame_name"],
              "crop": record["crop"],
              **pred,
          })
        progress_bar.update(len(chunk))

      track_predictions[tracker_id] = per_crop_preds

    progress_bar.close()
    return track_predictions

  def resolve_track_label(
      self, per_crop_predictions: list[dict[str, Any]]
  ) -> tuple[str, int]:
    """Resolves a final class by majority vote, tie-breaking by confidence sum."""
    vote_counter = collections.Counter(
        pred["predicted_class"] for pred in per_crop_predictions
    )
    confidence_sums = collections.defaultdict(float)

    for pred in per_crop_predictions:
      confidence_sums[pred["predicted_class"]] += pred[
          "predicted_probability_percent"
      ]

    highest_votes = max(vote_counter.values())
    top_classes = [
        cls for cls, votes in vote_counter.items() if votes == highest_votes
    ]

    if len(top_classes) == 1:
      return top_classes[0], highest_votes

    highest_conf = max(confidence_sums[cls] for cls in top_classes)
    tied_conf_classes = sorted([
        cls
        for cls in top_classes
        if math.isclose(confidence_sums[cls], highest_conf)
    ])

    return tied_conf_classes[0], highest_votes

  def _extract_blended_crop(
      self, image: Image.Image, state: dict[str, torch.Tensor], idx: int
  ) -> Image.Image:
    """Generates a soft-edged crop merged against the ImageNet mean background."""
    img_arr = np.array(image)

    # FIX: Explicitly convert the PyTorch tensor to a NumPy array before
    # manipulating.
    raw_mask_array = state["masks"][idx].numpy()
    mask = self._fill_mask_holes(np.squeeze(raw_mask_array))

    box = state["boxes"][idx].tolist()

    h, w = mask.shape
    x_min, y_min, x_max, y_max = [round(v) for v in box]
    x_min, y_min = max(0, x_min - self._buffer), max(0, y_min - self._buffer)
    x_max, y_max = min(w, x_max + self._buffer), min(h, y_max + self._buffer)

    roi_img = img_arr[y_min:y_max, x_min:x_max]
    roi_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255

    dilated = cv2.dilate(roi_mask, np.ones((5, 5), np.uint8), iterations=1)
    alpha = (
        cv2.GaussianBlur(dilated, (5, 5), 0).astype(np.float32)[
            :, :, np.newaxis
        ]
        / 255.0
    )
    bg = np.array(self._bg_color, dtype=np.float32)

    blended = (roi_img.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(
        np.uint8
    )

    canvas = np.full(
        (self._crop_size[0], self._crop_size[1], 3),
        self._bg_color,
        dtype=np.uint8,
    )
    ch, cw = canvas.shape[:2]
    bh, bw = blended.shape[:2]
    scale = min(cw / bw, ch / bh)
    rw, rh = int(bw * scale), int(bh * scale)

    resized = cv2.resize(blended, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ox, oy = (cw - rw) // 2, (ch - rh) // 2
    canvas[oy : oy + rh, ox : ox + rw] = resized

    return Image.fromarray(canvas)

  def _fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
    """Fills interior mask holes via flood-fill."""
    # Extra safety check to ensure it operates strictly as a numpy array
    mask_u8 = np.asarray(mask).astype(np.uint8) * 255
    h, w = mask_u8.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1 : h + 1, 1 : w + 1] = mask_u8

    flood_filled = padded.copy()
    cv2.floodFill(flood_filled, mask=None, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood_filled[1 : h + 1, 1 : w + 1])
    return cv2.bitwise_or(mask_u8, holes).astype(bool)
