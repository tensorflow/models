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

"""ObjectExtractor: SAM3-based object segmentation and crop extraction."""

from typing import Any
import cv2
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import supervision
import torch

_INFERENCE_KEYS_TO_DROP = frozenset([
    "backbone_out",
    "geometric_prompt",
    "image_embeddings",
])
_STATE_ARRAY_KEYS = ("masks", "masks_logits", "boxes", "scores")
_IMAGENET_MEAN_RGB = (124, 116, 104)
_CROP_BUFFER = 5


class ObjectExtractor:
  """Segments objects in an image using SAM3 and returns cropped detections.

  Wraps model loading, inference, filtering, merging, and cropping into a
  single reusable object. All stateless image-processing helpers are exposed
  as static methods so they can also be called without an instance.

  Usage:
      extractor = ObjectExtractor(
          checkpoint_path="sam3.pth",
          confidence_threshold=0.5,
          score_threshold=0.5,
          containment_threshold=0.5,
          max_short_side=1024,
          crop_size=(224, 224),
      )
      resized_image, state, detections = extractor.extract(pil_image, "bottle")
  """

  def __init__(
      self,
      checkpoint_path: str,
      confidence_threshold: float,
      score_threshold: float,
      containment_threshold: float,
      max_short_side: int,
      crop_size: tuple[int, int],
      device: str | None = None,
  ) -> None:
    """Load and initialize the SAM3 model and processor.

    Args:
        checkpoint_path: Path to the SAM3 weights file.
        confidence_threshold: Confidence threshold for the SAM3 processor.
        score_threshold: Minimum score to keep a detection.
        containment_threshold: Ratio above which a smaller mask is dropped if it
          is contained within a larger one.
        max_short_side: Maximum allowed length for the shorter dimension of the
          input image during inference.
        crop_size: Target (height, width) for cropped object images.
        device: Torch device string. Defaults to 'cuda' if available, otherwise
          'cpu'.
    """
    self._confidence_threshold = confidence_threshold
    self._score_threshold = score_threshold
    self._containment_threshold = containment_threshold
    self._max_short_side = max_short_side
    self._crop_size = crop_size
    self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self._model = (
        build_sam3_image_model(checkpoint_path=checkpoint_path)
        .eval()
        .to(device=self._device)
    )
    self._processor = Sam3Processor(
        self._model, confidence_threshold=self._confidence_threshold
    )

  # ------------------------------------------------------------------
  # Public pipeline
  # ------------------------------------------------------------------

  def extract(
      self, image: Image.Image, prompt: str
  ) -> tuple[Image.Image, dict[str, Any], supervision.Detections]:
    """Run the full extraction pipeline on a single image.

    Steps: resize → SAM3 inference → sub-mask filtering → detections.

    Args:
        image: Input RGB PIL image.
        prompt: Text prompt for SAM3.

    Returns:
        A tuple of (resized_image, inference_state, detections).
    """
    resized_image = self.resize_image_for_inference(image, self._max_short_side)
    state = self._run_inference(resized_image, prompt)
    state = self.filter_contained_sub_masks(state, self._containment_threshold)
    detections = self.convert_sam3_state_to_detections(
        state, score_threshold=self._score_threshold
    )
    return resized_image, state, detections

  # ------------------------------------------------------------------
  # Private inference helpers
  # ------------------------------------------------------------------

  def _run_inference(self, image: Image.Image, prompt: str) -> dict[str, Any]:
    """Runs SAM3 inference on the input image."""
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
      state = self._processor.set_image(image)
      state = self._processor.set_text_prompt(state=state, prompt=prompt)

    for key in _INFERENCE_KEYS_TO_DROP:
      state.pop(key, None)

    return self._move_state_to_cpu(state)

  def _move_state_to_cpu(
      self, inference_state: dict[str, Any]
  ) -> dict[str, Any]:
    """Recursively moves tensors in the state dict to CPU."""
    for key, value in inference_state.items():
      if isinstance(value, torch.Tensor):
        inference_state[key] = value.cpu()
      elif isinstance(value, dict):
        self._move_state_to_cpu(value)
    return inference_state

  def filter_contained_sub_masks(
      self, state: dict[str, Any], containment_threshold: float
  ) -> dict[str, Any]:
    """Removes smaller masks that are largely contained within larger masks.

    Args:
        state: Dict with 'masks', 'masks_logits', 'boxes', 'scores'. masks is a
          bool tensor of shape [N, H, W].
        containment_threshold: Ratio above which the smaller mask is dropped.

    Returns:
        Filtered state dict.
    """
    masks = state["masks"]
    num_masks = masks.shape[0]
    if num_masks == 0:
      return state

    flat_masks = masks.view(num_masks, -1).float()
    areas = flat_masks.sum(dim=1)
    pairwise_intersection = flat_masks @ flat_masks.T

    indices_to_remove: set[int] = set()
    for i in range(num_masks):
      if i in indices_to_remove:
        continue
      for j in range(i + 1, num_masks):
        if j in indices_to_remove:
          continue

        intersection = pairwise_intersection[i, j].item()
        area_i, area_j = areas[i].item(), areas[j].item()
        smaller_index = i if area_i <= area_j else j
        smaller_area = min(area_i, area_j)

        if smaller_area == 0:
          indices_to_remove.add(smaller_index)
          continue

        if intersection / smaller_area > containment_threshold:
          indices_to_remove.add(smaller_index)

    keep = torch.tensor(
        sorted(set(range(num_masks)) - indices_to_remove), dtype=torch.long
    )
    for key in _STATE_ARRAY_KEYS:
      state[key] = state[key][keep]
    return state

  @staticmethod
  def resize_image_for_inference(
      image: Image.Image, max_short_side: int
  ) -> Image.Image:
    """Resize so the short side does not exceed max_short_side (aspect-ratio safe).

    Args:
        image: Input RGB PIL image.
        max_short_side: Maximum allowed length for the shorter dimension.

    Returns:
        Resized PIL image, or the original if already within the limit.
    """
    w, h = image.size
    short_side = min(w, h)
    if short_side <= max_short_side:
      return image
    scale = max_short_side / short_side
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

  @staticmethod
  def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fills all interior holes in a binary mask via border flood-fill.

    More robust than morphological closing: fills holes of any size.

    Args:
        mask: Binary mask of shape (H, W), dtype bool or uint8.

    Returns:
        Hole-filled binary mask, dtype bool.
    """
    mask_u8 = np.asarray(mask).astype(np.uint8) * 255
    h, w = mask_u8.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1 : h + 1, 1 : w + 1] = mask_u8

    filled = padded.copy()
    cv2.floodFill(filled, mask=None, seedPoint=(0, 0), newVal=255)
    filled = filled[1 : h + 1, 1 : w + 1]

    interior_holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask_u8, interior_holes).astype(bool)

  @staticmethod
  def get_padded_box(
      box: list[float], mask_shape: tuple[int, ...], buffer: int = _CROP_BUFFER
  ) -> tuple[int, int, int, int]:
    """Expands a bounding box by buffer pixels, clamped to mask bounds.

    Args:
        box: [x_min, y_min, x_max, y_max].
        mask_shape: (H, W, ...) shape of the reference mask.
        buffer: Pixels to add on each side.

    Returns:
        (x_min, y_min, x_max, y_max) clamped to valid image bounds.
    """
    mask_h, mask_w = mask_shape[:2]
    x_min, y_min, x_max, y_max = map(round, box)
    return (
        max(0, x_min - buffer),
        max(0, y_min - buffer),
        min(mask_w, x_max + buffer),
        min(mask_h, y_max + buffer),
    )

  @staticmethod
  def letterbox_image(
      image: np.ndarray,
      size: tuple[int, int],
      color: tuple[int, int, int] = (0, 0, 0),
  ) -> np.ndarray:
    """Resize onto a fixed canvas while preserving aspect ratio.

    Args:
        image: (H, W, 3) numpy array.
        size: Target (height, width).
        color: RGB fill color for padding.

    Returns:
        Letterboxed numpy array of shape (size[0], size[1], 3).
    """
    ih, iw = image.shape[:2]
    th, tw = size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas

  @staticmethod
  def crop_with_mean_background_blend(
      image_array: np.ndarray,
      mask: np.ndarray,
      box: list[float],
      size: tuple[int, int],
      background_color: tuple[int, int, int] = _IMAGENET_MEAN_RGB,
  ) -> Image.Image:
    """Soft-edged letterboxed crop blended against a background color.

    Args:
        image_array: (H, W, 3) RGB numpy array.
        mask: Binary mask of shape (H, W).
        box: [x_min, y_min, x_max, y_max].
        size: Output (height, width) after letterboxing.
        background_color: RGB tuple for the blended background.

    Returns:
        Letterboxed PIL image with soft mask blending.
    """
    x_min, y_min, x_max, y_max = ObjectExtractor.get_padded_box(box, mask.shape)
    roi_image = image_array[y_min:y_max, x_min:x_max]
    roi_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(roi_mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated, (5, 5), 0)

    alpha = blurred.astype(np.float32) / 255.0
    bg = np.array(background_color, dtype=np.float32)
    blended = roi_image.astype(np.float32) * alpha[:, :, None] + bg * (
        1.0 - alpha[:, :, None]
    )

    lb = ObjectExtractor.letterbox_image(
        blended.astype(np.uint8), size=size, color=background_color
    )
    return Image.fromarray(lb)

  @staticmethod
  def convert_sam3_state_to_detections(
      state: dict[str, Any], score_threshold: float
  ) -> supervision.Detections:
    """Converts a SAM3 state dict to an ``sv.Detections`` object.

    Only detections with score >= ``score_threshold`` are kept. All
    detections are assigned ``class_id=0`` because the pipeline uses a
    single text prompt.

    Args:
        state: SAM3 state dict with 'boxes' and 'scores' tensors.
        score_threshold: Minimum score to keep a detection.

    Returns:
        ``sv.Detections`` with xyxy boxes, confidence scores, and
        ``class_id=0``.
    """
    boxes = state["boxes"].numpy().astype(np.float32)
    scores = state["scores"].numpy().astype(np.float32)

    if boxes.ndim == 1:
      boxes = boxes.reshape(-1, 4)

    keep_mask = scores >= score_threshold
    kept_boxes = boxes[keep_mask]
    kept_scores = scores[keep_mask]

    if kept_boxes.shape[0] == 0:
      return supervision.Detections.empty()

    class_ids = np.zeros(kept_boxes.shape[0], dtype=int)
    return supervision.Detections(
        xyxy=kept_boxes,
        confidence=kept_scores,
        class_id=class_ids,
    )

  @staticmethod
  def get_cropped_objects_for_each_tracking_id(
      image: Image.Image,
      state: dict[str, Any],
      detections: supervision.Detections,
      source_frame_name: str,
      crop_size: tuple[int, int],
      track_crop_records: dict[int, list[dict[str, Any]]],
  ) -> None:
    """Adds blended crops for assigned detections in a single frame.

    Detections without an assigned tracker_id (tracker_id == -1) are
    skipped. Detections from sv.Detections are matched back to rows of
    ``state`` by xyxy near-equality.

    Args:
        image: RGB PIL image used for SAM3 inference on this frame.
        state: SAM3 state dict with 'boxes' and 'masks' tensors on CPU.
        detections: sv.Detections returned by tracker.update() for this frame.
        source_frame_name: Filename (no directory) of the source frame, used to
          label thumbnails.
        crop_size: Output (height, width) for each blended crop.
        track_crop_records: Mutable dict keyed by tracker_id. Each value is a
          list of dicts shaped ``{'frame_name': str, 'crop': PIL.Image}``. New
          records are appended in place.
    """
    if len(detections) == 0:
      return
    if detections.tracker_id is None:
      return

    state_boxes = state["boxes"].numpy().astype(np.float32)
    detection_boxes = detections.xyxy.astype(np.float32)

    for detection_row in range(len(detections)):
      tracker_id = int(detections.tracker_id[detection_row])
      if tracker_id == -1:
        continue

      detection_box = detection_boxes[detection_row]
      matching_state_rows = np.where(
          np.all(
              np.isclose(state_boxes, detection_box, atol=1e-3),
              axis=1,
          )
      )[0]
      if matching_state_rows.size == 0:
        continue

      state_index = int(matching_state_rows[0])
      blended_crop = ObjectExtractor.extract_blended_crop(
          image=image,
          state=state,
          detection_index=state_index,
          crop_size=crop_size,
      )
      track_crop_records.setdefault(tracker_id, []).append({
          "frame_name": source_frame_name,
          "crop": blended_crop,
      })

  @staticmethod
  def extract_blended_crop(
      image: Image.Image,
      state: dict[str, Any],
      detection_index: int,
      crop_size: tuple[int, int],
  ) -> Image.Image:
    """Builds the blended crop variant for a single SAM3 detection.

    Args:
        image: RGB PIL image already resized for SAM3 inference.
        state: SAM3 state dict with 'masks' and 'boxes' tensors on CPU.
        detection_index: Row index into state['masks'] / state['boxes'].
        crop_size: Output (height, width) for the letterboxed crop.

    Returns:
        Blended PIL image of size crop_size.
    """
    image_array = np.array(image)
    raw_mask = np.squeeze(state["masks"][detection_index])
    filled_mask = ObjectExtractor.fill_mask_holes(raw_mask)
    box = state["boxes"][detection_index].tolist()
    blended_crop = ObjectExtractor.crop_with_mean_background_blend(
        image_array=image_array,
        mask=filled_mask,
        box=box,
        size=crop_size,
    )
    return blended_crop
