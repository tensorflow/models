# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Main bounding box detection and image segmentation logic."""

import math
from typing import Any, Optional
import warnings

from groundingdino.util import inference
import models_utils
import numpy as np
from sam2 import build_sam
from sam2 import sam2_image_predictor
import torch

# Suppress common warnings for a cleaner console output.
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ObjectDetectionSegmentation:
  """Encapsulates vision models for object detection and segmentation.

  This class provides a high-level API for using Grounding DINO and SAM2.
  Models are loaded into memory once during initialization to avoid redundant
  loading and improve performance for sequential processing tasks.

  Attributes:
    dino_model: The loaded Grounding DINO model.
    sam_predictor: The initialized SAM2 predictor instance.
    device: The PyTorch device (e.g., 'cuda' or 'cpu') the models run on.
  """

  def __init__(
      self,
      dino_config_path: str,
      dino_weights_path: str,
      sam_config_path: str,
      sam_checkpoint_path: str,
      device: str = 'cuda',
  ) -> None:
    """Initializes the vision pipeline by loading and setting up models.

    Args:
      dino_config_path: Path to the Grounding DINO configuration file.
      dino_weights_path: Path to the Grounding DINO model weights file.
      sam_config_path: Path to the SAM2 model configuration file.
      sam_checkpoint_path: Path to the SAM2 model checkpoint file.
      device: The hardware device to run models on (e.g., "cuda", "cpu").
    """
    self.device = torch.device(device)

    print('Loading Grounding DINO model...')
    self.dino_model = inference.load_model(dino_config_path, dino_weights_path)
    self.dino_model.to(self.device)
    print('✅ Grounding DINO model loaded.')

    print('Loading SAM2 model...')
    sam2_model = build_sam.build_sam2(
        sam_config_path, sam_checkpoint_path, device=self.device
    )
    self.sam_predictor = sam2_image_predictor.SAM2ImagePredictor(sam2_model)
    print('✅ SAM2 predictor initialized.')

  def _detect_objects(
      self,
      image_path: str,
      text_prompt: str,
      box_threshold: float = 0.25,
      text_threshold: float = 0.25,
  ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[str]]:
    """Detects objects in an image using Grounding DINO based on a prompt.

    Args:
      image_path: The file path to the input image.
      text_prompt: The text description of objects to detect.
      box_threshold: The confidence threshold for object bounding boxes.
      text_threshold: The confidence threshold for text-based labels.

    Returns:
      A tuple containing:
        - image: The original image loaded as a NumPy array.
        - boxes: Detected bounding boxes in CXCYWH format.
        - scores: Confidence scores for each detected box.
        - labels: Text labels corresponding to each box.
    """
    image, transformed_image = inference.load_image(image_path)
    transformed_image = transformed_image.to(self.device)

    boxes, scores, labels = inference.predict(
        model=self.dino_model,
        image=transformed_image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    return image, boxes, scores, labels

  def _segment_objects(
      self,
      image: np.ndarray,
      boxes: np.ndarray,
  ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    """Generates segmentation masks for batches of bounding boxes.

    Args:
      image: The source image as a NumPy array.
      boxes: A NumPy array of bounding boxes in [x1, y1, x2, y2] format.

    Returns:
      A tuple containing:
        - A list of boolean segmentation masks.
        - A list of confidence scores for each mask.
    """
    self.sam_predictor.set_image(image)

    # Stack all boxes for batched prediction
    # SAM2 expects shape (num_boxes, 4)
    batched_boxes = np.stack(boxes, axis=0)
    masks, scores, _ = self.sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=batched_boxes,
        multimask_output=False,
    )

    return [mask.squeeze() for mask in masks], list(scores)

  def _filter_boxes_by_area(
      self, boxes: np.ndarray, max_box_area: float
  ) -> list[np.ndarray]:
    """Filter bounding boxes by area to remove overly large detections.

    Args:
      boxes: Array of bounding boxes in [x1, y1, x2, y2] format.
      max_box_area: Maximum box area to keep

    Returns:
      List of valid bounding boxes that passed the area filter.
    """
    return [
        bbox
        for bbox in boxes
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < max_box_area
    ]

  def _filter_boxes_at_image_edge(
      self, boxes: list[np.ndarray], image_height: int, edge_tolerance: int = 5
  ) -> list[np.ndarray]:
    """Filter out bounding boxes that touch the top or bottom edges of the image.

    Objects at frame edges are likely partial views.
    Since each object appears in multiple frames, we can skip these.

    Args:
      boxes: List of bounding boxes in [x1, y1, x2, y2] format.
      image_height: Height of the image in pixels.
      edge_tolerance: Pixel tolerance for edge detection (default: 5).

    Returns:
      List of bounding boxes that don't touch top or bottom edges.
    """
    return [
        bbox
        for bbox in boxes
        if bbox[1] > edge_tolerance and bbox[3] < image_height - edge_tolerance
    ]

  def _calculate_texture_variance(
      self, image: np.ndarray, bbox: np.ndarray
  ) -> float:
    """Calculate texture variance within a bounding box.

    Args:
      image: The input image array.
      bbox: Bounding box in [x1, y1, x2, y2] format.

    Returns:
      Standard deviation of pixel intensities (grayscale) within the box.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    crop = image[y1:y2, x1:x2]

    # Return and convert to grayscale if not already
    return float(
        np.std(np.mean(crop, axis=2) if len(crop.shape) == 3 else crop)
    )

  def _filter_boxes_by_texture(
      self,
      boxes: list[np.ndarray],
      image: np.ndarray,
      min_texture_variance: float = 20.0,
  ) -> tuple[list[np.ndarray], dict[int, float]]:
    """Filter out boxes with very low texture variance.

    This filter exists to remove misdetections caused by lighting artifacts.

    Args:
      boxes: List of bounding boxes in [x1, y1, x2, y2] format.
      image: The input image array.
      min_texture_variance: Minimum std dev of pixel intensities to keep.

    Returns:
      Tuple of
      (filtered boxes, dict of texture variances).
    """
    variance_dict = {}
    filtered_boxes = []

    for i, bbox in enumerate(boxes):
      variance = self._calculate_texture_variance(image, bbox)
      variance_dict[i] = variance

      if variance <= min_texture_variance:
        print(f'  Box {i} filtered (low texture variance: {variance:.2f})')
      else:
        filtered_boxes.append(bbox)

    return filtered_boxes, variance_dict

  def _filter_valid_boxes(
      self,
      boxes: np.ndarray,
      image: np.ndarray,
      image_shape: tuple[int, ...],
      max_box_to_area_ratio: float,
      min_texture_variance: float = 10.0,
  ) -> tuple[list[np.ndarray], dict[int, float]]:
    """Apply area, edge, and texture-based filtering to boxes.

    Args:
      boxes: Array of bounding boxes in [x1, y1, x2, y2] format.
      image: The input image array (for texture analysis).
      image_shape: Shape of the image (height, width, channels).
      max_box_to_area_ratio: Maximum box area as ratio of image area.
      min_texture_variance: Minimum texture variance to keep box.

    Returns:
      Tuple of (list of valid bounding boxes, dict of texture variances).
    """
    # Filter boxes by overall image area
    image_area = math.prod(image_shape[:2])
    valid_boxes = self._filter_boxes_by_area(
        boxes, image_area * max_box_to_area_ratio
    )
    if not valid_boxes:
      print('No objects passed area filter.')
      return [], {}

    # Filter boxes by intersection with edge of image
    image_height = image_shape[0]
    valid_boxes = self._filter_boxes_at_image_edge(valid_boxes, image_height)
    if not valid_boxes:
      print('No objects passed edge filter.')
      return [], {}

    # Filter boxes by texture variance
    valid_boxes, variance_dict = self._filter_boxes_by_texture(
        valid_boxes, image, min_texture_variance
    )
    if not valid_boxes:
      print('No objects passed texture filter.')
      return [], {}

    return valid_boxes, variance_dict

  def detect_and_segment(
      self,
      image_path: str,
      text_prompt: str,
      max_box_to_area_ratio: float = 0.25,
      min_texture_variance: float = 10.0,
  ) -> Optional[dict[str, Any]]:
    """Runs detection and batched segmentation pipeline on an image.

    This first uses GroundingDINO to extract bboxes from an image
    based on a prompt, then passes all those boxes to SAM2 for
    mask extraction.

    Args:
      image_path: The file path to the input image.
      text_prompt: The text description of objects to use for box detection
      max_box_to_area_ratio: Maximum box area as ratio of image area
      min_texture_variance: Minimum texture variance to keep box

    Returns:
      A dictionary containing the processed data ('image', 'boxes', 'masks',
      'texture_variances') or None if no objects were detected.
    """
    print(f"\nProcessing '{image_path}'")
    image, cxchywh_boxes, _, _ = self._detect_objects(image_path, text_prompt)

    if cxchywh_boxes.shape[0] == 0:
      print('No objects detected.')
      return None

    xyxy_boxes = models_utils.convert_boxes_cxcywh_to_xyxy(
        cxchywh_boxes, image.shape
    )
    valid_boxes, variance_dict = self._filter_valid_boxes(
        xyxy_boxes,
        image,
        image.shape,
        max_box_to_area_ratio,
        min_texture_variance,
    )
    if not valid_boxes:
      return None

    masks, _ = self._segment_objects(image, valid_boxes)
    print(f'Segmentation complete. Generated {len(masks)} masks.')

    return {
        'image': image,
        'boxes': valid_boxes,
        'masks': masks,
        'texture_variances': variance_dict,
    }
