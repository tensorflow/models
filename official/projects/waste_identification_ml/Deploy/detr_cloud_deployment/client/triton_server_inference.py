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
"""Prediction from the Triton server."""

import os
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import tritonclient.http as httpclient


def _sigmoid(x: np.ndarray) -> np.ndarray:
  """Applies sigmoid function to an array of scores."""
  return 1 / (1 + np.exp(-x))


def _box_cxcywh_to_xyxyn(x: np.ndarray) -> np.ndarray:
  """Converts bounding boxes from cxcywh format to xyxyn format."""
  cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
  xmin = cx - w / 2
  ymin = cy - h / 2
  xmax = cx + w / 2
  ymax = cy + h / 2
  return np.stack([xmin, ymin, xmax, ymax], axis=-1)


class TritonObjectDetector:
  """Client for performing object detection inference using a Triton server.

  This class handles preprocessing, making inference requests to a Triton HTTP
  server, and post-processing the results, including scaling bounding boxes
  and masks.
  """

  def __init__(
      self,
      server_url: str = 'localhost:8000',
      model_name: str = 'detection_model',
      input_size: tuple[int, int] = (432, 432),
      verbose: bool = False,
  ):
    """Initializes the Triton Client and Model configuration."""
    self.client = httpclient.InferenceServerClient(
        url=server_url, verbose=verbose
    )
    self.model_name = model_name
    self.input_size = input_size

    # Normalization constants
    self.means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    self.stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

  def _resize_mask_batch(
      self, masks: np.ndarray, target_dims: Tuple[int, int]
  ) -> np.ndarray:
    """Resizes a batch of masks to the target dimensions."""
    target_w, target_h = target_dims
    masks_transposed = np.transpose(masks, (1, 2, 0))

    resized_batch = cv2.resize(
        masks_transposed, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )

    # If N=1, cv2.resize might drop the last dim, so we ensure 3D
    if resized_batch.ndim == 2:
      return resized_batch[np.newaxis, ...]

    return np.transpose(resized_batch, (2, 0, 1))

  def _scale_bbox_and_masks(
      self, results: Dict[str, Any], target_dims: Tuple[int, int]
  ) -> Dict[str, Any]:
    """Scales normalized boxes and small mask logits to target dimensions."""
    target_w, target_h = target_dims

    # Scale Bounding Boxes
    results['xyxy'][..., [0, 2]] *= target_w
    results['xyxy'][..., [1, 3]] *= target_h

    # Scale Masks
    if results['masks'] is not None:
      rescaled_masks = self._resize_mask_batch(
          results['masks'], target_dims
      )
      results['masks'] = (rescaled_masks > 0).astype(bool)

    return results

  def _get_input_batch_for_inference(self, image_path: str) -> np.ndarray:
    """Preprocesses an image for Triton inference.

    Loads an image, resizes it, converts it to RGB, normalizes pixel values,
    and transposes it to the channel-first format expected by the model.

    Args:
      image_path: The path to the input image file.

    Returns:
      A numpy array representing the preprocessed image, ready for inference.

    Raises:
      FileNotFoundError: If the image file does not exist.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
      raise FileNotFoundError(f'Image not found at {image_path}')

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(
        rgb_image, self.input_size, interpolation=cv2.INTER_AREA
    )

    # Normalize: (pixel / 255 - mean) / std
    float_image = resized_image.astype(np.float32) / 255.0
    normalized_image = (float_image - self.means) / self.stds

    # Transpose to CHW and add batch dimension
    transposed_image = np.transpose(normalized_image, (2, 0, 1))
    batched_image = np.expand_dims(transposed_image, axis=0).astype(np.float32)
    return batched_image

  def _reformat_triton_output_to_dict(
      self,
      outputs: List[np.ndarray],
      confidence_threshold: float,
      max_boxes: int,
  ) -> Dict[str, Any]:
    """Reformats and filters the raw outputs from the Triton server.

    Args:
      outputs: A list of numpy arrays containing the raw outputs from the Triton
        model. Expected to contain [boxes, probabilities, masks (optional)].
      confidence_threshold: Boxes with a confidence score below this threshold
        will be filtered out.
      max_boxes: The maximum number of top-scoring boxes to consider before
        applying the confidence threshold.

    Returns:
      A dict containing arrays of detection results for keys 'confidence',
      'labels', 'xyxy', and 'masks'. Bounding boxes are in [xmin, ymin,
      xmax, ymax] format. Masks are `None` if not present in model output.
      For example:

      {
          'confidence': np.array([0.9, 0.8]),
          'labels': np.array([1, 2]),
          'xyxy': np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
          'masks': np.array([mask1_data, mask2_data]) or None,
      }
    """
    raw_boxes = outputs[0].squeeze()
    raw_probs = _sigmoid(outputs[1])
    masks = outputs[2].squeeze() if len(outputs) == 3 else None

    scores = np.max(raw_probs, axis=2).squeeze()
    labels = np.argmax(raw_probs, axis=2).squeeze()

    # Filter by top-k bounding boxes
    sorted_idx = np.argsort(scores)[::-1][:max_boxes]

    # Filter by confidence score
    confidence_mask_filter = scores[sorted_idx] > confidence_threshold
    final_idx = sorted_idx[confidence_mask_filter]

    return {
        'confidence': scores[final_idx],
        'labels': labels[final_idx],
        'xyxy': _box_cxcywh_to_xyxyn(raw_boxes[final_idx]),
        'masks': masks[final_idx] if masks is not None else None,
    }

  def predict(
      self,
      image_path: str,
      confidence_threshold: float = 0.5,
      max_boxes: int = 100,
      output_dims: tuple[int, int] = (1024, 1024),
  ) -> Dict[str, Any]:
    """Performs inference on a single image using the Triton server.

    Args:
      image_path: The path to the input image file.
      confidence_threshold: Boxes with a confidence score below this threshold
        will be filtered out.
      max_boxes: The maximum number of top-scoring boxes to consider before
        applying the confidence threshold.
      output_dims: The dimensions (width, height) to which bounding boxes and
        masks should be scaled in the output.

    Returns:
      A dictionary containing the inference results:
        - 'confidence': A numpy array of confidence scores.
        - 'labels': A numpy array of predicted class labels (integer IDs).
        - 'xyxy': A numpy array of bounding boxes in [xmin, ymin, xmax, ymax]
          format, scaled to `output_dims`.
        - 'masks': A numpy array of boolean masks, rescaled to `output_dims`,
          or None if masks are not part of the model output.
    """

    # Preprocessing
    input_data = self._get_input_batch_for_inference(image_path)

    # Prepare Triton Input
    infer_input = httpclient.InferInput(
        'input', input_data.shape, datatype='FP32'
    )
    infer_input.set_data_from_numpy(input_data, binary_data=True)

    # Execute Inference
    response = self.client.infer(
        model_name=self.model_name, inputs=[infer_input]
    )

    # Extract results based on known output names
    raw_outputs = [
        response.as_numpy('dets'),
        response.as_numpy('labels'),
        response.as_numpy('4245'),
    ]

    # Reformat Triton output
    results = self._reformat_triton_output_to_dict(
        raw_outputs, confidence_threshold, max_boxes
    )

    # Scale to output dimensions
    results = self._scale_bbox_and_masks(results, output_dims)

    return results

  def _get_class_id_to_class_name_mapping(self):
    """Returns a mapping from class ID to class name."""
    labels_path = os.path.join(os.getcwd(), 'labels50.csv')
    labels_df = pd.read_csv(labels_path)
    class_id_to_class_name_mapper = labels_df.set_index('id').to_dict()['names']
    return class_id_to_class_name_mapper

  def get_class_names(self, results):
    """Returns the class names for the given results."""
    class_name_mapper = self._get_class_id_to_class_name_mapping()
    label_names = np.array(
        [class_name_mapper.get(c + 1, 'None') for c in results['labels']]
    )
    return label_names
