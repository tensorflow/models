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

"""Vision and LLM models for milk pouch detection."""

import math
import pathlib
import subprocess
from typing import Any, Optional, Sequence
import warnings

from groundingdino.util import inference
import numpy as np
import ollama
from PIL import Image
from sam2 import build_sam
from sam2 import sam2_image_predictor
import torch
import torchvision

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection import models_utils


# Suppress common warnings for a cleaner console output.
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

FEATURE_DIM = 768  # ViT-B/16 embedding size


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
        - xyxy_boxes: Detected bounding boxes in [x1, y1, x2, y2] format.
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

    xyxy_boxes = models_utils.convert_boxes_cxcywh_to_xyxy(boxes, image.shape)
    return image, xyxy_boxes, scores, labels

  def _segment_objects(
      self, image_source: np.ndarray, xyxy_boxes: np.ndarray
  ) -> tuple[list[np.ndarray], list[torch.Tensor], list[np.ndarray]]:
    """Generates segmentation masks for given bounding boxes using SAM2.

    Args:
      image_source: The source image as a NumPy array.
      xyxy_boxes: A NumPy array of bounding boxes in [x1, y1, x2, y2] format.

    Returns:
      A tuple containing:
        - all_masks: A list of boolean segmentation masks.
        - all_scores: A list of confidence scores for each mask.
        - all_boxes: A list of the original bounding boxes.
    """
    self.sam_predictor.set_image(image_source)

    all_masks, all_scores, all_boxes = [], [], []
    for bbox in xyxy_boxes:
      box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
      if box_area < 0.25 * math.prod(image_source.shape[:2]):
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],  # SAM expects a batch dimension.
            multimask_output=False,
        )
        # Squeeze to remove batch and multi-mask dimensions.
        all_masks.append(masks.squeeze())
        all_scores.append(scores)
        all_boxes.append(bbox)

    return all_masks, all_scores, all_boxes

  def detect_and_segment(
      self, image_path: str, text_prompt: str
  ) -> Optional[dict[str, Any]]:
    """Runs the full detection and segmentation pipeline on an image.

    Args:
      image_path: The file path to the input image.
      text_prompt: The text description of objects to detect and segment.

    Returns:
      A dictionary containing the processed data ('image', 'boxes', 'masks')
      or None if no objects were detected.
    """
    print(f"\nProcessing '{image_path}'")
    image, boxes, _, _ = self._detect_objects(image_path, text_prompt)

    if boxes.shape[0] == 0:
      print('No objects detected.')
      return None

    masks, _, mask_boxes = self._segment_objects(image, boxes)
    print('Segmentation complete.')

    return {
        'image': image,
        'boxes': mask_boxes,
        'masks': masks,
    }


class ImageClassifier:
  """ViT-based image classifier for categorizing images.

  This class loads a fine-tuned ViT-B/16 model and provides methods for
  image classification with automatic preprocessing.

  Attributes:
    model: The loaded ViT classifier model.
    device: The PyTorch device the model runs on.
    transform: The image preprocessing pipeline.
    class_names: List of class names for predictions.
  """

  def __init__(
      self,
      model_path: str,
      class_names: Sequence[str],
      device: str = 'cuda',
      image_size: tuple[int, int] = (224, 224),
  ) -> None:
    """Initializes the image classifier.

    Args:
      model_path: Path to the saved model state_dict.
      class_names: List of class names corresponding to model output indices.
      device: The hardware device to run the model on (e.g., "cuda", "cpu").
      image_size: Target size (height, width) for resizing images.
    """
    self.device = torch.device(device)
    self.class_names = class_names
    self.transform = self._get_default_transform(image_size)

    print('Loading ViT image classifier...')
    self.model = self._load_vit_classifier(
        pathlib.Path(model_path), len(class_names)
    )
    print('✅ ViT classifier loaded.')

  def _load_vit_classifier(
      self, model_path: pathlib.Path, num_classes: int
  ) -> torch.nn.Module:
    """Loads a fine-tuned ViT-B-16 model for inference.

    Args:
      model_path: Path to the saved model state_dict.
      num_classes: Number of output classes for the model head.

    Returns:
      A PyTorch model in evaluation mode.
    """
    print(f'Loading model to {self.device}')

    # Load base architecture.
    model = torchvision.models.vit_b_16(weights=None)

    # Freeze params.
    for parameter in model.parameters():
      parameter.requires_grad = False

    # Set custom head.
    model.heads = torch.nn.Linear(
        in_features=FEATURE_DIM, out_features=num_classes
    )

    # Load the state_dict.
    model.load_state_dict(torch.load(model_path, map_location=self.device))

    # Set to device and eval mode.
    model.to(self.device)
    model.eval()

    return model

  def _get_default_transform(
      self, image_size: tuple[int, int]
  ) -> torchvision.transforms.Compose:
    """Returns the default ImageNet transformation pipeline.

    Args:
      image_size: The target size (height, width) for resizing images.

    Returns:
      A torchvision Compose object representing the transformation pipeline.
    """
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        # Standard mean and std values for ImageNet pre-trained models.
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

  def _process_image(self, image_path: pathlib.Path) -> torch.Tensor:
    """Loads an image, applies transforms, and adds a batch dimension.

    Args:
      image_path: Path to the input image file.

    Returns:
      A transformed image tensor with a batch dimension of 1.
    """
    img = Image.open(image_path)
    # Transform and add an extra dimension (batch_size = 1).
    return self.transform(img).unsqueeze(dim=0)

  def _predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
    """Performs inference on a single image tensor.

    Args:
      image_tensor: The input image tensor (with batch dimension).

    Returns:
      The raw logits output from the model.
    """
    # Move tensor to the same device as the model.
    image_tensor = image_tensor.to(self.device)

    # Turn on inference mode.
    with torch.inference_mode():
      return self.model(image_tensor)

  def _get_prediction_details(self, logits: torch.Tensor) -> tuple[str, float]:
    """Converts raw logits to a predicted class and its probability.

    Args:
      logits: The raw logits output from the model.

    Returns:
      A tuple containing the predicted class name and its probability.
    """
    probs = torch.softmax(logits, dim=1)

    # Get the top probability and index.
    pred_prob, pred_idx = torch.max(probs, dim=1)

    # Get the class name and probability value.
    pred_class = self.class_names[pred_idx.item()]
    pred_prob_value = pred_prob.item()

    return (pred_class, pred_prob_value)

  def classify(self, image_path: str) -> tuple[str, float]:
    """Classifies an image and returns the predicted class and probability.

    Args:
      image_path: Path to the input image file.

    Returns:
      A tuple containing the predicted class name and its probability.
    """
    image_tensor = self._process_image(pathlib.Path(image_path))
    logits = self._predict(image_tensor)
    return self._get_prediction_details(logits)


class LlmModels:
  """Provides an interface to interact with a local LLM via Ollama."""

  def query_image_with_llm(
      self, image_path: str, prompt: str, model_name: str
  ) -> str:
    """Sends an image and a text prompt to a local Ollama LLM.

    Args:
      image_path: Path to the image file.
      prompt: The question or prompt for the LLM.
      model_name: The name of the Ollama model to use (e.g., 'llava').

    Returns:
      The text response from the LLM.
    """
    response: ollama.ChatResponse = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
        options={
            'temperature': 0.0,
        },
    )
    return response['message']['content']

  def stop_model(self, model_name: str) -> None:
    """Stops a running Ollama model to free up system resources.

    This function executes the 'ollama stop' command-line instruction.

    Args:
      model_name: The name of the Ollama model to stop.
    """
    print(f'Attempting to stop Ollama model: {model_name}...')
    try:
      result = subprocess.run(
          ['ollama', 'stop', model_name],
          capture_output=True,
          text=True,
          check=False,
      )
      if result.returncode == 0:
        print(f'✅ Successfully sent stop command for model: {model_name}')
      else:
        # This may not be an error if the model wasn't running.
        print(
            'Info: Could not stop model (may not be running):'
            f' {result.stderr.strip()}'
        )
    except FileNotFoundError:
      print(
          "⚠️ 'ollama' command not found. Is Ollama installed and in your PATH?"
      )
    except subprocess.CalledProcessError as e:
      print(f'⚠️ An unexpected error occurred: {e}')
