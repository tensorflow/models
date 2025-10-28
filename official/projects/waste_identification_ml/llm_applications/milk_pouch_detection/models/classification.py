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

"""ViT-based image classifier client for categorizing images."""

import pathlib
from typing import Sequence
import warnings

from PIL import Image
import torch
import torchvision

# Suppress common warnings for a cleaner console output.
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

FEATURE_DIM = 768  # ViT-B/16 embedding size


# TODO: b/455871640 - Add unit tests for this class.


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
