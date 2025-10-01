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

"""Utility functions for inference with fine-tuned models."""

from collections.abc import Sequence
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
import torchvision

FEATURE_DIM = 768   # ViT-B/16 embedding size


def load_vit_classifier(
    model_path: pathlib.Path, num_classes: int, device: torch.device
) -> torch.nn.Module:
  """Loads a fine-tuned ViT-B-16 model for inference.

  Args:
    model_path: Path to the saved model state_dict.
    num_classes: Number of output classes for the model head.
    device: The device to load the model on (e.g., 'cpu' or 'cuda').

  Returns:
    A PyTorch model in evaluation mode.
  """
  print(f"Loading model to {device}")

  # Load base architecture.
  model = torchvision.models.vit_b_16(weights=None)

  # Freeze params.
  for parameter in model.parameters():
    parameter.requires_grad = False

  # Set custom head.
  model.heads = torch.nn.Linear(
      in_features=FEATURE_DIM,
      out_features=num_classes
  )

  # Load the state_dict.
  model.load_state_dict(torch.load(model_path, map_location=device))

  # Set to device and eval mode.
  model.to(device)
  model.eval()

  return model


def get_default_transform(
    image_size: tuple[int, int] = (224, 224)
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
      # These are the standard mean and std values for ImageNet pre-trained
      # models.
      torchvision.transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])


def process_image(
    image_path: pathlib.Path, transform: torchvision.transforms.Compose
) -> torch.Tensor:
  """Loads an image, applies transforms, and adds a batch dimension.

  Args:
    image_path: Path to the input image file.
    transform: A torchvision Compose object for image transformation.

  Returns:
    A transformed image tensor with a batch dimension of 1.
  """
  img = Image.open(image_path)

  # Transform and add an extra dimension (batch_size = 1).
  return transform(img).unsqueeze(dim=0)


def predict(
    model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
  """Performs inference on a single image tensor.

  Args:
    model: The PyTorch model to use for inference.
    image_tensor: The input image tensor (with batch dimension).
    device: The device the model and tensor are on.

  Returns:
    The raw logits output from the model.
  """
  # Move tensor to the same device as the model.
  image_tensor = image_tensor.to(device)

  # Turn on inference mode.
  with torch.inference_mode():
    return model(image_tensor)


def get_prediction_details(
    logits: torch.Tensor, class_names: Sequence[str]
) -> tuple[str, float]:
  """Converts raw logits to a predicted class and its probability.

  Args:
    logits: The raw logits output from the model.
    class_names: A list of class names corresponding to the model's output
      indices.

  Returns:
    A tuple containing the predicted class name and its probability.
  """
  probs = torch.softmax(logits, dim=1)

  # Get the top probability and index.
  pred_prob, pred_idx = torch.max(probs, dim=1)

  # Get the class name and probability value.
  pred_class = class_names[pred_idx.item()]
  pred_prob_value = pred_prob.item()

  return (pred_class, pred_prob_value)


def plot_prediction(
    image_path: pathlib.Path, pred_class: str, pred_prob: float
):
  """Plots the original image with its prediction and probability.

  Args:
    image_path: Path to the input image file.
    pred_class: The predicted class name for the image.
    pred_prob: The predicted probability for the class.
  """
  img = Image.open(image_path)
  plt.figure()
  plt.imshow(img)
  plt.title(f"Pred: {pred_class} | Prob: {pred_prob:.3f}%")
  plt.axis(False)
  plt.show()


def show_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: Sequence[str]
) -> None:
  """Displays a confusion matrix heatmap with counts and row-normalized percentages.

  Args:
    confusion_matrix: A 2D NumPy array representing the confusion matrix.
    class_names: A list of class names corresponding to matrix indices.
  """
  matrix = confusion_matrix.copy()
  cell_counts = matrix.flatten()

  cm_row_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]

  row_percentages = [f"{value:.2f}" for value in cm_row_norm.flatten()]
  cell_labels = [
      f"{count}\n{percentage}"
      for count, percentage in zip(cell_counts, row_percentages)
  ]
  cell_labels = np.asarray(cell_labels).reshape(
      matrix.shape[0], matrix.shape[1]
  )

  df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)

  # Plot heatmap
  heatmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
  heatmap.yaxis.set_ticklabels(
      heatmap.yaxis.get_ticklabels(), rotation=0, ha="right"
  )
  heatmap.xaxis.set_ticklabels(
      heatmap.xaxis.get_ticklabels(), rotation=30, ha="right"
  )

  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.tight_layout()
  plt.show()
