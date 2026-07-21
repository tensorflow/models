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

"""DINOv3 classification module for batched image inference.

This module exposes:
  - ClassifierError: Exception raised for classifier construction or inference
    issues.
  - PoolingStrategy: Enum describing how backbone token features are pooled
    before the classification head.
  - Prediction: Typed result returned per input image by the classifier.
  - Dinov3ClassificationModule: The underlying nn.Module wrapping a DINOv3
    backbone and a linear head.
  - DINOv3Classifier: High-level batched classifier that accepts an
    already-loaded model; construct it via `DINOv3Classifier.from_config` for
    the common case of loading from a checkpoint on disk.
"""

from collections.abc import Mapping, Sequence
import enum
import logging
import pathlib
from typing import Self, TypedDict

import numpy
from PIL import Image
import torch
from torch import nn
from torch.nn import functional
from torchvision.transforms import v2

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader

_LOGGER = logging.getLogger(__name__)


class PoolingStrategy(enum.Enum):
  """Strategy for pooling backbone token features into a single vector.

  Attributes:
    CLS: Use only the CLS token embedding.
    CLS_MEAN_PATCH: Concatenate the CLS token with the mean of patch tokens.
  """

  CLS = "cls"
  CLS_MEAN_PATCH = "cls_mean_patch"


class ClassifierError(Exception):
  """Exception raised for classifier loading, construction, or shape issues."""


class Prediction(TypedDict):
  """Typed prediction result for a single image.

  Callers index by string key exactly as with a plain dict; the TypedDict adds
  static-analysis support without changing runtime behavior.

  Attributes:
    predicted_class: Name of the top-1 class.
    predicted_probability_percent: Top-1 probability expressed as a percentage
      in the range [0.0, 100.0].
    all_probabilities_percent: Mapping from class name to that class's
      probability as a percentage in the range [0.0, 100.0].
  """

  predicted_class: str
  predicted_probability_percent: float
  all_probabilities_percent: dict[str, float]


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


def _infer_pooling_from_state_dict(
    saved_state_dict: Mapping[str, torch.Tensor], hidden_size: int
) -> PoolingStrategy:
  """Infers the pooling strategy from the shape of the saved head weights.

  Args:
    saved_state_dict: The checkpoint's model state dict.
    hidden_size: Backbone hidden dimensionality.

  Returns:
    The inferred pooling strategy.

  Raises:
    ClassifierError: If the head input dimension matches neither the CLS nor
      the CLS_MEAN_PATCH expectation.
  """
  head_input_features = saved_state_dict["head.weight"].shape[1]
  if head_input_features == hidden_size:
    return PoolingStrategy.CLS
  if head_input_features == 2 * hidden_size:
    return PoolingStrategy.CLS_MEAN_PATCH
  raise ClassifierError(
      "Cannot infer pooling strategy. Head input dimension "
      f"{head_input_features} does not match hidden size {hidden_size} or "
      f"{2 * hidden_size}."
  )


def _load_checkpoint_state_dict(
    checkpoint_path: pathlib.Path | str, device: torch.device
) -> dict[str, torch.Tensor]:
  """Loads the model state dict from a checkpoint file.

  Args:
    checkpoint_path: Filesystem path to the checkpoint.
    device: Target device for `map_location`.

  Returns:
    The `model_state_dict` mapping from parameter name to tensor.

  Raises:
    ClassifierError: If the checkpoint is missing required keys.
  """
  checkpoint = torch.load(
      checkpoint_path, map_location=device, weights_only=True
  )
  if "model_state_dict" not in checkpoint:
    raise ClassifierError(
        "Checkpoint is missing required key 'model_state_dict'."
    )
  saved_state_dict = checkpoint["model_state_dict"]
  if "head.weight" not in saved_state_dict:
    raise ClassifierError(
        "Checkpoint state dict is missing required key 'head.weight'."
    )
  return saved_state_dict


def _build_image_transform(config: config_loader.DINOv3Config) -> v2.Compose:
  """Builds the preprocessing pipeline used to feed PIL images to the model.

  Args:
    config: DINOv3 model configuration providing image size and normalization
      statistics.

  Returns:
    A torchvision v2.Compose pipeline.
  """
  return v2.Compose([
      v2.ToImage(),
      v2.Resize(
          (config.inference_image_size, config.inference_image_size),
          antialias=True,
      ),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=config.image_mean, std=config.image_std),
  ])


class Dinov3ClassificationModule(nn.Module):
  """DINOv3 backbone paired with a linear classification head.

  Attributes:
    pooling: The pooling strategy applied to backbone features.
    backbone_model: The DINOv3 backbone.
    head: Linear layer mapping pooled features to per-class logits.
  """

  def __init__(
      self,
      backbone_model: nn.Module,
      hidden_size: int,
      number_of_classes: int,
      pooling: PoolingStrategy,
  ):
    """Initializes the module from an already-loaded backbone.

    Args:
      backbone_model: The DINOv3 backbone module. Injected so that this class
        does not perform filesystem I/O in its constructor.
      hidden_size: Dimensionality of the backbone's token embeddings.
      number_of_classes: Number of output classes for the head.
      pooling: Pooling strategy that determines the head's input dimension.
    """
    super().__init__()
    self.pooling = pooling
    self.backbone_model = backbone_model

    if pooling is PoolingStrategy.CLS:
      head_input_features = hidden_size
    else:
      head_input_features = 2 * hidden_size

    self.head = nn.Linear(
        in_features=head_input_features,
        out_features=number_of_classes,
        bias=True,
    )

  def extract_features(self, image_batch: torch.Tensor) -> torch.Tensor:
    """Extracts pooled features from a batch of images.

    Args:
      image_batch: Tensor of shape (batch_size, 3, height, width).

    Returns:
      Tensor of pooled features. Shape is (batch_size, hidden_size) for the
      CLS strategy and (batch_size, 2 * hidden_size) for CLS_MEAN_PATCH.
    """
    if self.pooling is PoolingStrategy.CLS:
      return self.backbone_model(image_batch)

    token_features = self.backbone_model.forward_features(image_batch)
    cls_token = token_features["x_norm_clstoken"]
    mean_patch_token = token_features["x_norm_patchtokens"].mean(dim=1)
    return torch.cat([cls_token, mean_patch_token], dim=1)

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    """Runs the full backbone + head forward pass.

    Args:
      image_batch: Tensor of shape (batch_size, 3, height, width).

    Returns:
      Per-class logits of shape (batch_size, number_of_classes).
    """
    return self.head(self.extract_features(image_batch))


class DINOv3Classifier:
  """High-level batched classifier backed by a DINOv3 model.

  The order of `class_names` is load-bearing: `class_names[i]` must correspond
  to logit index `i` in the trained model. Reordering this list will silently
  produce incorrect predictions.

  Construct instances via `DINOv3Classifier.from_config` for the common case of
  loading from a checkpoint. The `__init__` constructor accepts an
  already-built model and is intended for callers that manage model loading
  themselves (for example, unit tests).

  Attributes:
    class_names: Ordered list of class names indexed by logit position.
  """

  def __init__(
      self,
      model: Dinov3ClassificationModule,
      class_names: Sequence[str],
      image_transform: v2.Compose,
      device: torch.device,
  ):
    """Initializes the classifier from injected dependencies.

    Args:
      model: The already-loaded classification module, expected to be on
        `device` and in eval mode. This constructor does not move the model or
        change its mode.
      class_names: Ordered list of class names indexed by logit position.
      image_transform: Preprocessing pipeline applied to each PIL image before
        stacking into a batch.
      device: Target device on which the model resides and to which input
        batches will be moved.
    """
    self.class_names = list(class_names)
    self._model = model
    self._image_transform = image_transform
    self._device = device

  @classmethod
  def from_config(
      cls,
      config: config_loader.DINOv3Config,
      class_names: Sequence[str],
      device: str,
  ) -> Self:
    """Builds a classifier by loading a checkpoint from disk.

    Args:
      config: DINOv3 model configuration.
      class_names: Ordered list of class names indexed by logit position.
      device: Requested device string (e.g., 'cuda', 'cpu', 'mps').

    Returns:
      A ready-to-use DINOv3Classifier with its model on the resolved device
      and in eval mode.

    Raises:
      ClassifierError: If the checkpoint is missing required keys, if the
        pooling strategy cannot be inferred, or if the state dict fails to
        load into the constructed model.
    """
    resolved_device = _resolve_device(device)
    saved_state_dict = _load_checkpoint_state_dict(
        checkpoint_path=config.checkpoint_path, device=resolved_device
    )

    backbone_model = torch.hub.load(
        config.repo_dir,
        config.model_name,
        source="local",
        pretrained=False,
    )
    # DINOv3 Vision Transformers store token embedding dimension in
    # norm.normalized_shape[0].
    hidden_size = backbone_model.norm.normalized_shape[0]

    pooling = _infer_pooling_from_state_dict(
        saved_state_dict=saved_state_dict, hidden_size=hidden_size
    )

    model = Dinov3ClassificationModule(
        backbone_model=backbone_model,
        hidden_size=hidden_size,
        number_of_classes=len(class_names),
        pooling=pooling,
    ).to(resolved_device)

    try:
      model.load_state_dict(saved_state_dict)
    except RuntimeError as error:
      raise ClassifierError(
          f"Failed to load state dict into model: {error}"
      ) from error

    model.eval()
    return cls(
        model=model,
        class_names=class_names,
        image_transform=_build_image_transform(config),
        device=resolved_device,
    )

  @torch.no_grad()
  def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]:
    """Classifies a batch of PIL images in a single forward pass.

    Args:
      images: Sequence of PIL images to classify.

    Returns:
      List of Prediction dicts in the same order as `images`. Each Prediction
      contains 'predicted_class', 'predicted_probability_percent', and
      'all_probabilities_percent'. Returns an empty list when `images` is
      empty.
    """
    if not images:
      return []

    image_tensors = [self._image_transform(image) for image in images]
    batch_tensor = torch.stack(image_tensors).to(self._device)

    logits = self._model(batch_tensor)
    probabilities = functional.softmax(logits, dim=1).cpu().numpy()

    predictions: list[Prediction] = []
    for probability_row in probabilities:
      predicted_index = int(numpy.argmax(probability_row))
      all_probabilities_percent: dict[str, float] = {}
      for class_name, probability in zip(self.class_names, probability_row):
        all_probabilities_percent[class_name] = float(probability * 100.0)
      prediction: Prediction = {
          "predicted_class": self.class_names[predicted_index],
          "predicted_probability_percent": float(
              probability_row[predicted_index] * 100.0
          ),
          "all_probabilities_percent": all_probabilities_percent,
      }
      predictions.append(prediction)
    return predictions
