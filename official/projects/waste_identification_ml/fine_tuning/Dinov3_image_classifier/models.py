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

"""Builds a linear classifier on top of a DINOv3 backbone."""

import logging
import pathlib
from typing import Self

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)

# Pooling strategy names recognized by `Dinov3Classification`. Exposed as
# module constants so training scripts can validate CLI input against the
# same set the model accepts.
POOLING_CLS = "cls"
POOLING_CLS_MEAN_PATCH = "cls_mean_patch"
SUPPORTED_POOLING_STRATEGIES = (POOLING_CLS, POOLING_CLS_MEAN_PATCH)

# Default number of output classes for classification models.
DEFAULT_NUMBER_OF_CLASSES = 2


def load_model(
    model_name: str,
    repo_dir: pathlib.Path,
    weights: pathlib.Path | None = None,
) -> nn.Module:
  """Loads a DINOv3 backbone via torch.hub from a local repository.

  Args:
    model_name: Name of the DINOv3 model variant (e.g., 'dinov3_vits16').
    repo_dir: Path to the cloned Facebook DINOv3 repository.
    weights: Optional path to a pretrained weights file. If None, the model is
      loaded with random weights.

  Returns:
    The DINOv3 backbone model.

  Raises:
    ValueError: If `model_name` or `repo_dir` is falsy (empty or None).
  """
  if not model_name:
    raise ValueError("model_name must be a non-empty string.")
  if str(repo_dir) in ("", "."):
    raise ValueError("repo_dir must be a non-empty path.")

  if weights is not None:
    _LOGGER.info("Loading pretrained backbone weights from: %s", weights)
    return torch.hub.load(
        str(repo_dir),
        model_name,
        source="local",
        weights=str(weights),
    )

  _LOGGER.info("No pretrained weights path given. Loading with random weights.")
  return torch.hub.load(str(repo_dir), model_name, source="local")


class Dinov3Classification(nn.Module):
  """DINOv3 backbone with a linear classification head.

  The feature vector fed to the classification head is controlled by the
  `pooling` argument:

    - 'cls':            Use only the final CLS token. Head input dimension
                        equals the backbone hidden size.
    - 'cls_mean_patch': Concatenate the final CLS token with the mean of
                        the final patch tokens. Head input dimension equals
                        twice the backbone hidden size.

  Attributes:
    backbone_model: The DINOv3 feature extractor. Note that when `fine_tune` is
      False, its parameters (`requires_grad`) are frozen in-place during
      initialization.
    head: A linear layer mapping pooled features to class logits.
    pooling: The pooling strategy used for feature extraction.
  """

  def __init__(
      self,
      backbone_model: nn.Module,
      number_of_classes: int = DEFAULT_NUMBER_OF_CLASSES,
      pooling: str = POOLING_CLS,
      fine_tune: bool = False,
  ):
    """Initializes the DINOv3 classifier with an already-loaded backbone.

    Args:
      backbone_model: The pre-loaded DINOv3 backbone module (`nn.Module`). Note
        that if `fine_tune=False`, the parameters of `backbone_model` are
        modified in-place (`requires_grad=False`).
      number_of_classes: Number of output classes.
      pooling: Feature extraction strategy. One of
        `SUPPORTED_POOLING_STRATEGIES`.
      fine_tune: If True, backbone parameters remain trainable. If False, they
        are frozen.

    Raises:
      ValueError: If `pooling` is not a supported strategy.
    """
    super().__init__()

    if pooling not in SUPPORTED_POOLING_STRATEGIES:
      raise ValueError(
          f"Unsupported pooling strategy: {pooling!r}. "
          f"Expected one of {SUPPORTED_POOLING_STRATEGIES}."
      )

    self.pooling = pooling
    self.backbone_model = backbone_model

    backbone_hidden_size = self.backbone_model.norm.normalized_shape[0]
    if pooling == POOLING_CLS:
      head_input_features = backbone_hidden_size
    else:
      # 'cls_mean_patch' concatenates two vectors of size
      # backbone_hidden_size.
      head_input_features = 2 * backbone_hidden_size

    self.head = nn.Linear(
        in_features=head_input_features,
        out_features=number_of_classes,
        bias=True,
    )

    if not fine_tune:
      for parameter in self.backbone_model.parameters():
        parameter.requires_grad = False

  @classmethod
  def from_model_name(
      cls,
      model_name: str,
      repo_dir: pathlib.Path,
      number_of_classes: int = DEFAULT_NUMBER_OF_CLASSES,
      weights: pathlib.Path | None = None,
      pooling: str = POOLING_CLS,
      fine_tune: bool = False,
  ) -> Self:
    """Factory method that loads a DINOv3 backbone and constructs the classifier.

    Args:
      model_name: Name of the DINOv3 model variant.
      repo_dir: Path to the cloned Facebook DINOv3 repository.
      number_of_classes: Number of output classes.
      weights: Optional path to a pretrained backbone weights file.
      pooling: Feature extraction strategy. One of
        `SUPPORTED_POOLING_STRATEGIES`.
      fine_tune: If True, backbone parameters remain trainable. If False, they
        are frozen.

    Returns:
      A new `Dinov3Classification` instance.
    """
    backbone_model = load_model(
        model_name=model_name, repo_dir=repo_dir, weights=weights
    )
    return cls(
        backbone_model=backbone_model,
        number_of_classes=number_of_classes,
        pooling=pooling,
        fine_tune=fine_tune,
    )

  def extract_features(self, image_batch: torch.Tensor) -> torch.Tensor:
    """Computes the pooled feature vector fed to the head.

    Args:
      image_batch: Input image tensor of shape (batch, channels, height, width).

    Returns:
      Pooled feature tensor of shape (batch, head_input_features), where
      head_input_features depends on the configured pooling strategy.
    """
    if self.pooling == POOLING_CLS:
      return self.backbone_model(image_batch)

    token_features = self.backbone_model.forward_features(image_batch)
    cls_token = token_features["x_norm_clstoken"]
    patch_tokens = token_features["x_norm_patchtokens"]
    mean_patch_token = patch_tokens.mean(dim=1)
    return torch.cat([cls_token, mean_patch_token], dim=1)

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    """Runs a forward pass through the backbone and classification head.

    Args:
      image_batch: Input image tensor of shape (batch, channels, height, width).

    Returns:
      Class logits of shape (batch, number_of_classes).
    """
    features = self.extract_features(image_batch)
    return self.head(features)
