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

"""DINOv3-based grade classifier: model definition and inference class.

Checkpoints must be produced by our training scripts
(``train_classifier_repurpose_finetune_2.py`` or
``train_classifier_repurpose_linear_probe.py``). The pooling strategy
('cls' or 'cls_mean_patch') is auto-detected from the saved head's
input dimension; no manual override is required.

Image preprocessing matches the training-time validation transform:
``Resize((image_size, image_size))`` + ``ToTensor`` + ImageNet normalization.

Class names are NOT stored in our checkpoints — the caller must pass them in.
"""

from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as torch_functional
from torchvision import transforms

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
DEFAULT_INFERENCE_IMAGE_SIZE = 256

POOLING_CLS = "cls"
POOLING_CLS_MEAN_PATCH = "cls_mean_patch"
SUPPORTED_POOLING_STRATEGIES = (POOLING_CLS, POOLING_CLS_MEAN_PATCH)


def _load_backbone(dinov3_repo_dir: str, model_name: str) -> nn.Module:
  """Loads the DINOv3 backbone from a local repository.

  Args:
      dinov3_repo_dir: Path to the cloned DINOv3 repository.
      model_name: Name of the DINOv3 backbone variant to load.

  Returns:
      The loaded DINOv3 backbone model.
  """
  return torch.hub.load(
      dinov3_repo_dir,
      model_name,
      source="local",
      pretrained=False,
  )


class Dinov3Classification(nn.Module):
  """DINOv3 backbone with a linear classification head.

  Mirrors the training-side class in ``models.py``. The pooling
  strategy controls the feature vector fed to the head:

    - 'cls':            CLS token only; head input = backbone hidden size.
    - 'cls_mean_patch': CLS + mean patch tokens; head input = 2x hidden size.
  """

  def __init__(
      self,
      dinov3_repo_dir: str,
      model_name: str,
      num_classes: int,
      pooling: str = POOLING_CLS,
  ):
    """Initializes the Dinov3Classification model.

    Args:
        dinov3_repo_dir: Path to the cloned DINOv3 repository.
        model_name: Name of the DINOv3 backbone variant to load.
        num_classes: The number of output classes for the classification head.
        pooling: The pooling strategy to use ('cls' or 'cls_mean_patch').
          Defaults to 'cls'.

    Raises:
        ValueError: If an unsupported pooling strategy is provided.
    """
    super().__init__()

    if pooling not in SUPPORTED_POOLING_STRATEGIES:
      raise ValueError(
          f"Unsupported pooling strategy: {pooling!r}. "
          f"Expected one of {SUPPORTED_POOLING_STRATEGIES}."
      )

    self.pooling = pooling
    self.backbone_model = _load_backbone(
        dinov3_repo_dir=dinov3_repo_dir,
        model_name=model_name,
    )

    backbone_hidden_size = self.backbone_model.norm.normalized_shape[0]
    head_input_features = (
        backbone_hidden_size
        if pooling == POOLING_CLS
        else 2 * backbone_hidden_size
    )
    self.head = nn.Linear(
        in_features=head_input_features,
        out_features=num_classes,
        bias=True,
    )

  def extract_features(self, x: torch.Tensor) -> torch.Tensor:
    if self.pooling == POOLING_CLS:
      return self.backbone_model(x)

    token_features = self.backbone_model.forward_features(x)
    cls_token = token_features["x_norm_clstoken"]
    patch_tokens = token_features["x_norm_patchtokens"]
    return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.extract_features(x))


class DinoClassifier:
  """Loads a trained DINOv3 classifier and runs batched inference.

  Usage::

      classifier = GradeClassifier(
          classifier_checkpoint_path="model.pth",
          dinov3_repo_dir="/path/to/dinov3",
          model_name="dinov3_vitl16",
          class_names=["grade_a", "grade_b", "grade_c"],
          device=torch.device("cuda"),
      )
      results = classifier.predict(pil_images)
  """

  def __init__(
      self,
      classifier_checkpoint_path: str,
      dinov3_repo_dir: str,
      model_name: str,
      class_names: Sequence[str],
      device: str | None = None,
      image_size: int = DEFAULT_INFERENCE_IMAGE_SIZE,
  ):
    """Loads the checkpoint and prepares the model for inference.

    Args:
        classifier_checkpoint_path: Path to a checkpoint produced by our
          training scripts. Must contain ``model_state_dict``.
        dinov3_repo_dir: Path to the cloned Facebook DINOv3 repository.
        model_name: DINOv3 backbone variant (e.g. ``'dinov3_vitl16'``). Must
          match the backbone the checkpoint was trained with.
        class_names: Class names in label-index order. Not stored in
          checkpoints, so the caller must supply them.
        device: Torch device on which the model will run.
        image_size: Square input size for the eval transform. Must be a multiple
          of the backbone's patch size (16 for vit*16).

    Raises:
        KeyError: If the checkpoint is missing ``model_state_dict`` or
            ``head.weight``.
        ValueError: If the saved head dimension cannot be matched to a
            known pooling strategy for this backbone.
    """
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.class_names = list(class_names)

    saved_state_dict = self._read_state_dict(
        classifier_checkpoint_path,
        self.device,
    )
    self.pooling = self._detect_pooling(
        saved_state_dict,
        dinov3_repo_dir,
        model_name,
    )
    print(f"[INFO]: Detected pooling strategy from checkpoint: {self.pooling}")

    self.model = Dinov3Classification(
        dinov3_repo_dir=dinov3_repo_dir,
        model_name=model_name,
        num_classes=len(class_names),
        pooling=self.pooling,
    ).to(self.device)
    self.model.load_state_dict(saved_state_dict)
    self.model.eval()

    self._eval_transform = self._build_eval_transform(image_size)

  # ------------------------------------------------------------------
  # Inference
  # ------------------------------------------------------------------

  @torch.no_grad()
  def predict(self, pil_images: Sequence[Image.Image]) -> list[dict[str, Any]]:
    """Classifies a list of PIL images in a single forward pass.

    Args:
        pil_images: Sequence of RGB PIL images (length >= 1).

    Returns:
        One prediction dict per input image, in the same order::

            {
                "predicted_class": str,
                "predicted_probability": float,  # 0-100
                "all_probabilities": dict[str, float],
            }

    Raises:
        ValueError: If ``pil_images`` is empty.
    """
    if not pil_images:
      raise ValueError("predict requires at least one image.")

    batch = torch.stack([self._eval_transform(img) for img in pil_images]).to(
        self.device
    )

    logits = self.model(batch)
    probabilities = torch_functional.softmax(logits, dim=1).cpu().numpy()

    predictions = []
    for probability_row in probabilities:
      predicted_index = int(np.argmax(probability_row))
      predictions.append({
          "predicted_class": self.class_names[predicted_index],
          "predicted_probability": float(
              probability_row[predicted_index] * 100.0
          ),
          "all_probabilities": {
              name: float(p * 100.0)
              for name, p in zip(self.class_names, probability_row)
          },
      })
    return predictions

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _read_state_dict(
      checkpoint_path: str, device: torch.device
  ) -> dict[str, Any]:
    """Reads and validates the model state dict from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device to map the loaded parameters to.

    Returns:
        The state dict containing model weights.

    Raises:
        KeyError: If 'model_state_dict' or 'head.weight' is missing.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
      raise KeyError(
          f"Checkpoint at '{checkpoint_path}' is missing "
          "'model_state_dict'. Was it produced by our training scripts?"
      )
    state_dict = checkpoint["model_state_dict"]
    if "head.weight" not in state_dict:
      raise KeyError(
          "Checkpoint state dict is missing 'head.weight'; cannot infer "
          "pooling. Was the checkpoint produced by our Dinov3Classification?"
      )
    return state_dict

  @staticmethod
  def _detect_pooling(
      state_dict: dict[str, Any],
      dinov3_repo_dir: str,
      model_name: str,
  ) -> str:
    """Infers the pooling strategy from the shape of the saved head's weights.

    Args:
        state_dict: The model state dict loaded from checkpoint.
        dinov3_repo_dir: Path to the cloned DINOv3 repository.
        model_name: Name of the DINOv3 backbone variant.

    Returns:
        The inferred pooling strategy ('cls' or 'cls_mean_patch').

    Raises:
        ValueError: If the pooling strategy cannot be inferred from the head
            dimensions.
    """
    # Load a throwaway backbone only to read its hidden size, then discard it.
    probe = _load_backbone(
        dinov3_repo_dir=dinov3_repo_dir, model_name=model_name
    )
    backbone_hidden_size = probe.norm.normalized_shape[0]
    del probe

    head_input_features = state_dict["head.weight"].shape[1]
    if head_input_features == backbone_hidden_size:
      return POOLING_CLS
    if head_input_features == 2 * backbone_hidden_size:
      return POOLING_CLS_MEAN_PATCH

    raise ValueError(
        "Cannot infer pooling strategy: head input dim "
        f"{head_input_features} matches neither {backbone_hidden_size} "
        f"(cls) nor {2 * backbone_hidden_size} (cls_mean_patch). "
        "This usually means the checkpoint was trained with a different "
        "backbone than the one configured here."
    )

  @staticmethod
  def _build_eval_transform(image_size: int) -> transforms.Compose:
    """Builds the transformation pipeline for evaluation image preprocessing.

    Args:
        image_size: Target square image size.

    Returns:
        A torchvision transforms Compose object.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
