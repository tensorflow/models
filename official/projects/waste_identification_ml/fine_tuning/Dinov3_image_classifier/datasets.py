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

"""Datasets and data loaders for the DINOv3 image classifier."""

from collections.abc import Sequence
from typing import TypeAlias
import torch
from torch.utils import data as torch_data
from torchvision import datasets
from torchvision.transforms import v2

DatasetsTuple: TypeAlias = tuple[
    datasets.ImageFolder, datasets.ImageFolder, list[str]
]
DataLoadersTuple: TypeAlias = tuple[
    torch_data.DataLoader, torch_data.DataLoader
]


def _get_train_transform(
    image_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> v2.Compose:
  """Builds the training image transform pipeline.

  Uses the `torchvision.transforms.v2` API, matching the DINOv3
  reference preprocessing: resize the uint8 image, convert to float32
  scaled to [0, 1], then normalize with the given statistics.

  Note: Unlike typical supervised training, this fine-tuning setup intentionally
  omits strong data augmentations like `RandomResizedCrop` and
  `RandomHorizontalFlip`. This aligns with common practices when fine-tuning
  models pre-trained with self-supervised methods like DINOv3, where the
  pre-training itself involves extensive augmentations. The focus during
  fine-tuning is often on adapting the pre-trained features to the new
  dataset with minimal disruption from further aggressive augmentations.

  Args:
      image_size: Target side length in pixels for the square resize.
      image_mean: Per-channel mean for normalization, as a 3-tuple of floats in
        `(R, G, B)` order.
      image_std: Per-channel standard deviation for normalization, as a 3-tuple
        of floats in `(R, G, B)` order.

  Returns:
      A pipeline that resizes, converts to a float32 tensor in `[0, 1]`, and
      normalizes with the given statistics. No augmentation is applied.
  """
  return v2.Compose([
      v2.ToImage(),
      v2.Resize((image_size, image_size), antialias=True),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=image_mean, std=image_std),
  ])


def _get_valid_transform(
    image_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> v2.Compose:
  """Builds the validation image transform pipeline.

  Identical to the training transform: resize, float conversion, and
  normalization, with no augmentation.

  Args:
      image_size: Target side length in pixels for the square resize.
      image_mean: Per-channel mean for normalization, as a 3-tuple of floats in
        `(R, G, B)` order.
      image_std: Per-channel standard deviation for normalization, as a 3-tuple
        of floats in `(R, G, B)` order.

  Returns:
      A pipeline that resizes, converts to a float32 tensor in `[0, 1]`, and
      normalizes with the given statistics.
  """
  return v2.Compose([
      v2.ToImage(),
      v2.Resize((image_size, image_size), antialias=True),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=image_mean, std=image_std),
  ])


def get_datasets(
    train_dir: str,
    valid_dir: str,
    image_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> DatasetsTuple:
  """Builds the training and validation datasets.

  Args:
      train_dir: Path to the training directory in PyTorch ImageFolder format
        (one subdirectory per class).
      valid_dir: Path to the validation directory in PyTorch ImageFolder format
        (one subdirectory per class).
      image_size: Target side length in pixels for the square resize applied to
        both training and validation images.
      image_mean: Per-channel mean for normalization, as a 3-tuple of floats in
        `(R, G, B)` order.
      image_std: Per-channel standard deviation for normalization, as a 3-tuple
        of floats in `(R, G, B)` order.

  Returns:
      The `(dataset_train, dataset_valid, class_names)` tuple, where
      `class_names` is ordered alphabetically.
  """
  dataset_train = datasets.ImageFolder(
      train_dir,
      transform=_get_train_transform(image_size, image_mean, image_std),
  )
  dataset_valid = datasets.ImageFolder(
      valid_dir,
      transform=_get_valid_transform(image_size, image_mean, image_std),
  )
  return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(
    dataset_train: torch_data.Dataset,
    dataset_valid: torch_data.Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoadersTuple:
  """Builds the training and validation data loaders.

  Loaders use pinned memory, persistent workers, and prefetching to keep
  the GPU fed during training on CUDA hardware.

  Args:
      dataset_train: The training dataset.
      dataset_valid: The validation dataset.
      batch_size: Number of samples per batch for both loaders.
      num_workers: Number of parallel worker processes for data loading. Used
        for both loaders.

  Returns:
      The `(train_loader, valid_loader)` tuple. The training loader shuffles
      each epoch; the validation loader does not.
  """
  persistent_workers = num_workers > 0
  prefetch_factor = 4 if num_workers > 0 else None
  train_loader = torch_data.DataLoader(
      dataset_train,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=persistent_workers,
      prefetch_factor=prefetch_factor,
  )
  valid_loader = torch_data.DataLoader(
      dataset_valid,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=persistent_workers,
      prefetch_factor=prefetch_factor,
  )
  return train_loader, valid_loader
