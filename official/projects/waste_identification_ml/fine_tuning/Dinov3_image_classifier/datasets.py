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
import logging
import os
import pathlib
from typing import TypeAlias

import torch
from torch.utils import data as torch_data
from torchvision import datasets
from torchvision.transforms import v2

_LOGGER = logging.getLogger(__name__)

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
  """Builds the training image transform pipeline."""
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
  """Builds the validation image transform pipeline."""
  return v2.Compose([
      v2.ToImage(),
      v2.Resize((image_size, image_size), antialias=True),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=image_mean, std=image_std),
  ])


def get_datasets(
    train_dir: str | pathlib.Path,
    valid_dir: str | pathlib.Path,
    image_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> DatasetsTuple:
  """Builds the training and validation datasets.

  Args:
      train_dir: Path to the training directory in PyTorch ImageFolder format.
      valid_dir: Path to the validation directory in PyTorch ImageFolder format.
      image_size: Target side length in pixels for the square resize.
      image_mean: Per-channel mean for normalization in (R, G, B) order.
      image_std: Per-channel standard deviation for normalization in (R, G, B)
        order.

  Returns:
      The (dataset_train, dataset_valid, class_names) tuple.
  """
  dataset_train = datasets.ImageFolder(
      os.fspath(train_dir),
      transform=_get_train_transform(image_size, image_mean, image_std),
  )
  dataset_valid = datasets.ImageFolder(
      os.fspath(valid_dir),
      transform=_get_valid_transform(image_size, image_mean, image_std),
  )
  return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(
    dataset_train: torch_data.Dataset,
    dataset_valid: torch_data.Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoadersTuple:
  """Builds the training and validation data loaders."""
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


def compute_balanced_class_weights(
    train_directory: str | pathlib.Path,
) -> torch.Tensor:
  """Computes sklearn-style balanced class weights from a directory tree."""
  train_directory = pathlib.Path(train_directory)
  class_names = sorted(
      entry.name for entry in train_directory.iterdir() if entry.is_dir()
  )
  if not class_names:
    raise ValueError(
        f"No class subdirectories found in train_directory: {train_directory}"
    )

  class_counts = []
  for class_name in class_names:
    class_path = train_directory / class_name
    number_of_files = sum(
        1 for entry in class_path.iterdir() if entry.is_file()
    )
    if number_of_files == 0:
      raise ValueError(f"Class directory is empty: {class_path}")
    class_counts.append(number_of_files)

  number_of_classes = len(class_counts)
  total_samples = sum(class_counts)
  weights = [
      total_samples / (number_of_classes * count) for count in class_counts
  ]

  _LOGGER.info("Class counts: %s", dict(zip(class_names, class_counts)))
  _LOGGER.info("Class weights: %s", dict(zip(class_names, weights)))

  return torch.tensor(weights, dtype=torch.float32)
