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

# -*- coding: utf-8 -*-
"""Trains an image classifier with a pretrained Vision Transformer backbone.

The script fine-tunes a ViT-B/16 model on a custom image dataset. Only the
classifier head is trained; all backbone parameters stay frozen.

The dataset is expected to follow the ``torchvision.datasets.ImageFolder``
layout::

    dataset/
    |-- train/
    |   |-- category_1/
    |   |-- category_2/
    |-- val/
        |-- category_1/
        |-- category_2/

Required local dependencies:
    * training_with_callbacks.py must be importable.

Example:
    $ python training.py
"""

from collections.abc import Mapping
import os
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data as torch_data
import torchvision
from torchvision import datasets
import training_with_callbacks

# Selected before pyplot is used so the script runs on headless machines.
matplotlib.use("Agg")

# Dataset locations.
TRAIN_DATA_DIRECTORY = "/home/umairsabir/saahas/milk_others/data_copy/train"
VALIDATION_DATA_DIRECTORY = "/home/umairsabir/saahas/milk_others/data_copy/val"

# Destination for the best checkpoint written by the early stopping callback.
MODEL_OUTPUT_PATH = "/home/umairsabir/vit_classifier/output_2/"

# Loss curve image, written into the same directory as the model checkpoint.
LOSS_CURVE_FILENAME = "loss_curves.png"

# Seed applied to the CPU and CUDA generators before the head is created.
RANDOM_SEED = 42

# Data loading settings. The worker count is detected at import time; see
# detect_number_of_dataloader_workers below.
BATCH_SIZE = 64

# Upper bound on dataloader workers. Beyond roughly this many, each extra
# worker costs memory and startup time without feeding the GPU any faster.
MAXIMUM_NUMBER_OF_WORKERS = 8

# Optimization settings. The scheduler steps once per epoch inside
# training_with_callbacks.train, so T_max is expressed in epochs and is set to
# the full run so the cosine performs a single decay rather than repeating.
NUMBER_OF_EPOCHS = 50
LEARNING_RATE = 1e-5
SCHEDULER_MAXIMUM_STEPS = NUMBER_OF_EPOCHS
SCHEDULER_MINIMUM_LEARNING_RATE = 1e-6

# Early stopping settings.
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 0.001


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
  """Seeds the PyTorch random number generators for reproducible runs.

  Args:
      seed: Value applied to both the CPU and the CUDA generators.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def plot_loss_curves(results: Mapping[str, list[float]]) -> None:
  """Plots training and validation loss and accuracy side by side.

  Draws into a new matplotlib figure. The caller is responsible for saving
  or closing it.

  Args:
      results: Metric history returned by training_with_callbacks.train, holding
        the keys "train_loss", "test_loss", "train_acc" and "test_acc". Each
        value holds one entry per completed epoch.
  """
  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, results["train_loss"], label="train_loss")
  plt.plot(epochs, results["test_loss"], label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, results["train_acc"], label="train_accuracy")
  plt.plot(epochs, results["test_acc"], label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


def detect_number_of_dataloader_workers() -> int:
  """Returns a dataloader worker count suited to the current machine.

  Prefers the number of CPUs the process is actually allowed to use, which
  on a container or VM can be lower than the number of CPUs the host
  reports. Falls back to the host count where the affinity call is not
  available, for example on Windows and macOS.

  Returns:
      The usable CPU count, capped at MAXIMUM_NUMBER_OF_WORKERS and never
      below one.
  """
  if hasattr(os, "sched_getaffinity"):
    available_cpu_count = len(os.sched_getaffinity(0))
  else:
    available_cpu_count = os.cpu_count() or 1

  return max(1, min(available_cpu_count, MAXIMUM_NUMBER_OF_WORKERS))


NUMBER_OF_WORKERS = detect_number_of_dataloader_workers()


def create_dataloaders(
    train_directory: str,
    validation_directory: str,
    transform: Callable[..., torch.Tensor],
    batch_size: int,
    number_of_workers: int = NUMBER_OF_WORKERS,
) -> tuple[torch_data.DataLoader, torch_data.DataLoader, list[str]]:
  """Builds training and validation dataloaders from image folders.

  Args:
      train_directory: Path to the training split, one subfolder per class.
      validation_directory: Path to the validation split, one subfolder per
        class.
      transform: Preprocessing applied to every image.
      batch_size: Number of samples per batch.
      number_of_workers: Number of subprocesses used for data loading.

  Returns:
      A tuple of (training dataloader, validation dataloader, class names).
      The class names are taken from the training split and are sorted
      alphabetically by ``ImageFolder``.
  """
  train_dataset = datasets.ImageFolder(train_directory, transform=transform)
  validation_dataset = datasets.ImageFolder(
      validation_directory, transform=transform
  )

  train_dataloader = torch_data.DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=number_of_workers,
      pin_memory=True,
  )
  validation_dataloader = torch_data.DataLoader(
      validation_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=number_of_workers,
      pin_memory=True,
  )

  return train_dataloader, validation_dataloader, train_dataset.classes


def build_pretrained_vit_classifier(
    number_of_classes: int,
    device: str,
) -> tuple[nn.Module, Callable[..., torch.Tensor]]:
  """Creates a ViT-B/16 model with a frozen backbone and a new head.

  Args:
      number_of_classes: Number of output classes for the classifier head.
      device: Device the model is moved to, for example "cuda" or "cpu".

  Returns:
      A tuple of (model, preprocessing transform). The transform is the one
      the pretrained weights were trained with and should be applied to the
      custom dataset as well.
  """
  pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  model = torchvision.models.vit_b_16(weights=pretrained_weights).to(device)

  for parameter in model.parameters():
    parameter.requires_grad = False

  set_random_seeds()
  model.heads = nn.Linear(in_features=768, out_features=number_of_classes).to(
      device
  )

  return model, pretrained_weights.transforms()


def main() -> None:
  """Runs the full fine-tuning pipeline and plots the loss curves."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")
  print(f"Dataloader workers: {NUMBER_OF_WORKERS}")

  # The head size depends on the dataset, so the transform is taken from a
  # throwaway model built with a single output before the real one is made.
  pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  preprocessing_transform = pretrained_weights.transforms()
  print(preprocessing_transform)

  train_dataloader, validation_dataloader, class_names = create_dataloaders(
      train_directory=TRAIN_DATA_DIRECTORY,
      validation_directory=VALIDATION_DATA_DIRECTORY,
      transform=preprocessing_transform,
      batch_size=BATCH_SIZE,
  )
  print(f"Found {len(class_names)} classes: {class_names}")

  model, _ = build_pretrained_vit_classifier(
      number_of_classes=len(class_names), device=device
  )

  early_stopping = training_with_callbacks.EarlyStopping(
      patience=EARLY_STOPPING_PATIENCE,
      delta=EARLY_STOPPING_DELTA,
      verbose=True,
      base_path=MODEL_OUTPUT_PATH,
  )

  # To counteract class imbalance, pass per-class weights to the loss:
  # weight = total_samples / (number_of_classes * samples_per_class)
  loss_function = nn.CrossEntropyLoss()

  optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=SCHEDULER_MAXIMUM_STEPS,
      eta_min=SCHEDULER_MINIMUM_LEARNING_RATE,
  )

  training_results = training_with_callbacks.train(
      model=model,
      train_dataloader=train_dataloader,
      test_dataloader=validation_dataloader,
      optimizer=optimizer,
      loss_fn=loss_function,
      epochs=NUMBER_OF_EPOCHS,
      device=device,
      early_stopping=early_stopping,
      scheduler=scheduler,
  )

  plot_loss_curves(training_results)
  loss_curve_path = os.path.join(
      os.path.dirname(MODEL_OUTPUT_PATH), LOSS_CURVE_FILENAME
  )
  plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
  plt.close("all")
  print(f"Saved loss curves to: {loss_curve_path}")


if __name__ == "__main__":
  os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
  main()
