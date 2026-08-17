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

"""Training entry point for the DINOv3 image classifier.

This script fine-tunes a DINOv3 ViT-L/16 backbone with a fresh linear
classification head on your own image dataset. Point it at a folder laid
out in PyTorch's `ImageFolder` format (one subdirectory per class), and
it handles the rest: loading the pretrained backbone from a local clone
of the Facebook DINOv3 repository, sizing the head to the classes it
discovers, and training end-to-end on a single GPU.

How training works
------------------
The optimizer is AdamW, and it treats different parts of the model
differently. Parameters are split into four groups so that (a) the
pretrained backbone can train slowly at one learning rate while the
freshly initialized head trains faster at another, and (b) weight decay
is applied only to the parameters that actually benefit from it — the 2D
weight matrices — while biases and 1D parameters (LayerNorm weights and
similar) are excluded, following standard ViT fine-tuning practice.

The head trains at a much larger learning rate than the backbone. The
head starts from random initialization and needs to fit; the backbone
starts from strong pretrained features and needs to nudge, not shift.
This split is one of the most consistent ways to get stable fine-tuning
on top of a self-supervised backbone.

The learning-rate schedule is linear warmup followed by cosine annealing.
For the first ~10% of epochs, the LR climbs linearly from 1% of its peak
up to the peak; after that, it decays smoothly along a cosine curve down
to a small floor. Warmup keeps early optimizer updates small while
AdamW's moment estimates are still settling — important when the head's
LR is high enough that a cold start could destabilize training.

Training uses bf16 autocast on CUDA for speed and memory headroom, with
gradient clipping (max L2 norm 1.0) as a safety net. Each epoch, the
script reports training and validation loss along with top-1 accuracy.

Reproducibility and performance
-------------------------------
Python, NumPy, and PyTorch RNGs are seeded from a fixed constant so runs
are broadly repeatable. cuDNN runs in benchmark mode (not deterministic)
and TF32 matmul is enabled — both trade a little run-to-run determinism
for meaningful throughput gains on Ampere+ GPUs. If you need bit-exact
reproducibility more than you need speed, flip the flags in
`seed_everything`.

Callbacks and outputs
---------------------
Two callbacks run alongside the loop: `SaveBestModel` writes a
checkpoint whenever validation loss improves, and `EarlyStopping` halts
training if validation loss stops improving for a configurable number of
consecutive epochs. When training ends (either by finishing the schedule
or by hitting early stopping), a final checkpoint and accuracy/loss
plots are written to `OUTPUT_DIRECTORY`.

If your classes are imbalanced, set `USE_CLASS_WEIGHTS = True`. The
script will count files per class folder, compute sklearn-style balanced
weights, and pass them to `CrossEntropyLoss`.

Configuration
-------------
All configuration lives as module-level constants at the top of this
file. There are no CLI arguments and no external config — edit the
constants in place to change the recipe. Checkpoints, plots, and the
training log all land in `OUTPUT_DIRECTORY`.
"""

import logging
import os
import pathlib
import random
from typing import Any, Sized, TypeAlias, cast

# Must be set before importing torch so CUDA picks up the right device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pylint: disable=g-import-not-at-top,wrong-import-position

import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import auto as tqdm_auto

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import datasets
from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import models as model_module
from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import training_callbacks
# pylint: enable=g-import-not-at-top,wrong-import-position

_LOGGER = logging.getLogger(__name__)

EpochMetrics: TypeAlias = tuple[float, float]
ParameterGroup: TypeAlias = dict[str, Any]

# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset paths (PyTorch ImageFolder format: one subdirectory per class).
# ---------------------------------------------------------------------------
TRAIN_DIRECTORY = pathlib.Path(
    "/home/umairsabir/pfc/bottle_grade_detection_classifier/train/"
)
VALIDATION_DIRECTORY = pathlib.Path(
    "/home/umairsabir/pfc/bottle_grade_detection_classifier/val/"
)

# ---------------------------------------------------------------------------
# Backbone configuration.
#
# DINOV3_REPO_DIRECTORY is the path to the cloned DINOv3 repository. It is
# used by torch.hub.load with source='local' to load the model architecture
# without hitting the internet.
#
# DINOV3_WEIGHTS_PATH is the full path to the pretrained backbone weights
# (.pth file).
#
# MODEL_NAME must match an entry in the DINOv3 hub:
#   https://github.com/facebookresearch/dinov3
# ---------------------------------------------------------------------------
DINOV3_REPO_DIRECTORY = pathlib.Path("/home/umairsabir/dinov3")
DINOV3_WEIGHTS_PATH = pathlib.Path(
    "/home/umairsabir/dinov3_weights/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
MODEL_NAME = "dinov3_vitl16"

# ---------------------------------------------------------------------------
# Output directory. Saved checkpoints and plots are written here.
# ---------------------------------------------------------------------------
OUTPUT_DIRECTORY = pathlib.Path(
    "/home/umairsabir/dinov3-image-classifier/training/output/version_1/"
)
CHECKPOINT_NAME = "model"

# ---------------------------------------------------------------------------
# Training schedule.
# ---------------------------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 64
IMAGE_SIZE = 256

# ---------------------------------------------------------------------------
# Data loading.
#
# NUMBER_OF_WORKERS is the number of parallel worker processes used by the
# DataLoader. Set this based on the number of CPU cores available on the
# training machine.
# ---------------------------------------------------------------------------
NUMBER_OF_WORKERS = 12

# ---------------------------------------------------------------------------
# Image normalization statistics.
#
# DINOv3 backbones expect ImageNet-style normalization. These values are
# tied to the pretrained backbone and should not be changed unless the
# backbone itself is retrained with different statistics.
# ---------------------------------------------------------------------------
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Model head configuration.
#
# POOLING_STRATEGY controls how features feed the classifier head.
#   - POOLING_CLS: use only the final CLS token.
#   - POOLING_CLS_MEAN_PATCH: concatenate CLS token with the mean of final
#     patch tokens, doubling the head input dimension.
#
# FINE_TUNE=True trains the full backbone; False trains only the head.
# ---------------------------------------------------------------------------
POOLING_STRATEGY = model_module.POOLING_CLS
FINE_TUNE = True

# ---------------------------------------------------------------------------
# Class imbalance handling. When True, sklearn-style balanced class weights
# are computed from file counts per class folder in TRAIN_DIRECTORY.
# ---------------------------------------------------------------------------
USE_CLASS_WEIGHTS = False

# ---------------------------------------------------------------------------
# Early stopping. Set EARLY_STOPPING_PATIENCE to 0 to disable.
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MINIMUM_DELTA = 0.0

# ---------------------------------------------------------------------------
# Optimizer / scheduler hyperparameters.
#
# The backbone uses a lower LR than the head (standard for fine-tuning a
# pretrained backbone). The schedule is a linear warmup for the first
# WARMUP_EPOCHS_FRACTION of training, followed by cosine annealing down to
# COSINE_MINIMUM_LEARNING_RATE for the remaining epochs.
# ---------------------------------------------------------------------------
BACKBONE_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS_FRACTION = 0.1
WARMUP_START_FACTOR = 0.01
COSINE_MINIMUM_LEARNING_RATE = 1e-6

# ---------------------------------------------------------------------------
# Gradient clipping. The max L2 norm allowed for gradients in each step;
# gradients are scaled down if their norm exceeds this value. The standard
# value for supervised ViT fine-tuning is 1.0.
# ---------------------------------------------------------------------------
GRADIENT_CLIP_MAX_NORM = 1.0

# ---------------------------------------------------------------------------
# Logging. The log file is written inside OUTPUT_DIRECTORY alongside the
# saved checkpoints and plots.
# ---------------------------------------------------------------------------
LOG_FILENAME = "training.log"

# ---------------------------------------------------------------------------
# Percent conversion factor for accuracy reporting.
# ---------------------------------------------------------------------------
_PERCENT = 100.0

# ---------------------------------------------------------------------------
# Prefix used to identify head parameters via `nn.Module.named_parameters`.
# `Dinov3Classification` exposes its classification head as `self.head`, so
# every head-owned parameter name starts with 'head.'.
# ---------------------------------------------------------------------------
_HEAD_PARAMETER_PREFIX = "head."


def seed_everything(seed: int) -> None:
  """Seeds Python, NumPy, and PyTorch RNGs and configures cuDNN for speed.

  Sets `cudnn.deterministic = False` and `cudnn.benchmark = True` for
  throughput; flip these if strict reproducibility is required. Also
  enables TF32 matmul on Ampere+ GPUs.

  Args:
    seed: Integer seed applied to all RNGs.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision("high")


def configure_logging(output_directory: pathlib.Path) -> None:
  """Configures the root logger to write to both console and a file.

  The console handler shows the bare message. The file handler prepends an
  ISO-8601 timestamp and the log level so the log file is easy to scan
  after training. `tqdm` progress bars write to stderr and are not
  captured here, so the log file stays free of progress-bar carriage
  returns.

  Args:
    output_directory: Directory where the log file is created. Must already
      exist.
  """
  log_path = output_directory / LOG_FILENAME

  root_logger = logging.getLogger()
  root_logger.setLevel(logging.INFO)
  # Prevent duplicate handlers if this function is called more than once
  # (e.g., from an interactive session).
  root_logger.handlers.clear()

  console_handler = logging.StreamHandler()
  console_handler.setFormatter(logging.Formatter("%(message)s"))
  root_logger.addHandler(console_handler)

  file_handler = logging.FileHandler(str(log_path), mode="w")
  file_handler.setFormatter(
      logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  )
  root_logger.addHandler(file_handler)

  _LOGGER.info("Logging to: %s", log_path)


def build_optimizer_parameter_groups(
    classifier_model: nn.Module,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[ParameterGroup]:
  """Builds AdamW parameter groups with split LRs and selective weight decay.

  Splits trainable parameters into four groups:
    1. Backbone parameters that should receive weight decay.
    2. Backbone parameters that should NOT receive weight decay (biases and
       1D parameters such as LayerNorm weights).
    3. Head parameters that should receive weight decay.
    4. Head parameters that should NOT receive weight decay.

  Backbone groups use `backbone_learning_rate`; head groups use
  `head_learning_rate`. The classifier head is identified by the attribute
  name `head` on `Dinov3Classification`.

  Args:
    classifier_model: The `Dinov3Classification` model whose parameters will
      be grouped.
    backbone_learning_rate: Learning rate applied to all backbone
      parameters.
    head_learning_rate: Learning rate applied to all classifier head
      parameters.
    weight_decay: Weight decay value applied to non-bias, non-norm
      parameters. Bias and 1D parameters get weight decay 0.

  Returns:
    A list of parameter group dictionaries suitable for passing to an
    `AdamW` optimizer.
  """
  backbone_decay_parameters: list[nn.Parameter] = []
  backbone_no_decay_parameters: list[nn.Parameter] = []
  head_decay_parameters: list[nn.Parameter] = []
  head_no_decay_parameters: list[nn.Parameter] = []

  for parameter_name, parameter in classifier_model.named_parameters():
    if not parameter.requires_grad:
      continue

    is_head_parameter = parameter_name.startswith(_HEAD_PARAMETER_PREFIX)
    # Exclude biases and 1D parameters (e.g., LayerNorm weights) from
    # weight decay. This is standard practice for ViT fine-tuning.
    excluded_from_weight_decay = (
        parameter.ndim <= 1 or parameter_name.endswith(".bias")
    )

    if is_head_parameter and excluded_from_weight_decay:
      head_no_decay_parameters.append(parameter)
    elif is_head_parameter:
      head_decay_parameters.append(parameter)
    elif excluded_from_weight_decay:
      backbone_no_decay_parameters.append(parameter)
    else:
      backbone_decay_parameters.append(parameter)

  parameter_groups: list[ParameterGroup] = [
      {
          "params": backbone_decay_parameters,
          "lr": backbone_learning_rate,
          "weight_decay": weight_decay,
      },
      {
          "params": backbone_no_decay_parameters,
          "lr": backbone_learning_rate,
          "weight_decay": 0.0,
      },
      {
          "params": head_decay_parameters,
          "lr": head_learning_rate,
          "weight_decay": weight_decay,
      },
      {
          "params": head_no_decay_parameters,
          "lr": head_learning_rate,
          "weight_decay": 0.0,
      },
  ]

  _LOGGER.info(
      "Backbone params (decay/no-decay): %d/%d, LR=%s",
      len(backbone_decay_parameters),
      len(backbone_no_decay_parameters),
      backbone_learning_rate,
  )
  _LOGGER.info(
      "Head params (decay/no-decay): %d/%d, LR=%s",
      len(head_decay_parameters),
      len(head_no_decay_parameters),
      head_learning_rate,
  )

  return parameter_groups


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float,
    cosine_minimum_learning_rate: float,
) -> optim.lr_scheduler.SequentialLR:
  """Builds a linear-warmup followed by cosine-annealing LR scheduler.

  During the first `warmup_epochs`, the learning rate scales linearly from
  `warmup_start_factor * base_lr` up to `base_lr`. After that, it follows
  a cosine annealing schedule down to `cosine_minimum_learning_rate` over
  the remaining epochs.

  Args:
    optimizer: The optimizer whose LR will be scheduled.
    total_epochs: Total number of training epochs.
    warmup_epochs: Number of warmup epochs at the start of training. Must be
      at least 1 and strictly less than `total_epochs`.
    warmup_start_factor: Multiplier on the base LR at the very first step
      of warmup (e.g., 0.01 means start at 1% of base LR).
    cosine_minimum_learning_rate: Floor value for the cosine annealing
      phase.

  Returns:
    A `torch.optim.lr_scheduler.SequentialLR` combining the warmup and
    cosine schedulers.
  """
  warmup_scheduler = optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=warmup_start_factor,
      end_factor=1.0,
      total_iters=warmup_epochs,
  )
  cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=total_epochs - warmup_epochs,
      eta_min=cosine_minimum_learning_rate,
  )
  return optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[warmup_epochs],
  )


def compute_warmup_epochs(
    total_epochs: int, warmup_epochs_fraction: float
) -> int:
  """Computes the number of warmup epochs from a fraction of total epochs.

  Guarantees at least 1 warmup epoch and at least 1 cosine epoch remaining.

  Args:
    total_epochs: Total number of training epochs.
    warmup_epochs_fraction: Fraction of total epochs to spend in warmup
      (e.g., 0.1 for 10%).

  Returns:
    The number of warmup epochs, clamped to `[1, total_epochs - 1]`.
  """
  fractional_warmup_epochs = int(round(total_epochs * warmup_epochs_fraction))
  return min(max(1, fractional_warmup_epochs), max(1, total_epochs - 1))


def train_one_epoch(
    classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_max_norm: float,
) -> EpochMetrics:
  """Runs a single training epoch with bf16 mixed precision.

  Args:
    classifier_model: The model to train.
    train_loader: DataLoader yielding training batches.
    optimizer: Optimizer used to update model parameters.
    criterion: Loss function.
    device: Torch device to run computation on.
    gradient_clip_max_norm: Maximum L2 norm for gradient clipping. The
      gradients of all trainable parameters are rescaled in-place so that
      their combined L2 norm does not exceed this value.

  Returns:
    An `(epoch_loss, epoch_accuracy)` tuple where `epoch_loss` is the mean
    loss across batches and `epoch_accuracy` is the top-1 accuracy as a
    percentage.
  """
  classifier_model.train()
  _LOGGER.info("Training")
  running_loss = 0.0
  running_correct = 0
  batch_count = 0

  for images, labels in tqdm_auto.tqdm(train_loader, total=len(train_loader)):
    batch_count += 1
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      outputs = classifier_model(images)
      loss = criterion(outputs, labels)

    running_loss += loss.item()
    _, predictions = torch.max(outputs.data, 1)
    running_correct += (predictions == labels).sum().item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        classifier_model.parameters(), max_norm=gradient_clip_max_norm
    )
    optimizer.step()

  epoch_loss = running_loss / batch_count
  epoch_accuracy = _PERCENT * (
      running_correct / len(cast(Sized, train_loader.dataset))
  )
  return epoch_loss, epoch_accuracy


def validate(
    classifier_model: nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
  """Runs a single validation pass with bf16 mixed precision.

  Args:
    classifier_model: The model to evaluate.
    validation_loader: DataLoader yielding validation batches.
    criterion: Loss function.
    device: Torch device to run computation on.

  Returns:
    An `(epoch_loss, epoch_accuracy)` tuple where `epoch_loss` is the mean
    loss across batches and `epoch_accuracy` is the top-1 accuracy as a
    percentage.
  """
  classifier_model.eval()
  _LOGGER.info("Validation")
  running_loss = 0.0
  running_correct = 0
  batch_count = 0

  with torch.no_grad():
    for images, labels in tqdm_auto.tqdm(
        validation_loader, total=len(validation_loader)
    ):
      batch_count += 1
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = classifier_model(images)
        loss = criterion(outputs, labels)

      running_loss += loss.item()
      _, predictions = torch.max(outputs.data, 1)
      running_correct += (predictions == labels).sum().item()

  epoch_loss = running_loss / batch_count
  epoch_accuracy = _PERCENT * (
      running_correct / len(cast(Sized, validation_loader.dataset))
  )
  return epoch_loss, epoch_accuracy


def main() -> None:
  """Runs the full training and validation loop."""
  seed_everything(SEED)
  model_module.validate_image_size(IMAGE_SIZE, model_module.DINOV3_PATCH_SIZE)

  OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
  configure_logging(OUTPUT_DIRECTORY)

  dataset_train, dataset_valid, class_names = datasets.get_datasets(
      train_dir=str(TRAIN_DIRECTORY),
      valid_dir=str(VALIDATION_DIRECTORY),
      image_size=IMAGE_SIZE,
      image_mean=IMAGE_MEAN,
      image_std=IMAGE_STD,
  )
  _LOGGER.info("Number of training images: %d", len(dataset_train))
  _LOGGER.info("Number of validation images: %d", len(dataset_valid))
  _LOGGER.info("Classes: %s", class_names)

  train_loader, validation_loader = datasets.get_data_loaders(
      dataset_train=dataset_train,
      dataset_valid=dataset_valid,
      batch_size=BATCH_SIZE,
      num_workers=NUMBER_OF_WORKERS,
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  _LOGGER.info("Computation device: %s", device)
  _LOGGER.info("Image size: %d", IMAGE_SIZE)
  _LOGGER.info("Pooling: %s", POOLING_STRATEGY)
  _LOGGER.info("Backbone LR: %s", BACKBONE_LEARNING_RATE)
  _LOGGER.info("Head LR: %s", HEAD_LEARNING_RATE)
  _LOGGER.info("Weight decay: %s", WEIGHT_DECAY)
  _LOGGER.info("Epochs to train for: %d", EPOCHS)

  classifier_model = model_module.Dinov3Classification.from_model_name(
      model_name=MODEL_NAME,
      repo_dir=DINOV3_REPO_DIRECTORY,
      number_of_classes=len(class_names),
      weights=DINOV3_WEIGHTS_PATH,
      pooling=POOLING_STRATEGY,
      fine_tune=FINE_TUNE,
  ).to(device)
  _LOGGER.info("Model architecture:\n%s", classifier_model)

  total_parameters = sum(p.numel() for p in classifier_model.parameters())
  _LOGGER.info("%s total parameters.", f"{total_parameters:,}")
  total_trainable_parameters = sum(
      p.numel() for p in classifier_model.parameters() if p.requires_grad
  )
  _LOGGER.info("%s training parameters.", f"{total_trainable_parameters:,}")

  optimizer_parameter_groups = build_optimizer_parameter_groups(
      classifier_model=classifier_model,
      backbone_learning_rate=BACKBONE_LEARNING_RATE,
      head_learning_rate=HEAD_LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
  )
  optimizer = optim.AdamW(optimizer_parameter_groups)

  if USE_CLASS_WEIGHTS:
    class_weights = datasets.compute_balanced_class_weights(
        TRAIN_DIRECTORY
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
  else:
    criterion = nn.CrossEntropyLoss()

  save_best_model = training_callbacks.SaveBestModel()

  early_stopping = None
  if EARLY_STOPPING_PATIENCE > 0:
    early_stopping = training_callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        minimum_delta=EARLY_STOPPING_MINIMUM_DELTA,
    )

  warmup_epochs = compute_warmup_epochs(EPOCHS, WARMUP_EPOCHS_FRACTION)
  _LOGGER.info("Warmup epochs: %d/%d", warmup_epochs, EPOCHS)
  scheduler = build_warmup_cosine_scheduler(
      optimizer=optimizer,
      total_epochs=EPOCHS,
      warmup_epochs=warmup_epochs,
      warmup_start_factor=WARMUP_START_FACTOR,
      cosine_minimum_learning_rate=COSINE_MINIMUM_LEARNING_RATE,
  )

  train_loss_history: list[float] = []
  validation_loss_history: list[float] = []
  train_accuracy_history: list[float] = []
  validation_accuracy_history: list[float] = []

  for epoch in range(EPOCHS):
    _LOGGER.info("Epoch %d of %d", epoch + 1, EPOCHS)
    train_epoch_loss, train_epoch_accuracy = train_one_epoch(
        classifier_model=classifier_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        gradient_clip_max_norm=GRADIENT_CLIP_MAX_NORM,
    )
    validation_epoch_loss, validation_epoch_accuracy = validate(
        classifier_model=classifier_model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    train_loss_history.append(train_epoch_loss)
    validation_loss_history.append(validation_epoch_loss)
    train_accuracy_history.append(train_epoch_accuracy)
    validation_accuracy_history.append(validation_epoch_accuracy)

    _LOGGER.info(
        "Training loss: %.3f, training acc: %.3f",
        train_epoch_loss,
        train_epoch_accuracy,
    )
    _LOGGER.info(
        "Validation loss: %.3f, validation acc: %.3f",
        validation_epoch_loss,
        validation_epoch_accuracy,
    )

    save_best_model(
        current_validation_loss=validation_epoch_loss,
        epoch=epoch,
        model=classifier_model,
        output_directory=OUTPUT_DIRECTORY,
        checkpoint_name=CHECKPOINT_NAME,
    )

    if early_stopping is not None and early_stopping(validation_epoch_loss):
      _LOGGER.info("Stopping early at epoch %d/%d", epoch + 1, EPOCHS)
      _LOGGER.info("-" * 50)
      break

    _LOGGER.info("-" * 50)
    scheduler.step()
    _LOGGER.info("LR for next epoch: %s", scheduler.get_last_lr())

  training_callbacks.save_model(
      epochs=EPOCHS,
      model=classifier_model,
      optimizer=optimizer,
      output_directory=OUTPUT_DIRECTORY,
      checkpoint_name=CHECKPOINT_NAME,
  )
  training_callbacks.save_plots(
      train_accuracy=train_accuracy_history,
      validation_accuracy=validation_accuracy_history,
      train_loss=train_loss_history,
      validation_loss=validation_loss_history,
      output_directory=OUTPUT_DIRECTORY,
  )
  _LOGGER.info("TRAINING COMPLETE")


if __name__ == "__main__":
  main()
