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
The optimizer is AdamW, and it treats the model as one thing: the same
learning rate and the same weight decay apply to every trainable
parameter — every layer of the backbone and the classification head
alike. There's no split-LR setup, no per-parameter-group weight decay,
and no warmup phase. The learning rate follows a plain cosine curve
that decays smoothly from its peak on epoch 0 to its floor on the final
epoch.

The peak learning rate is deliberately small. DINOv3's pretrained
features are already strong, and a low LR keeps updates gentle so the
backbone doesn't drift far from its self-supervised optimum while the
head has time to fit. Training uses bf16 autocast on CUDA for speed and
memory headroom, with gradient clipping (max L2 norm 1.0) as a safety
net. Each epoch, the script reports training and validation loss along
with top-1 accuracy.

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pylint: disable=g-import-not-at-top,wrong-import-position
import pathlib
import random
from typing import Sized, TypeAlias, cast

import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import auto as tqdm_auto

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import datasets
from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import models as model_module
from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import training_callbacks

# pylint: enable=wrong-import-position

_LOGGER = logging.getLogger(__name__)

EpochMetrics: TypeAlias = tuple[float, float]

SEED = 42

TRAIN_DIRECTORY = pathlib.Path(
    "/home/umairsabir/saahas/accepted_rejected/accepted_rejected_data/train/"
)
VALIDATION_DIRECTORY = pathlib.Path(
    "/home/umairsabir/saahas/accepted_rejected/accepted_rejected_data/val/"
)

DINOV3_REPO_DIRECTORY = pathlib.Path("/home/umairsabir/dinov3")
DINOV3_WEIGHTS_PATH = pathlib.Path(
    "/home/umairsabir/dinov3_original_weight/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
MODEL_NAME = "dinov3_vitl16"

OUTPUT_DIRECTORY = pathlib.Path(
    "/home/umairsabir/saahas/accepted_rejected/model_output/version_3/"
)
CHECKPOINT_NAME = "model"

EPOCHS = 30
BATCH_SIZE = 64
IMAGE_SIZE = 256
NUMBER_OF_WORKERS = 12

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

POOLING_STRATEGY = model_module.POOLING_CLS
FINE_TUNE = True

USE_CLASS_WEIGHTS = False

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MINIMUM_DELTA = 1e-4

# Fixed Cosine Scheduler Floor
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.02
COSINE_MINIMUM_LEARNING_RATE = 1e-8

GRADIENT_CLIP_MAX_NORM = 1.0
LOG_FILENAME = "training.log"
_PERCENT = 100.0


def seed_everything(seed: int) -> None:
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


def collect_trainable_parameters(
    classifier_model: nn.Module,
) -> list[nn.Parameter]:
  """Collects all trainable parameters into a single list.

  All trainable parameters share the same learning rate and weight decay;
  there is no head/backbone split and no weight-decay exclusion for biases
  or LayerNorm weights.

  Args:
    classifier_model: The model whose parameters will be collected.

  Returns:
    A list of `torch.nn.Parameter` objects with `requires_grad=True`.
  """
  trainable_parameters = [
      parameter
      for parameter in classifier_model.parameters()
      if parameter.requires_grad
  ]
  _LOGGER.info(
      "Trainable parameter tensors: %d, LR=%s, weight_decay=%s",
      len(trainable_parameters),
      LEARNING_RATE,
      WEIGHT_DECAY,
  )
  return trainable_parameters


def build_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    cosine_minimum_learning_rate: float,
) -> optim.lr_scheduler.CosineAnnealingLR:
  return optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=total_epochs,
      eta_min=cosine_minimum_learning_rate,
  )


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
    gradient_clip_max_norm: Maximum L2 norm for gradient clipping. The gradients
      of all trainable parameters are rescaled in-place so that their combined
      L2 norm does not exceed this value.

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
  seed_everything(SEED)
  model_module.validate_image_size(IMAGE_SIZE, model_module.DINOV3_PATCH_SIZE)

  OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
  configure_logging(OUTPUT_DIRECTORY)

  dataset_train, dataset_valid, class_names = datasets.get_datasets(
      train_dir=TRAIN_DIRECTORY,
      valid_dir=VALIDATION_DIRECTORY,
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
  _LOGGER.info("Learning rate: %s", LEARNING_RATE)
  _LOGGER.info("Weight decay: %s", WEIGHT_DECAY)
  _LOGGER.info("Cosine min LR: %s", COSINE_MINIMUM_LEARNING_RATE)
  _LOGGER.info("Schedule: pure cosine annealing (no warmup)")
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

  trainable_parameters = collect_trainable_parameters(classifier_model)
  optimizer = optim.AdamW(
      trainable_parameters,
      lr=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
  )

  if USE_CLASS_WEIGHTS:
    class_weights = datasets.compute_balanced_class_weights(TRAIN_DIRECTORY).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
  else:
    criterion = nn.CrossEntropyLoss()

  save_best_model = training_callbacks.SaveBestModel(
      minimum_delta=EARLY_STOPPING_MINIMUM_DELTA
  )

  early_stopping = None
  if EARLY_STOPPING_PATIENCE > 0:
    early_stopping = training_callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        minimum_delta=EARLY_STOPPING_MINIMUM_DELTA,
    )

  scheduler = build_cosine_scheduler(
      optimizer=optimizer,
      total_epochs=EPOCHS,
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
