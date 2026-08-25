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

"""Training utilities: checkpointing, early stopping, plot saving."""

from collections.abc import Sequence
import logging
import math
import pathlib

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

_LOGGER = logging.getLogger(__name__)

_CHECKPOINT_EXTENSION = ".pth"
_MATPLOTLIB_STYLE = "ggplot"
_PLOT_FIGURE_SIZE = (10, 7)
_TRAIN_COLOR = "tab:blue"
_VALIDATION_COLOR = "tab:red"


class SaveBestModel:
  """Saves the best model to disk when validation loss improves."""

  def __init__(
      self,
      best_validation_loss: float = float("inf"),
      minimum_delta: float = 1e-4,
  ):
    self.best_validation_loss = best_validation_loss
    self.minimum_delta = minimum_delta

  def __call__(
      self,
      current_validation_loss: float,
      epoch: int,
      model: nn.Module,
      output_directory: pathlib.Path,
      checkpoint_name: str,
  ) -> None:
    if not math.isfinite(current_validation_loss):
      _LOGGER.warning(
          "[SaveBestModel] Non-finite validation loss (%s). Skipping checkpoint"
          " save.",
          current_validation_loss,
      )
      return

    # Requires a meaningful improvement exceeding minimum_delta
    if current_validation_loss > (
        self.best_validation_loss - self.minimum_delta
    ):
      return

    self.best_validation_loss = current_validation_loss
    _LOGGER.info(
        "Best validation loss improved to %.5f. Saving best model for"
        " epoch %d.",
        self.best_validation_loss,
        epoch + 1,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    full_checkpoint_path = output_directory / (
        f"best_{checkpoint_name}{_CHECKPOINT_EXTENSION}"
    )
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
        str(full_checkpoint_path),
    )

    head_checkpoint_path = output_directory / (
        f"best_head_{checkpoint_name}{_CHECKPOINT_EXTENSION}"
    )
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.head.state_dict()},
        str(head_checkpoint_path),
    )


class EarlyStopping:
  """Signals when training should stop after a plateau in validation loss."""

  def __init__(self, patience: int = 5, minimum_delta: float = 1e-4):
    self.patience = patience
    self.minimum_delta = minimum_delta
    self.best_loss = float("inf")
    self.counter = 0
    self.should_stop = False

  def __call__(self, current_validation_loss: float) -> bool:
    if not math.isfinite(current_validation_loss):
      _LOGGER.warning(
          "[EarlyStopping] Non-finite validation loss encountered. Stopping"
          " training."
      )
      self.should_stop = True
      return True

    if current_validation_loss <= (self.best_loss - self.minimum_delta):
      self.best_loss = current_validation_loss
      self.counter = 0
      return self.should_stop

    self.counter += 1
    _LOGGER.info(
        "[EarlyStopping] No improvement (delta < %.5f). Counter: %d/%d",
        self.minimum_delta,
        self.counter,
        self.patience,
    )
    if self.counter >= self.patience:
      self.should_stop = True
      _LOGGER.info(
          "[EarlyStopping] Triggered at patience=%d. Stopping training.",
          self.patience,
      )
    return self.should_stop


def save_model(
    epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    output_directory: pathlib.Path,
    checkpoint_name: str,
) -> None:
  """Saves the final trained model and optimizer state to disk."""
  output_directory.mkdir(parents=True, exist_ok=True)
  full_checkpoint_path = output_directory / (
      f"{checkpoint_name}{_CHECKPOINT_EXTENSION}"
  )
  torch.save(
      {
          "epoch": epochs,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
      },
      str(full_checkpoint_path),
  )

  head_checkpoint_path = output_directory / (
      f"head_{checkpoint_name}{_CHECKPOINT_EXTENSION}"
  )
  torch.save(
      {
          "epoch": epochs,
          "model_state_dict": model.head.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
      },
      str(head_checkpoint_path),
  )


def save_plots(
    train_accuracy: Sequence[float],
    validation_accuracy: Sequence[float],
    train_loss: Sequence[float],
    validation_loss: Sequence[float],
    output_directory: pathlib.Path,
) -> None:
  """Saves accuracy and loss curves as PNGs."""
  output_directory.mkdir(parents=True, exist_ok=True)
  with plt.style.context(_MATPLOTLIB_STYLE):
    _save_curve_pair(
        train_series=train_accuracy,
        validation_series=validation_accuracy,
        y_axis_label="Accuracy",
        train_label="train accuracy",
        validation_label="validation accuracy",
        output_path=output_directory / "accuracy.png",
    )
    _save_curve_pair(
        train_series=train_loss,
        validation_series=validation_loss,
        y_axis_label="Loss",
        train_label="train loss",
        validation_label="validation loss",
        output_path=output_directory / "loss.png",
    )


def _save_curve_pair(
    train_series: Sequence[float],
    validation_series: Sequence[float],
    y_axis_label: str,
    train_label: str,
    validation_label: str,
    output_path: pathlib.Path,
) -> None:
  """Plots one train/validation pair to a PNG and closes the figure."""
  figure = plt.figure(figsize=_PLOT_FIGURE_SIZE)
  try:
    plt.plot(train_series, color=_TRAIN_COLOR, linestyle="-", label=train_label)
    plt.plot(
        validation_series,
        color=_VALIDATION_COLOR,
        linestyle="-",
        label=validation_label,
    )
    plt.xlabel("Epochs")
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.savefig(str(output_path))
  finally:
    plt.close(figure)
