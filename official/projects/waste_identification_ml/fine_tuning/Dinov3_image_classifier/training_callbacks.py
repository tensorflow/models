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
  """Saves the best model to disk when validation loss improves.

  Two checkpoint files are written each time an improvement is observed:
    1. The full model's state dict, at `<output_directory>/best_<name>.pth`.
    2. The model's head-only state dict, at
       `<output_directory>/best_head_<name>.pth`.

  The head-only checkpoint assumes the model exposes a `.head` submodule
  (as `Dinov3Classification` does). Passing a model without a `.head`
  attribute will raise `AttributeError` at save time.

  Attributes:
    best_validation_loss: Lowest validation loss observed so far.
  """

  def __init__(self, best_validation_loss: float = float("inf")):
    """Initializes the best-loss tracker.

    Args:
      best_validation_loss: Starting value for the best validation loss.
        Defaults to infinity so the first epoch always saves.
    """
    self.best_validation_loss = best_validation_loss

  def __call__(
      self,
      current_validation_loss: float,
      epoch: int,
      model: nn.Module,
      output_directory: pathlib.Path,
      checkpoint_name: str,
  ) -> None:
    """Saves the model if validation loss improved this epoch.

    Args:
      current_validation_loss: Validation loss for the current epoch.
      epoch: Zero-based epoch index.
      model: Model whose state should be saved. Must expose a `.head` submodule
        for the head-only checkpoint.
      output_directory: Directory to write the checkpoint files into.
      checkpoint_name: Base file name for the saved checkpoints (no extension).
    """
    if current_validation_loss >= self.best_validation_loss:
      return

    self.best_validation_loss = current_validation_loss
    _LOGGER.info(
        "Best validation loss: %s. Saving best model for epoch %d.",
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
  """Signals when training should stop after a plateau in validation loss.

  Tracks the best validation loss and counts how many epochs have passed
  without meaningful improvement. When the count reaches `patience`,
  further calls return True.

  Once the stop signal has been raised, subsequent calls continue to
  return True. Reuse of a triggered instance across separate training
  runs is not recommended; construct a fresh instance instead.

  Attributes:
    patience: Number of epochs without improvement after which stopping is
      signaled.
    minimum_delta: Minimum validation loss improvement that counts as progress.
    best_loss: Lowest validation loss observed so far.
    counter: Number of consecutive epochs without improvement.
    should_stop: Whether stopping has been signaled.
  """

  def __init__(self, patience: int = 7, minimum_delta: float = 0.0):
    """Initializes the early-stopping tracker.

    Args:
      patience: Number of epochs without improvement after which training will
        be stopped.
      minimum_delta: Minimum validation loss improvement that counts as
        progress.
    """
    self.patience = patience
    self.minimum_delta = minimum_delta
    self.best_loss = float("inf")
    self.counter = 0
    self.should_stop = False

  def __call__(self, current_validation_loss: float) -> bool:
    """Updates the counter and returns whether training should stop.

    Args:
      current_validation_loss: Validation loss for the current epoch.

    Returns:
      True if training should stop, False otherwise.
    """
    if current_validation_loss < self.best_loss - self.minimum_delta:
      self.best_loss = current_validation_loss
      self.counter = 0
      return self.should_stop

    self.counter += 1
    _LOGGER.info(
        "[EarlyStopping] No improvement. Counter: %d/%d",
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
  """Saves the final trained model and optimizer state to disk.

  Two checkpoints are written:
    1. The full model (backbone + head) and optimizer state, at
       `<output_directory>/<checkpoint_name>.pth`.
    2. The classifier head only plus optimizer state, at
       `<output_directory>/head_<checkpoint_name>.pth`.

  The head-only checkpoint assumes the model exposes a `.head` submodule
  (as `Dinov3Classification` does). Passing a model without a `.head`
  attribute will raise `AttributeError`.

  Args:
    epochs: Total number of epochs the model was trained for.
    model: Model whose state should be saved. Must expose a `.head` submodule
      for the head-only checkpoint.
    optimizer: Optimizer whose state should be saved.
    output_directory: Directory to write the checkpoint files into.
    checkpoint_name: Base file name for the saved checkpoints (no extension).
  """
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
  """Saves accuracy and loss curves as PNGs.

  The ggplot matplotlib style is applied only within this function so that
  importing this module does not mutate matplotlib's global style state.

  Args:
    train_accuracy: Per-epoch training accuracy values.
    validation_accuracy: Per-epoch validation accuracy values.
    train_loss: Per-epoch training loss values.
    validation_loss: Per-epoch validation loss values.
    output_directory: Directory to write the plot files into.
  """
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
  """Plots one train/validation pair to a PNG and closes the figure.

  Args:
    train_series: Per-epoch training values.
    validation_series: Per-epoch validation values.
    y_axis_label: Label for the y-axis (e.g. 'Accuracy', 'Loss').
    train_label: Legend label for the training curve.
    validation_label: Legend label for the validation curve.
    output_path: PNG path to write.
  """
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
