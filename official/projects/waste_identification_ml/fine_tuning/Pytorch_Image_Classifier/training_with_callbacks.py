# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Contains functions for training PyTorch models with callbacks."""

from collections.abc import Mapping
import torch
import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
   model: A PyTorch model to be trained.
   dataloader: A DataLoader instance for the model to be trained on.
   loss_fn: A PyTorch loss function to minimize.
   optimizer: A PyTorch optimizer to help minimize the loss function.
   device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
   A tuple of training loss and training accuracy metrics.
   In the form (train_loss, train_accuracy). For example:
   (0.1112, 0.8743)
  """
  model.train()

  train_loss, train_acc = 0, 0

  for _, (inputs, labels) in enumerate(dataloader):
    inputs, labels = inputs.to(device), labels.to(device)

    y_pred = model(inputs)

    loss = loss_fn(y_pred, labels)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == labels).sum().item() / len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
   model: A PyTorch model to be tested.
   dataloader: A DataLoader instance for the model to be tested on.
   loss_fn: A PyTorch loss function to calculate loss on the test data.
   device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
   A tuple of testing loss and testing accuracy metrics.
   In the form (test_loss, test_accuracy). For example:
   (0.0223, 0.8985)
  """
  model.eval()

  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    for _, (inputs, y) in enumerate(dataloader):
      inputs, y = inputs.to(device), y.to(device)

      test_pred_logits = model(inputs)

      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    early_stopping=None,
    scheduler=None,
) -> Mapping[str, list[float]]:
  """Modified train function to include early stopping and checkpoint saving."""

  results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

  model.to(device)

  for epoch in tqdm.tqdm(range(epochs)):
    train_loss, train_acc = train_step(
        model, train_dataloader, loss_fn, optimizer, device
    )
    test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    if scheduler:
      scheduler.step()

    if early_stopping:
      early_stopping.check(val_loss=test_loss, model=model, epoch=epoch)
      if early_stopping.stop_training:
        print(f"EarlyStopping Triggered at epoch {epoch+1}.")
        break

  return results


class EarlyStopping:
  """EarlyStopping class to stop training when a metric has stopped improving.

  Saves the best model checkpoint based on the monitored validation loss.
  """

  def __init__(
      self, patience=5, delta=0.0, verbose=False, base_path="best_model"
  ):
    """Initializes the EarlyStopping callback.

    Args:
      patience (int): How many number of epochs to waitafter no improvement
      in val loss.
      delta (float): Minimum loss difference to qualify as an improvement.
      verbose (bool): Print updates.
      base_path (str): Base filename or path prefix for saving best
        checkpoints.
    """
    self.patience = patience
    self.delta = delta
    self.verbose = verbose
    self.base_path = base_path  # e.g., "checkpoints/vit"

    self.best_loss = float("inf")
    self.no_improvement_count = 0
    self.stop_training = False
    self.best_epoch = -1

  def check(self, val_loss: float, model: torch.nn.Module, epoch: int):
    if val_loss < self.best_loss - self.delta:
      self.best_loss = val_loss
      self.no_improvement_count = 0
      self.best_epoch = epoch
      self.save_checkpoint(model, epoch)
    else:
      self.no_improvement_count += 1
      if self.no_improvement_count >= self.patience:
        self.stop_training = True
        if self.verbose:
          print(
              f"EarlyStopping No improvement for {self.patience} epochs."
              " Stopping early."
          )

  def save_checkpoint(self, model: torch.nn.Module, epoch: int):
    checkpoint_path = f"{self.base_path}_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    if self.verbose:
      print(
          "EarlyStopping Validation loss improved. Saving model at:"
          f" {checkpoint_path}"
      )
