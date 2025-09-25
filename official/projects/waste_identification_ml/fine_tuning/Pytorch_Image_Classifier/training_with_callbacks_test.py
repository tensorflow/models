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

import unittest
from unittest import mock
import torch
from torch.utils import data
from official.projects.waste_identification_ml.fine_tuning.Pytorch_Image_Classifier import training_with_callbacks

Mock = mock.Mock
MagicMock = mock.MagicMock
TensorDataset = data.TensorDataset
DataLoader = data.DataLoader


class SimpleModel(torch.nn.Module):
  """A simple model for testing purposes."""

  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = torch.nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.linear(x)


class TestTrainingEngine(unittest.TestCase):
  """Test suite for the training engine functions."""

  def setUp(self):
    """Set up common resources for tests."""
    super().setUp()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.batch_size = 4
    self.num_batches = 10
    self.input_size = 5
    self.output_size = 2

    x = torch.randn(self.num_batches * self.batch_size, self.input_size)
    y = torch.randint(
        0, self.output_size, (self.num_batches * self.batch_size,)
    )
    self.dataset = TensorDataset(x, y)

    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

    self.model = SimpleModel(self.input_size, self.output_size).to(self.device)
    self.loss_fn = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

  def test_train_step(self):
    """Test the train_step function."""
    # Use real objects for a simple integration test of the step
    train_loss, train_acc = training_with_callbacks.train_step(
        model=self.model,
        dataloader=self.dataloader,
        loss_fn=self.loss_fn,
        optimizer=self.optimizer,
        device=self.device,
    )

    self.assertIsInstance(train_loss, float)
    self.assertIsInstance(train_acc, float)

    self.assertGreaterEqual(train_loss, 0.0)
    self.assertGreaterEqual(train_acc, 0.0)
    self.assertLessEqual(train_acc, 1.0)

  def test_test_step(self):
    """Test the test_step function."""
    # Use real objects for a simple integration test of the step
    test_loss, test_acc = training_with_callbacks.test_step(
        model=self.model,
        dataloader=self.dataloader,
        loss_fn=self.loss_fn,
        device=self.device,
    )

    self.assertIsInstance(test_loss, float)
    self.assertIsInstance(test_acc, float)

    self.assertGreaterEqual(test_loss, 0.0)
    self.assertGreaterEqual(test_acc, 0.0)
    self.assertLessEqual(test_acc, 1.0)

  def test_train_function(self):
    """Test the main train function."""
    epochs = 3
    results = training_with_callbacks.train(
        model=self.model,
        train_dataloader=self.dataloader,
        test_dataloader=self.dataloader,
        optimizer=self.optimizer,
        loss_fn=self.loss_fn,
        epochs=epochs,
        device=self.device,
    )

    self.assertIsInstance(results, dict)

    expected_keys = ["train_loss", "train_acc", "test_loss", "test_acc"]
    for key in expected_keys:
      self.assertIn(key, results)
      self.assertIsInstance(results[key], list)
      self.assertEqual(len(results[key]), epochs)

  def test_train_function_with_early_stopping(self):
    """Test the train function with early stopping mock."""
    epochs = 10  # Set more epochs than stopping patience

    mock_early_stopping = Mock()
    mock_early_stopping.stop_training = False

    def check_side_effect(**kwargs):
      epoch = kwargs.get("epoch", 0)
      if epoch == 2:
        mock_early_stopping.stop_training = True

    mock_early_stopping.check.side_effect = check_side_effect

    _ = training_with_callbacks.train(
        model=self.model,
        train_dataloader=self.dataloader,
        test_dataloader=self.dataloader,
        optimizer=self.optimizer,
        loss_fn=self.loss_fn,
        epochs=epochs,
        device=self.device,
        early_stopping=mock_early_stopping,
    )

    self.assertEqual(mock_early_stopping.check.call_count, 3)

  def test_train_step_calculates_loss_and_accuracy_correctly(self):
    model = torch.nn.Linear(2, 2)
    # Set model weights for deterministic output.
    with torch.no_grad():
      model.weight.data = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
      model.bias.data = torch.tensor([1.0, 0.0])

    # model(x) = [x[0]+x[1]+1, 0]
    # For input [1,1], output is [3,0]. With label 0, this is "correct".
    # For input [0,0], output is [1,0]. With label 1, this is "incorrect".
    # Prediction for both will be class 0.
    inputs = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1])
    dataset = data.TensorDataset(inputs, labels)
    dataloader = data.DataLoader(dataset, batch_size=2)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Expected loss:
    # y_pred = [[3.0, 0.0], [1.0, 0.0]]
    # loss = CrossEntropyLoss(y_pred, [0, 1]) ~= 0.681
    expected_loss = 0.681

    # Expected accuracy:
    # y_pred_class = argmax(y_pred, dim=1) = [0, 0]
    # labels = [0, 1]
    # accuracy = (([0,0] == [0,1]).sum()) / 2 = 1 / 2 = 0.5
    expected_acc = 0.5

    train_loss, train_acc = training_with_callbacks.train_step(
        model, dataloader, loss_fn, optimizer, device
    )

    self.assertAlmostEqual(train_loss, expected_loss, places=3)
    self.assertAlmostEqual(train_acc, expected_acc, places=5)


if __name__ == "__main__":
  unittest.main()
