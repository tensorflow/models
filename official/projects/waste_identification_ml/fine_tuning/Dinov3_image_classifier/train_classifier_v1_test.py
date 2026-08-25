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

"""Unit tests for training.py."""

import contextlib
import logging
import pathlib
import random
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data as torch_data

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import train_classifier_v1 as training


class _TinyClassifier(nn.Module):
  """Minimal classifier with both a frozen and a trainable submodule.

  Used to verify that `collect_trainable_parameters` returns only the
  parameters whose `requires_grad` flag is True.
  """

  def __init__(self, number_of_classes: int = 3):
    super().__init__()
    self.backbone_model = nn.Linear(in_features=8, out_features=16)
    self.head = nn.Linear(in_features=16, out_features=number_of_classes)
    for parameter in self.backbone_model.parameters():
      parameter.requires_grad = False

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    return self.head(self.backbone_model(image_batch))


def _make_random_dataset(
    number_of_samples: int, number_of_classes: int
) -> torch_data.TensorDataset:
  """Builds a small in-memory dataset of random inputs and integer labels.

  Args:
    number_of_samples: Total number of `(input, label)` pairs to generate.
    number_of_classes: Upper bound (exclusive) for randomly drawn labels.

  Returns:
    A `TensorDataset` with feature tensors of shape `(number_of_samples, 8)`
    and integer labels of shape `(number_of_samples,)`.
  """
  features = torch.randn((number_of_samples, 8))
  labels = torch.randint(
      low=0, high=number_of_classes, size=(number_of_samples,)
  )
  return torch_data.TensorDataset(features, labels)


class SeedEverythingTest(absltest.TestCase):
  """Tests for the seed_everything helper."""

  def test_seeds_produce_reproducible_python_random(self):
    """Verifies Python's `random` module is seeded."""
    training.seed_everything(123)
    first_value = random.random()
    training.seed_everything(123)
    second_value = random.random()
    self.assertEqual(first_value, second_value)

  def test_seeds_produce_reproducible_numpy_random(self):
    """Verifies NumPy's global RNG is seeded."""
    training.seed_everything(123)
    first_array = np.random.rand(4)
    training.seed_everything(123)
    second_array = np.random.rand(4)
    np.testing.assert_array_equal(first_array, second_array)

  def test_seeds_produce_reproducible_torch_random(self):
    """Verifies PyTorch's CPU RNG is seeded."""
    training.seed_everything(123)
    first_tensor = torch.rand(4)
    training.seed_everything(123)
    second_tensor = torch.rand(4)
    torch.testing.assert_close(first_tensor, second_tensor)

  def test_sets_cudnn_and_matmul_flags_for_throughput(self):
    """Verifies cuDNN benchmark/deterministic and TF32 flags are configured."""
    training.seed_everything(42)
    self.assertFalse(torch.backends.cudnn.deterministic)
    self.assertTrue(torch.backends.cudnn.benchmark)
    self.assertEqual(torch.get_float32_matmul_precision(), "high")


class CollectTrainableParametersTest(absltest.TestCase):
  """Tests for the collect_trainable_parameters helper."""

  def test_returns_only_trainable_parameters(self):
    """Verifies parameters with requires_grad=False are excluded."""
    classifier_model = _TinyClassifier()
    trainable_parameters = training.collect_trainable_parameters(
        classifier_model
    )
    # _TinyClassifier freezes the backbone (weight + bias) and leaves the
    # head trainable (weight + bias) → exactly 2 trainable tensors.
    self.assertLen(trainable_parameters, 2)
    for parameter in trainable_parameters:
      self.assertTrue(parameter.requires_grad)

  def test_returns_all_parameters_when_nothing_is_frozen(self):
    """Verifies a fully trainable model yields every parameter tensor."""
    classifier_model = nn.Linear(in_features=4, out_features=2)
    trainable_parameters = training.collect_trainable_parameters(
        classifier_model
    )
    # Linear layer has exactly two tensors: weight and bias.
    self.assertLen(trainable_parameters, 2)

  def test_returns_empty_list_when_all_frozen(self):
    """Verifies a fully frozen model yields an empty list."""
    classifier_model = nn.Linear(in_features=4, out_features=2)
    for parameter in classifier_model.parameters():
      parameter.requires_grad = False
    trainable_parameters = training.collect_trainable_parameters(
        classifier_model
    )
    self.assertEmpty(trainable_parameters)


class BuildCosineSchedulerTest(absltest.TestCase):
  """Tests for the build_cosine_scheduler helper."""

  def _make_optimizer(self, learning_rate: float) -> optim.Optimizer:
    """Creates a trivial optimizer with a single parameter group."""
    parameter = torch.nn.Parameter(torch.zeros(2))
    return optim.SGD([parameter], lr=learning_rate)

  def test_returns_cosine_annealing_scheduler(self):
    """Verifies the returned object is a CosineAnnealingLR instance."""
    optimizer = self._make_optimizer(learning_rate=1e-3)
    scheduler = training.build_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        cosine_minimum_learning_rate=1e-6,
    )
    self.assertIsInstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

  def test_configures_t_max_and_eta_min(self):
    """Verifies T_max and eta_min are forwarded correctly."""
    optimizer = self._make_optimizer(learning_rate=1e-3)
    scheduler = training.build_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=25,
        cosine_minimum_learning_rate=5e-7,
    )
    self.assertEqual(scheduler.T_max, 25)
    self.assertEqual(scheduler.eta_min, 5e-7)

  def test_starts_at_optimizer_learning_rate_and_decays(self):
    """Verifies the LR starts at the optimizer's LR and decreases over time."""
    optimizer = self._make_optimizer(learning_rate=1e-3)
    scheduler = training.build_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        cosine_minimum_learning_rate=1e-6,
    )
    initial_learning_rate = scheduler.get_last_lr()[0]
    self.assertAlmostEqual(initial_learning_rate, 1e-3)

    scheduler.step()
    self.assertLess(scheduler.get_last_lr()[0], initial_learning_rate)


class TrainOneEpochTest(parameterized.TestCase):
  """Tests for the train_one_epoch helper."""

  def setUp(self):
    super().setUp()
    # torch.autocast(device_type='cuda', ...) fails on CPU-only test hosts, so
    # patch it to a no-op context manager. The test's job is to verify the
    # loop's control flow, not the precision mode.
    self.enter_context(
        mock.patch.object(
            torch,
            "autocast",
            autospec=True,
            side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
        )
    )

  def _run_one_epoch(
      self, gradient_clip_max_norm: float = 1.0
  ) -> tuple[float, float]:
    """Runs a single training epoch on a tiny synthetic dataset."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    dataset = _make_random_dataset(number_of_samples=8, number_of_classes=3)
    train_loader = torch_data.DataLoader(dataset, batch_size=4)
    optimizer = optim.SGD(
        training.collect_trainable_parameters(classifier_model),
        lr=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    return training.train_one_epoch(
        classifier_model=classifier_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        gradient_clip_max_norm=gradient_clip_max_norm,
    )

  def test_returns_finite_loss_and_percentage_accuracy(self):
    """Verifies returned metrics are finite and accuracy is in [0, 100]."""
    epoch_loss, epoch_accuracy = self._run_one_epoch()
    self.assertTrue(np.isfinite(epoch_loss))
    self.assertGreaterEqual(epoch_accuracy, 0.0)
    self.assertLessEqual(epoch_accuracy, 100.0)

  def test_sets_model_to_train_mode(self):
    """Verifies the model is left in training mode after the epoch runs."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    classifier_model.eval()  # Start from eval to prove the switch happens.
    dataset = _make_random_dataset(number_of_samples=4, number_of_classes=3)
    train_loader = torch_data.DataLoader(dataset, batch_size=2)
    optimizer = optim.SGD(
        training.collect_trainable_parameters(classifier_model),
        lr=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    training.train_one_epoch(
        classifier_model=classifier_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        gradient_clip_max_norm=1.0,
    )
    self.assertTrue(classifier_model.training)

  def test_updates_trainable_parameters(self):
    """Verifies at least one trainable parameter changes after the epoch."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    original_head_weight = classifier_model.head.weight.detach().clone()

    dataset = _make_random_dataset(number_of_samples=8, number_of_classes=3)
    train_loader = torch_data.DataLoader(dataset, batch_size=4)
    optimizer = optim.SGD(
        training.collect_trainable_parameters(classifier_model),
        lr=1e-1,
    )
    criterion = nn.CrossEntropyLoss()

    training.train_one_epoch(
        classifier_model=classifier_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        gradient_clip_max_norm=1.0,
    )
    self.assertFalse(
        torch.equal(original_head_weight, classifier_model.head.weight)
    )

  def test_does_not_update_frozen_parameters(self):
    """Verifies frozen backbone parameters remain unchanged."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    original_backbone_weight = (
        classifier_model.backbone_model.weight.detach().clone()
    )

    dataset = _make_random_dataset(number_of_samples=8, number_of_classes=3)
    train_loader = torch_data.DataLoader(dataset, batch_size=4)
    optimizer = optim.SGD(
        training.collect_trainable_parameters(classifier_model),
        lr=1e-1,
    )
    criterion = nn.CrossEntropyLoss()

    training.train_one_epoch(
        classifier_model=classifier_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        gradient_clip_max_norm=1.0,
    )
    torch.testing.assert_close(
        original_backbone_weight, classifier_model.backbone_model.weight
    )

  def test_calls_gradient_clipping(self):
    """Verifies clip_grad_norm_ is called with the configured max norm."""
    with mock.patch.object(
        torch.nn.utils, "clip_grad_norm_", autospec=True
    ) as mock_clip:
      self._run_one_epoch(gradient_clip_max_norm=2.5)

    self.assertGreater(mock_clip.call_count, 0)
    # Every call should have received the configured max_norm.
    for call in mock_clip.call_args_list:
      self.assertEqual(call.kwargs["max_norm"], 2.5)


class ValidateTest(absltest.TestCase):
  """Tests for the validate helper."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            torch,
            "autocast",
            autospec=True,
            side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
        )
    )

  def _run_validation(self) -> tuple[float, float]:
    """Runs a single validation pass on a tiny synthetic dataset."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    dataset = _make_random_dataset(number_of_samples=8, number_of_classes=3)
    validation_loader = torch_data.DataLoader(dataset, batch_size=4)
    criterion = nn.CrossEntropyLoss()

    return training.validate(
        classifier_model=classifier_model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

  def test_returns_finite_loss_and_percentage_accuracy(self):
    """Verifies returned metrics are finite and accuracy is in [0, 100]."""
    epoch_loss, epoch_accuracy = self._run_validation()
    self.assertTrue(np.isfinite(epoch_loss))
    self.assertGreaterEqual(epoch_accuracy, 0.0)
    self.assertLessEqual(epoch_accuracy, 100.0)

  def test_sets_model_to_eval_mode(self):
    """Verifies the model is left in eval mode after validation runs."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    classifier_model.train()  # Start from train to prove the switch happens.
    dataset = _make_random_dataset(number_of_samples=4, number_of_classes=3)
    validation_loader = torch_data.DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()

    training.validate(
        classifier_model=classifier_model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    self.assertFalse(classifier_model.training)

  def test_does_not_update_parameters(self):
    """Verifies validation leaves all parameters unchanged."""
    torch.manual_seed(0)
    classifier_model = _TinyClassifier(number_of_classes=3)
    original_head_weight = classifier_model.head.weight.detach().clone()

    dataset = _make_random_dataset(number_of_samples=8, number_of_classes=3)
    validation_loader = torch_data.DataLoader(dataset, batch_size=4)
    criterion = nn.CrossEntropyLoss()

    training.validate(
        classifier_model=classifier_model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        original_head_weight, classifier_model.head.weight
    )


class ConfigureLoggingTest(absltest.TestCase):
  """Tests for the configure_logging helper."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_directory = pathlib.Path(self.temp_dir.name)
    # Snapshot and restore root-logger handlers/level so tests don't leak
    # global logging state into each other or into unrelated tests.
    root_logger = logging.getLogger()
    self._saved_handlers = list(root_logger.handlers)
    self._saved_level = root_logger.level

  def tearDown(self):
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
      handler.close()
      root_logger.removeHandler(handler)
    for handler in self._saved_handlers:
      root_logger.addHandler(handler)
    root_logger.setLevel(self._saved_level)
    self.temp_dir.cleanup()
    super().tearDown()

  def test_creates_log_file_in_output_directory(self):
    """Verifies a log file is created at the expected path."""
    training.configure_logging(self.output_directory)
    log_path = self.output_directory / training.LOG_FILENAME
    self.assertTrue(log_path.exists())

  def test_attaches_console_and_file_handlers(self):
    """Verifies exactly one StreamHandler and one FileHandler are attached."""
    training.configure_logging(self.output_directory)
    handlers = logging.getLogger().handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    # StreamHandler is the base class of FileHandler, so filter it out
    # explicitly.
    console_handlers = [
        h
        for h in handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    self.assertLen(file_handlers, 1)
    self.assertLen(console_handlers, 1)

  def test_second_call_does_not_duplicate_handlers(self):
    """Verifies handlers.clear() prevents duplicate handlers on re-init."""
    training.configure_logging(self.output_directory)
    training.configure_logging(self.output_directory)
    handlers = logging.getLogger().handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    self.assertLen(file_handlers, 1)


if __name__ == "__main__":
  absltest.main()
