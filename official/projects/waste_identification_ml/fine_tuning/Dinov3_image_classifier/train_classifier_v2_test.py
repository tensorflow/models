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

"""Unit tests for train_classifier_v2.py."""

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

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import train_classifier_v2


class _TinyClassifier(nn.Module):
  """Minimal classifier mirroring `Dinov3Classification`'s public shape.

  The backbone contains a mix of parameter kinds — a 2D Linear weight, a
  1D LayerNorm weight, and biases — so that
  `build_optimizer_parameter_groups` can be exercised across every branch
  of its decay/no-decay split.
  """

  def __init__(self, number_of_classes: int = 3):
    super().__init__()
    self.backbone_model = nn.Sequential(
        nn.Linear(in_features=8, out_features=16),
        nn.LayerNorm(16),
    )
    self.head = nn.Linear(in_features=16, out_features=number_of_classes)

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
    train_classifier_v2.seed_everything(123)
    first_value = random.random()
    train_classifier_v2.seed_everything(123)
    second_value = random.random()
    self.assertEqual(first_value, second_value)

  def test_seeds_produce_reproducible_numpy_random(self):
    """Verifies NumPy's global RNG is seeded."""
    train_classifier_v2.seed_everything(123)
    first_array = np.random.rand(4)
    train_classifier_v2.seed_everything(123)
    second_array = np.random.rand(4)
    np.testing.assert_array_equal(first_array, second_array)

  def test_seeds_produce_reproducible_torch_random(self):
    """Verifies PyTorch's CPU RNG is seeded."""
    train_classifier_v2.seed_everything(123)
    first_tensor = torch.rand(4)
    train_classifier_v2.seed_everything(123)
    second_tensor = torch.rand(4)
    torch.testing.assert_close(first_tensor, second_tensor)

  def test_sets_cudnn_and_matmul_flags_for_throughput(self):
    """Verifies cuDNN benchmark/deterministic and TF32 flags are configured."""
    train_classifier_v2.seed_everything(42)
    self.assertFalse(torch.backends.cudnn.deterministic)
    self.assertTrue(torch.backends.cudnn.benchmark)
    self.assertEqual(torch.get_float32_matmul_precision(), "high")


class BuildOptimizerParameterGroupsTest(absltest.TestCase):
  """Tests for the build_optimizer_parameter_groups helper."""

  def setUp(self):
    super().setUp()
    self.classifier_model = _TinyClassifier(number_of_classes=3)

  def test_returns_four_groups_in_expected_order(self):
    """Verifies the function always returns four groups in canonical order."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    self.assertLen(parameter_groups, 4)

    # Group 0: backbone decay. Group 1: backbone no-decay.
    # Group 2: head decay.     Group 3: head no-decay.
    self.assertEqual(parameter_groups[0]["lr"], 1e-5)
    self.assertEqual(parameter_groups[0]["weight_decay"], 0.05)
    self.assertEqual(parameter_groups[1]["lr"], 1e-5)
    self.assertEqual(parameter_groups[1]["weight_decay"], 0.0)
    self.assertEqual(parameter_groups[2]["lr"], 1e-3)
    self.assertEqual(parameter_groups[2]["weight_decay"], 0.05)
    self.assertEqual(parameter_groups[3]["lr"], 1e-3)
    self.assertEqual(parameter_groups[3]["weight_decay"], 0.0)

  def test_head_weight_lands_in_head_decay_group(self):
    """Verifies the 2D head weight goes into the head decay group."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    head_decay_ids = {id(p) for p in parameter_groups[2]["params"]}
    self.assertIn(id(self.classifier_model.head.weight), head_decay_ids)

  def test_head_bias_lands_in_head_no_decay_group(self):
    """Verifies the head bias is excluded from weight decay."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    head_no_decay_ids = {id(p) for p in parameter_groups[3]["params"]}
    self.assertIn(id(self.classifier_model.head.bias), head_no_decay_ids)

  def test_backbone_linear_weight_lands_in_backbone_decay_group(self):
    """Verifies the 2D backbone Linear weight receives weight decay."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    backbone_linear_weight = self.classifier_model.backbone_model[0].weight
    backbone_decay_ids = {id(p) for p in parameter_groups[0]["params"]}
    self.assertIn(id(backbone_linear_weight), backbone_decay_ids)

  def test_backbone_layernorm_weight_lands_in_backbone_no_decay_group(self):
    """Verifies the 1D LayerNorm weight is excluded from weight decay."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    layernorm_weight = self.classifier_model.backbone_model[1].weight
    self.assertEqual(layernorm_weight.ndim, 1)
    backbone_no_decay_ids = {id(p) for p in parameter_groups[1]["params"]}
    self.assertIn(id(layernorm_weight), backbone_no_decay_ids)

  def test_backbone_linear_bias_lands_in_backbone_no_decay_group(self):
    """Verifies the backbone Linear bias is excluded from weight decay."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    backbone_linear_bias = self.classifier_model.backbone_model[0].bias
    backbone_no_decay_ids = {id(p) for p in parameter_groups[1]["params"]}
    self.assertIn(id(backbone_linear_bias), backbone_no_decay_ids)

  def test_frozen_parameters_are_excluded(self):
    """Verifies parameters with requires_grad=False are not assigned to any group."""
    # Freeze every backbone parameter.
    for parameter in self.classifier_model.backbone_model.parameters():
      parameter.requires_grad = False

    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    # Backbone groups should now be empty.
    self.assertEmpty(parameter_groups[0]["params"])
    self.assertEmpty(parameter_groups[1]["params"])
    # Head groups should still receive their parameters.
    self.assertNotEmpty(parameter_groups[2]["params"])
    self.assertNotEmpty(parameter_groups[3]["params"])

  def test_every_trainable_parameter_appears_exactly_once(self):
    """Verifies grouping is a partition (no drops, no duplicates)."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    grouped_ids: list[int] = []
    for group in parameter_groups:
      grouped_ids.extend(id(parameter) for parameter in group["params"])
    trainable_ids = [
        id(parameter)
        for parameter in self.classifier_model.parameters()
        if parameter.requires_grad
    ]
    self.assertCountEqual(grouped_ids, trainable_ids)
    # No duplicates.
    self.assertEqual(len(grouped_ids), len(set(grouped_ids)))

  def test_grouping_is_compatible_with_adamw(self):
    """Verifies AdamW accepts the produced parameter groups without error."""
    parameter_groups = train_classifier_v2.build_optimizer_parameter_groups(
        classifier_model=self.classifier_model,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )
    optimizer = optim.AdamW(parameter_groups)
    self.assertLen(optimizer.param_groups, 4)


class ComputeWarmupEpochsTest(parameterized.TestCase):
  """Tests for the compute_warmup_epochs helper."""

  @parameterized.named_parameters(
      # 10% of 40 = 4.
      ("standard_40_epochs", 40, 0.1, 4),
      # 10% of 30 = 3.
      ("standard_30_epochs", 30, 0.1, 3),
      # 25% of 20 = 5.
      ("quarter_of_20", 20, 0.25, 5),
  )
  def test_returns_rounded_fraction_of_total_epochs(
      self, total_epochs, warmup_epochs_fraction, expected_warmup_epochs
  ):
    """Verifies the helper returns round(total * fraction) in normal cases."""
    self.assertEqual(
        train_classifier_v2.compute_warmup_epochs(
            total_epochs, warmup_epochs_fraction
        ),
        expected_warmup_epochs,
    )

  def test_clamps_to_at_least_one_warmup_epoch(self):
    """Verifies zero or fractional warmup rounds up to at least 1 epoch."""
    self.assertEqual(train_classifier_v2.compute_warmup_epochs(40, 0.0), 1)
    # 5 * 0.01 = 0.05 → round to 0 → clamped to 1.
    self.assertEqual(train_classifier_v2.compute_warmup_epochs(5, 0.01), 1)

  def test_leaves_at_least_one_cosine_epoch(self):
    """Verifies warmup is clamped so at least one cosine epoch remains."""
    # A fraction of 1.0 would consume the entire schedule; clamp to n-1.
    self.assertEqual(train_classifier_v2.compute_warmup_epochs(10, 1.0), 9)
    # A fraction that rounds to more than total_epochs must still leave room.
    self.assertEqual(train_classifier_v2.compute_warmup_epochs(10, 5.0), 9)


class BuildWarmupCosineSchedulerTest(absltest.TestCase):
  """Tests for the build_warmup_cosine_scheduler helper."""

  def _make_optimizer(self, learning_rate: float) -> optim.Optimizer:
    """Creates a trivial optimizer with a single parameter group."""
    parameter = torch.nn.Parameter(torch.zeros(2))
    return optim.SGD([parameter], lr=learning_rate)

  def test_returns_sequential_lr_scheduler(self):
    """Verifies the returned object is a SequentialLR instance."""
    optimizer = self._make_optimizer(learning_rate=1e-3)
    scheduler = train_classifier_v2.build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=3,
        warmup_start_factor=0.01,
        cosine_minimum_learning_rate=1e-6,
    )
    self.assertIsInstance(scheduler, optim.lr_scheduler.SequentialLR)

  def test_first_epoch_lr_matches_warmup_start_factor(self):
    """Verifies epoch 0's LR equals warmup_start_factor * base_lr."""
    base_learning_rate = 1e-3
    warmup_start_factor = 0.01
    optimizer = self._make_optimizer(learning_rate=base_learning_rate)
    scheduler = train_classifier_v2.build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=3,
        warmup_start_factor=warmup_start_factor,
        cosine_minimum_learning_rate=1e-6,
    )
    # SequentialLR + LinearLR: at step 0, LR = start_factor * base_lr.
    initial_learning_rate = scheduler.get_last_lr()[0]
    self.assertAlmostEqual(
        initial_learning_rate, warmup_start_factor * base_learning_rate
    )

  def test_learning_rate_reaches_base_lr_at_end_of_warmup(self):
    """Verifies LR reaches base_lr after `warmup_epochs` steps."""
    base_learning_rate = 1e-3
    warmup_epochs = 3
    optimizer = self._make_optimizer(learning_rate=base_learning_rate)
    scheduler = train_classifier_v2.build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=0.01,
        cosine_minimum_learning_rate=1e-6,
    )
    # Advance to the end of warmup.
    for _ in range(warmup_epochs):
      scheduler.step()
    self.assertAlmostEqual(scheduler.get_last_lr()[0], base_learning_rate)

  def test_learning_rate_decays_after_warmup(self):
    """Verifies LR strictly decreases during the cosine phase."""
    optimizer = self._make_optimizer(learning_rate=1e-3)
    scheduler = train_classifier_v2.build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=3,
        warmup_start_factor=0.01,
        cosine_minimum_learning_rate=1e-6,
    )
    # Advance past warmup to the start of cosine.
    for _ in range(3):
      scheduler.step()
    lr_at_cosine_start = scheduler.get_last_lr()[0]

    scheduler.step()
    lr_after_one_cosine_step = scheduler.get_last_lr()[0]
    self.assertLess(lr_after_one_cosine_step, lr_at_cosine_start)


class TrainOneEpochTest(absltest.TestCase):
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
    optimizer = optim.SGD(classifier_model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    return train_classifier_v2.train_one_epoch(
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
    optimizer = optim.SGD(classifier_model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_classifier_v2.train_one_epoch(
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
    optimizer = optim.SGD(classifier_model.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss()

    train_classifier_v2.train_one_epoch(
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

  def test_calls_gradient_clipping(self):
    """Verifies clip_grad_norm_ is called with the configured max norm."""
    with mock.patch.object(
        torch.nn.utils, "clip_grad_norm_", autospec=True
    ) as mock_clip:
      self._run_one_epoch(gradient_clip_max_norm=2.5)

    self.assertGreater(mock_clip.call_count, 0)
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

    return train_classifier_v2.validate(
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

    train_classifier_v2.validate(
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

    train_classifier_v2.validate(
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
    train_classifier_v2.configure_logging(self.output_directory)
    log_path = self.output_directory / train_classifier_v2.LOG_FILENAME
    self.assertTrue(log_path.exists())

  def test_attaches_console_and_file_handlers(self):
    """Verifies exactly one StreamHandler and one FileHandler are attached."""
    train_classifier_v2.configure_logging(self.output_directory)
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
    train_classifier_v2.configure_logging(self.output_directory)
    train_classifier_v2.configure_logging(self.output_directory)
    handlers = logging.getLogger().handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    self.assertLen(file_handlers, 1)


if __name__ == "__main__":
  absltest.main()
