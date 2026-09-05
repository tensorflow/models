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
"""Unit tests for train_image_classifier_transfer_learning.

All heavy dependencies are mocked, so the tests neither download pretrained
weights nor read any images from disk.
"""

from unittest import mock

from absl.testing import absltest
from torch import nn
import vit_training as train_script


def _make_fake_os(available_cpu_count=None, has_affinity=True):
  """Returns a stand-in for the ``os`` module with a fixed CPU count.

  Args:
    available_cpu_count: Value reported by either ``sched_getaffinity`` or
      ``cpu_count``. Pass None to simulate ``cpu_count`` failing.
    has_affinity: Whether the fake module exposes ``sched_getaffinity``. Set to
      False to exercise the Windows and macOS fallback path.

  Returns:
    A mock whose ``spec`` limits it to the attributes a real platform
    would expose, so ``hasattr`` behaves the way the code under test
    expects.
  """
  if has_affinity:
    fake_os = mock.Mock(spec=["sched_getaffinity", "cpu_count"])
    fake_os.sched_getaffinity.return_value = set(range(available_cpu_count))
  else:
    fake_os = mock.Mock(spec=["cpu_count"])

  fake_os.cpu_count.return_value = available_cpu_count
  return fake_os


class DetectNumberOfDataloaderWorkersTest(absltest.TestCase):
  """Tests for detect_number_of_dataloader_workers."""

  def test_uses_affinity_count_when_below_maximum(self):
    fake_os = _make_fake_os(available_cpu_count=4)

    with mock.patch.object(train_script, "os", fake_os):
      worker_count = train_script.detect_number_of_dataloader_workers()

    self.assertEqual(worker_count, 4)

  def test_caps_worker_count_at_maximum(self):
    fake_os = _make_fake_os(available_cpu_count=64)

    with mock.patch.object(train_script, "os", fake_os):
      worker_count = train_script.detect_number_of_dataloader_workers()

    self.assertEqual(worker_count, train_script.MAXIMUM_NUMBER_OF_WORKERS)

  def test_falls_back_to_cpu_count_without_affinity_support(self):
    fake_os = _make_fake_os(available_cpu_count=2, has_affinity=False)

    with mock.patch.object(train_script, "os", fake_os):
      worker_count = train_script.detect_number_of_dataloader_workers()

    self.assertEqual(worker_count, 2)
    fake_os.cpu_count.assert_called_once()

  def test_returns_one_when_cpu_count_is_unavailable(self):
    fake_os = _make_fake_os(available_cpu_count=None, has_affinity=False)

    with mock.patch.object(train_script, "os", fake_os):
      worker_count = train_script.detect_number_of_dataloader_workers()

    self.assertEqual(worker_count, 1)


class SetRandomSeedsTest(absltest.TestCase):
  """Tests for set_random_seeds."""

  def test_seeds_both_the_cpu_and_cuda_generators(self):
    with mock.patch.object(train_script.torch, "manual_seed") as mock_cpu_seed:
      with mock.patch.object(
          train_script.torch.cuda, "manual_seed"
      ) as mock_cuda_seed:
        train_script.set_random_seeds(seed=123)

    mock_cpu_seed.assert_called_once_with(123)
    mock_cuda_seed.assert_called_once_with(123)


class PlotLossCurvesTest(absltest.TestCase):
  """Tests for plot_loss_curves."""

  def test_plots_one_line_per_metric_over_the_completed_epochs(self):
    results = {
        "train_loss": [0.9, 0.5, 0.3],
        "test_loss": [1.0, 0.6, 0.4],
        "train_acc": [0.4, 0.7, 0.8],
        "test_acc": [0.3, 0.6, 0.75],
    }

    with mock.patch.object(train_script.plt, "figure"):
      with mock.patch.object(train_script.plt, "subplot"):
        with mock.patch.object(train_script.plt, "legend"):
          with mock.patch.object(train_script.plt, "plot") as mock_plot:
            train_script.plot_loss_curves(results)

    self.assertEqual(mock_plot.call_count, 4)
    for plot_call in mock_plot.call_args_list:
      plotted_epochs, plotted_values = plot_call.args
      self.assertEqual(list(plotted_epochs), [0, 1, 2])
      self.assertLen(plotted_values, 3)


class CreateDataloadersTest(absltest.TestCase):
  """Tests for create_dataloaders."""

  def setUp(self):
    super().setUp()
    self.train_dataset = mock.MagicMock()
    self.train_dataset.classes = ["milk", "others"]
    self.validation_dataset = mock.MagicMock()

    image_folder_patcher = mock.patch.object(
        train_script.datasets, "ImageFolder"
    )
    self.mock_image_folder = image_folder_patcher.start()
    self.addCleanup(image_folder_patcher.stop)
    self.mock_image_folder.side_effect = [
        self.train_dataset,
        self.validation_dataset,
    ]

    dataloader_patcher = mock.patch.object(
        train_script.torch_data, "DataLoader"
    )
    self.mock_dataloader = dataloader_patcher.start()
    self.addCleanup(dataloader_patcher.stop)

    self.transform = mock.MagicMock()

  def _call_create_dataloaders(self):
    """Calls the function under test with a fixed set of arguments."""
    return train_script.create_dataloaders(
        train_directory="/data/train",
        validation_directory="/data/val",
        transform=self.transform,
        batch_size=32,
        number_of_workers=2,
    )

  def test_builds_image_folders_from_both_split_directories(self):
    self._call_create_dataloaders()

    self.mock_image_folder.assert_has_calls([
        mock.call("/data/train", transform=self.transform),
        mock.call("/data/val", transform=self.transform),
    ])

  def test_shuffles_training_data_but_not_validation_data(self):
    self._call_create_dataloaders()

    train_call, validation_call = self.mock_dataloader.call_args_list
    self.assertTrue(train_call.kwargs["shuffle"])
    self.assertFalse(validation_call.kwargs["shuffle"])

  def test_passes_batch_size_and_worker_count_to_both_dataloaders(self):
    self._call_create_dataloaders()

    for dataloader_call in self.mock_dataloader.call_args_list:
      self.assertEqual(dataloader_call.kwargs["batch_size"], 32)
      self.assertEqual(dataloader_call.kwargs["num_workers"], 2)

  def test_returns_class_names_from_the_training_split(self):
    _, _, class_names = self._call_create_dataloaders()

    self.assertEqual(class_names, ["milk", "others"])


class BuildPretrainedVitClassifierTest(absltest.TestCase):
  """Tests for build_pretrained_vit_classifier."""

  def setUp(self):
    super().setUp()
    self.fake_parameters = [
        mock.MagicMock(requires_grad=True) for _ in range(3)
    ]
    self.fake_model = mock.MagicMock()
    # The function chains .to(device) onto the constructed model, so the
    # mock has to return itself to stay the object under inspection.
    self.fake_model.to.return_value = self.fake_model
    self.fake_model.parameters.return_value = self.fake_parameters

    vit_patcher = mock.patch.object(train_script.torchvision.models, "vit_b_16")
    self.mock_vit_b_16 = vit_patcher.start()
    self.addCleanup(vit_patcher.stop)
    self.mock_vit_b_16.return_value = self.fake_model

    set_seeds_patcher = mock.patch.object(train_script, "set_random_seeds")
    self.mock_set_random_seeds = set_seeds_patcher.start()
    self.addCleanup(set_seeds_patcher.stop)

  def test_freezes_every_backbone_parameter(self):
    train_script.build_pretrained_vit_classifier(
        number_of_classes=2, device="cpu"
    )

    for parameter in self.fake_parameters:
      self.assertFalse(parameter.requires_grad)

  def test_replaces_head_with_linear_layer_sized_to_the_dataset(self):
    model, _ = train_script.build_pretrained_vit_classifier(
        number_of_classes=7, device="cpu"
    )

    self.assertIsInstance(model.heads, nn.Linear)
    self.assertEqual(model.heads.in_features, 768)
    self.assertEqual(model.heads.out_features, 7)

  def test_seeds_before_creating_the_new_head(self):
    train_script.build_pretrained_vit_classifier(
        number_of_classes=2, device="cpu"
    )

    self.mock_set_random_seeds.assert_called_once()

  def test_returns_the_preprocessing_transform(self):
    _, preprocessing_transform = train_script.build_pretrained_vit_classifier(
        number_of_classes=2, device="cpu"
    )

    self.assertTrue(callable(preprocessing_transform))


if __name__ == "__main__":
  absltest.main()
