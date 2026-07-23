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

import pathlib
from unittest import mock

from absl.testing import absltest
from torch import nn

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import training_callbacks


def _make_model_with_head() -> nn.Module:
  """Returns a small model exposing a `.head` submodule."""
  model = nn.Sequential()
  model.add_module("backbone", nn.Linear(4, 4))
  model.head = nn.Linear(4, 2)
  return model


class SaveBestModelTest(absltest.TestCase):
  """Tests for SaveBestModel."""

  def setUp(self):
    super().setUp()
    self.mock_torch_save = self.enter_context(
        mock.patch.object(training_callbacks.torch, "save", autospec=True)
    )

  def test_saves_full_and_head_checkpoints_on_first_call(self):
    """Verifies the first call saves two checkpoints and updates best loss."""
    save_best = training_callbacks.SaveBestModel()
    save_best(
        current_validation_loss=0.5,
        epoch=0,
        model=_make_model_with_head(),
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run1",
    )
    self.assertEqual(save_best.best_validation_loss, 0.5)
    self.assertEqual(self.mock_torch_save.call_count, 2)
    saved_paths = [call.args[1] for call in self.mock_torch_save.call_args_list]
    self.assertIn("/tmp/ckpt/best_run1.pth", saved_paths)
    self.assertIn("/tmp/ckpt/best_head_run1.pth", saved_paths)

  def test_skips_saving_when_loss_did_not_improve(self):
    """Verifies no checkpoints are written when loss is not strictly lower."""
    save_best = training_callbacks.SaveBestModel(best_validation_loss=0.3)
    save_best(
        current_validation_loss=0.5,
        epoch=1,
        model=_make_model_with_head(),
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run1",
    )
    self.assertEqual(save_best.best_validation_loss, 0.3)
    self.mock_torch_save.assert_not_called()

  def test_saves_only_when_loss_strictly_improves(self):
    """Verifies equal loss does not trigger a save (strict less-than)."""
    save_best = training_callbacks.SaveBestModel(best_validation_loss=0.3)
    save_best(
        current_validation_loss=0.3,
        epoch=1,
        model=_make_model_with_head(),
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run1",
    )
    self.mock_torch_save.assert_not_called()

  def test_epoch_in_saved_checkpoint_is_one_based(self):
    """Verifies the saved 'epoch' value is the given zero-based epoch + 1."""
    save_best = training_callbacks.SaveBestModel()
    save_best(
        current_validation_loss=0.5,
        epoch=4,
        model=_make_model_with_head(),
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run1",
    )
    saved_state_dict = self.mock_torch_save.call_args_list[0].args[0]
    self.assertEqual(saved_state_dict["epoch"], 5)

  def test_raises_attribute_error_when_model_has_no_head(self):
    """Verifies saving raises AttributeError if model lacks `.head` submodule."""
    save_best = training_callbacks.SaveBestModel()
    with self.assertRaises(AttributeError):
      save_best(
          current_validation_loss=0.5,
          epoch=0,
          model=nn.Sequential(nn.Linear(4, 2)),
          output_directory=pathlib.Path("/tmp/ckpt"),
          checkpoint_name="run1",
      )


class EarlyStoppingTest(absltest.TestCase):
  """Tests for EarlyStopping."""

  def test_resets_counter_on_improvement(self):
    """Verifies the counter is reset when validation loss improves."""
    stopper = training_callbacks.EarlyStopping(patience=3)
    stopper(0.5)  # First call: improves from inf.
    stopper(0.6)  # No improvement, counter -> 1.
    self.assertEqual(stopper.counter, 1)
    stopper(0.4)  # Improvement, counter resets to 0.
    self.assertEqual(stopper.counter, 0)
    self.assertFalse(stopper.should_stop)

  def test_stops_after_patience_epochs_without_improvement(self):
    """Verifies stop is signaled after `patience` epochs without progress."""
    stopper = training_callbacks.EarlyStopping(patience=2)
    self.assertFalse(stopper(0.5))
    self.assertFalse(stopper(0.6))  # counter 1
    self.assertTrue(stopper(0.6))  # counter 2 -> stop

  def test_respects_minimum_delta_threshold(self):
    """Verifies improvements smaller than minimum_delta do not reset counter."""
    stopper = training_callbacks.EarlyStopping(patience=5, minimum_delta=0.1)
    stopper(0.5)
    # 0.45 is better than 0.5 but does not exceed the 0.1 delta threshold.
    stopper(0.45)
    self.assertEqual(stopper.counter, 1)

  def test_should_stop_stays_true_once_triggered(self):
    """Verifies should_stop stays True after triggering, even on improvement."""
    stopper = training_callbacks.EarlyStopping(patience=1)
    stopper(0.5)
    self.assertTrue(stopper(0.6))
    # Even a subsequent improvement does not reset should_stop.
    self.assertTrue(stopper(0.1))


class SaveModelTest(absltest.TestCase):
  """Tests for save_model."""

  def setUp(self):
    super().setUp()
    self.mock_torch_save = self.enter_context(
        mock.patch.object(training_callbacks.torch, "save", autospec=True)
    )

  def test_writes_full_and_head_checkpoints(self):
    """Verifies two checkpoint files are written with expected paths."""
    model = _make_model_with_head()
    optimizer = mock.create_autospec(
        training_callbacks.optim.Optimizer, instance=True
    )
    optimizer.state_dict.return_value = {"foo": "bar"}

    training_callbacks.save_model(
        epochs=10,
        model=model,
        optimizer=optimizer,
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run2",
    )

    self.assertEqual(self.mock_torch_save.call_count, 2)
    saved_paths = [call.args[1] for call in self.mock_torch_save.call_args_list]
    self.assertEqual(saved_paths[0], "/tmp/ckpt/run2.pth")
    self.assertEqual(saved_paths[1], "/tmp/ckpt/head_run2.pth")

  def test_full_checkpoint_includes_optimizer_state(self):
    """Verifies the full checkpoint carries the optimizer state dict."""
    model = _make_model_with_head()
    optimizer = mock.create_autospec(
        training_callbacks.optim.Optimizer, instance=True
    )
    optimizer.state_dict.return_value = {"lr": 1e-3}

    training_callbacks.save_model(
        epochs=3,
        model=model,
        optimizer=optimizer,
        output_directory=pathlib.Path("/tmp/ckpt"),
        checkpoint_name="run2",
    )
    full_payload = self.mock_torch_save.call_args_list[0].args[0]
    self.assertEqual(full_payload["epoch"], 3)
    self.assertIn("model_state_dict", full_payload)
    self.assertEqual(full_payload["optimizer_state_dict"], {"lr": 1e-3})

  def test_raises_attribute_error_when_model_has_no_head(self):
    """Verifies save_model raises AttributeError if model lacks `.head` submodule."""
    optimizer = mock.create_autospec(
        training_callbacks.optim.Optimizer, instance=True
    )
    with self.assertRaises(AttributeError):
      training_callbacks.save_model(
          epochs=1,
          model=nn.Sequential(nn.Linear(4, 2)),
          optimizer=optimizer,
          output_directory=pathlib.Path("/tmp/ckpt"),
          checkpoint_name="run1",
      )


class SavePlotsTest(absltest.TestCase):
  """Tests for save_plots and _save_curve_pair."""

  def test_saves_accuracy_and_loss_png_files(self):
    """Verifies two PNG paths are written to disk."""
    temp_dir = self.create_tempdir()
    output_dir = pathlib.Path(temp_dir.full_path)
    training_callbacks.save_plots(
        train_accuracy=[0.1, 0.2],
        validation_accuracy=[0.15, 0.25],
        train_loss=[0.9, 0.7],
        validation_loss=[0.85, 0.65],
        output_directory=output_dir,
    )
    self.assertTrue((output_dir / "accuracy.png").exists())
    self.assertTrue((output_dir / "loss.png").exists())


if __name__ == "__main__":
  absltest.main()
