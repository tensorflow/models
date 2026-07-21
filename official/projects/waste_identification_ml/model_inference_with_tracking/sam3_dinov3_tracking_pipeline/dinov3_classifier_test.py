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

"""Unit tests for the DINOv3 classification module and classifier."""

from unittest import mock

from absl.testing import absltest
from PIL import Image
import torch
from torch import nn

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import dinov3_classifier


class DummyBackbone(nn.Module):
  """Minimal backbone stand-in exposing the interface required by the module."""

  def __init__(self, hidden_size: int = 64):
    """Initializes the dummy backbone with a simulated norm layer.

    Args:
      hidden_size: Simulated token embedding dimensionality.
    """
    super().__init__()
    self.norm = nn.LayerNorm(hidden_size)
    self.hidden_size = hidden_size

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    """Returns dummy token embeddings of shape (batch_size, hidden_size).

    Args:
      image_batch: Batch of images.

    Returns:
      Tensor of ones with shape (batch_size, hidden_size).
    """
    batch_size = image_batch.shape[0]
    return torch.ones((batch_size, self.hidden_size), dtype=torch.float32)

  def forward_features(
      self, image_batch: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """Returns dummy CLS and patch token features.

    Args:
      image_batch: Batch of images.

    Returns:
      Mapping matching the DINOv3 feature-extraction interface.
    """
    batch_size = image_batch.shape[0]
    return {
        "x_norm_clstoken": torch.ones(
            (batch_size, self.hidden_size), dtype=torch.float32
        ),
        "x_norm_patchtokens": torch.ones(
            (batch_size, 16, self.hidden_size), dtype=torch.float32
        ),
    }


def _make_dummy_config() -> config_loader.DINOv3Config:
  """Returns a DINOv3Config populated with values safe for tests."""
  return config_loader.DINOv3Config(
      repo_dir="/dummy/repo",
      model_name="dinov3_vits14",
      checkpoint_path="/dummy/checkpoint.pt",
      inference_image_size=256,
      image_mean=(0.485, 0.456, 0.406),
      image_std=(0.229, 0.224, 0.225),
      classification_batch_size=16,
  )


def _make_dummy_state_dict(
    hidden_size: int = 32, number_of_classes: int = 3
) -> dict[str, torch.Tensor]:
  """Returns a state dict compatible with a DummyBackbone + CLS pooling head."""
  return {
      "backbone_model.norm.weight": torch.ones((hidden_size,)),
      "backbone_model.norm.bias": torch.zeros((hidden_size,)),
      "head.weight": torch.zeros((number_of_classes, hidden_size)),
      "head.bias": torch.zeros((number_of_classes,)),
  }


class ResolveDeviceTest(absltest.TestCase):
  """Tests for the module-level _resolve_device helper."""

  def test_cpu_resolves_directly(self):
    """Verifies that 'cpu' resolves to torch.device('cpu')."""
    self.assertEqual(
        dinov3_classifier._resolve_device("cpu"), torch.device("cpu")
    )

  @mock.patch.object(torch.cuda, "is_available", autospec=True)
  def test_cuda_resolves_when_available(self, mock_is_available):
    """Verifies that a CUDA device string resolves directly when available."""
    mock_is_available.return_value = True
    self.assertEqual(
        dinov3_classifier._resolve_device("cuda:0"), torch.device("cuda:0")
    )

  @mock.patch.object(torch.cuda, "is_available", autospec=True)
  def test_cuda_falls_back_to_cpu_when_unavailable(self, mock_is_available):
    """Verifies fallback to CPU when CUDA is requested but unavailable."""
    mock_is_available.return_value = False
    self.assertEqual(
        dinov3_classifier._resolve_device("cuda:0"), torch.device("cpu")
    )

  @mock.patch.object(torch.backends.mps, "is_available", autospec=True)
  def test_mps_falls_back_to_cpu_when_unavailable(self, mock_is_available):
    """Verifies fallback to CPU when MPS is requested but unavailable."""
    mock_is_available.return_value = False
    self.assertEqual(
        dinov3_classifier._resolve_device("mps"), torch.device("cpu")
    )


class InferPoolingFromStateDictTest(absltest.TestCase):
  """Tests for the module-level _infer_pooling_from_state_dict helper."""

  def test_cls_pooling_from_matching_head_shape(self):
    """Verifies CLS pooling inferred when head input equals hidden size."""
    state_dict = {"head.weight": torch.zeros((5, 64))}
    self.assertEqual(
        dinov3_classifier._infer_pooling_from_state_dict(
            state_dict, hidden_size=64
        ),
        dinov3_classifier.PoolingStrategy.CLS,
    )

  def test_cls_mean_patch_pooling_from_doubled_head_shape(self):
    """Verifies CLS_MEAN_PATCH pooling inferred when head input is doubled."""
    state_dict = {"head.weight": torch.zeros((5, 128))}
    self.assertEqual(
        dinov3_classifier._infer_pooling_from_state_dict(
            state_dict, hidden_size=64
        ),
        dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )

  def test_invalid_head_shape_raises_classifier_error(self):
    """Verifies an error is raised when head shape matches neither strategy."""
    state_dict = {"head.weight": torch.zeros((5, 100))}
    with self.assertRaisesRegex(
        dinov3_classifier.ClassifierError, "Cannot infer pooling strategy"
    ):
      dinov3_classifier._infer_pooling_from_state_dict(
          state_dict, hidden_size=64
      )


class LoadCheckpointStateDictTest(absltest.TestCase):
  """Tests for the module-level _load_checkpoint_state_dict helper."""

  @mock.patch.object(torch, "load", autospec=True)
  def test_missing_model_state_dict_raises_error(self, mock_load):
    """Verifies error when checkpoint lacks 'model_state_dict' key."""
    mock_load.return_value = {"other_key": {}}
    with self.assertRaisesRegex(
        dinov3_classifier.ClassifierError,
        "missing required key 'model_state_dict'",
    ):
      dinov3_classifier._load_checkpoint_state_dict(
          checkpoint_path="/dummy/checkpoint.pt", device=torch.device("cpu")
      )

  @mock.patch.object(torch, "load", autospec=True)
  def test_missing_head_weight_raises_error(self, mock_load):
    """Verifies error when state dict lacks 'head.weight' parameter."""
    mock_load.return_value = {
        "model_state_dict": {"other.weight": torch.zeros(1)}
    }
    with self.assertRaisesRegex(
        dinov3_classifier.ClassifierError,
        "missing required key 'head.weight'",
    ):
      dinov3_classifier._load_checkpoint_state_dict(
          checkpoint_path="/dummy/checkpoint.pt", device=torch.device("cpu")
      )

  @mock.patch.object(torch, "load", autospec=True)
  def test_valid_checkpoint_returns_state_dict(self, mock_load):
    """Verifies a valid checkpoint returns the nested state dict."""
    state_dict = _make_dummy_state_dict()
    mock_load.return_value = {"model_state_dict": state_dict}
    result = dinov3_classifier._load_checkpoint_state_dict(
        checkpoint_path="/dummy/checkpoint.pt", device=torch.device("cpu")
    )
    self.assertIs(result, state_dict)


class Dinov3ClassificationModuleTest(absltest.TestCase):
  """Tests for the Dinov3ClassificationModule PyTorch module."""

  def test_init_cls_strategy_sets_head_dimensions(self):
    """Verifies module initialization with CLS pooling strategy."""
    module = dinov3_classifier.Dinov3ClassificationModule(
        backbone_model=DummyBackbone(hidden_size=32),
        hidden_size=32,
        number_of_classes=5,
        pooling=dinov3_classifier.PoolingStrategy.CLS,
    )
    self.assertEqual(module.head.in_features, 32)
    self.assertEqual(module.head.out_features, 5)

  def test_init_cls_mean_patch_strategy_doubles_head_input(self):
    """Verifies module initialization with CLS_MEAN_PATCH pooling strategy."""
    module = dinov3_classifier.Dinov3ClassificationModule(
        backbone_model=DummyBackbone(hidden_size=32),
        hidden_size=32,
        number_of_classes=5,
        pooling=dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )
    self.assertEqual(module.head.in_features, 64)
    self.assertEqual(module.head.out_features, 5)

  def test_extract_features_cls_strategy_shape(self):
    """Verifies extract_features returns backbone forward output under CLS."""
    module = dinov3_classifier.Dinov3ClassificationModule(
        backbone_model=DummyBackbone(hidden_size=32),
        hidden_size=32,
        number_of_classes=4,
        pooling=dinov3_classifier.PoolingStrategy.CLS,
    )
    features = module.extract_features(torch.zeros((2, 3, 256, 256)))
    self.assertEqual(features.shape, (2, 32))

  def test_extract_features_cls_mean_patch_strategy_shape(self):
    """Verifies extract_features concatenates CLS and mean patch tokens."""
    module = dinov3_classifier.Dinov3ClassificationModule(
        backbone_model=DummyBackbone(hidden_size=32),
        hidden_size=32,
        number_of_classes=4,
        pooling=dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )
    features = module.extract_features(torch.zeros((2, 3, 256, 256)))
    self.assertEqual(features.shape, (2, 64))

  def test_forward_pass_produces_expected_logit_shape(self):
    """Verifies the full forward pass produces per-class logits."""
    module = dinov3_classifier.Dinov3ClassificationModule(
        backbone_model=DummyBackbone(hidden_size=32),
        hidden_size=32,
        number_of_classes=3,
        pooling=dinov3_classifier.PoolingStrategy.CLS,
    )
    logits = module(torch.zeros((2, 3, 256, 256)))
    self.assertEqual(logits.shape, (2, 3))


class DINOv3ClassifierTest(absltest.TestCase):
  """Tests for the DINOv3Classifier public API."""

  def setUp(self):
    """Initializes common test fixtures."""
    super().setUp()
    self.class_names = ["class_a", "class_b", "class_c"]
    self.image_transform = dinov3_classifier._build_image_transform(
        _make_dummy_config()
    )

  def _make_classifier_with_mock_model(
      self, mock_model: mock.MagicMock
  ) -> dinov3_classifier.DINOv3Classifier:
    """Constructs a DINOv3Classifier wrapping the given mock model."""
    return dinov3_classifier.DINOv3Classifier(
        model=mock_model,
        class_names=self.class_names,
        image_transform=self.image_transform,
        device=torch.device("cpu"),
    )

  def test_predict_batch_empty_input_returns_empty_list(self):
    """Verifies an empty input returns an empty list without inference."""
    mock_model = mock.MagicMock()
    classifier = self._make_classifier_with_mock_model(mock_model)
    self.assertEmpty(classifier.predict_batch([]))
    mock_model.assert_not_called()

  def test_predict_batch_returns_top_class_and_percentages(self):
    """Verifies predictions expose top class and percentage probabilities."""
    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([
        [2.0, 0.0, 0.0],  # class_a is the top class.
        [0.0, 3.0, 0.0],  # class_b is the top class.
    ])
    classifier = self._make_classifier_with_mock_model(mock_model)

    predictions = classifier.predict_batch([
        Image.new("RGB", (256, 256), color="red"),
        Image.new("RGB", (256, 256), color="blue"),
    ])

    self.assertLen(predictions, 2)
    self.assertEqual(predictions[0]["predicted_class"], "class_a")
    self.assertEqual(predictions[1]["predicted_class"], "class_b")
    self.assertIn("class_c", predictions[0]["all_probabilities_percent"])
    self.assertGreater(
        predictions[0]["predicted_probability_percent"],
        predictions[0]["all_probabilities_percent"]["class_b"],
    )

  def test_from_config_builds_classifier_successfully(self):
    """Verifies from_config wires the checkpoint into a ready-to-use model."""
    mock_hub_load = self.enter_context(
        mock.patch.object(torch.hub, "load", autospec=True)
    )
    mock_torch_load = self.enter_context(
        mock.patch.object(torch, "load", autospec=True)
    )
    mock_hub_load.return_value = DummyBackbone(hidden_size=32)
    mock_torch_load.return_value = {
        "model_state_dict": _make_dummy_state_dict(
            hidden_size=32, number_of_classes=3
        )
    }

    classifier = dinov3_classifier.DINOv3Classifier.from_config(
        config=_make_dummy_config(),
        class_names=self.class_names,
        device="cpu",
    )

    self.assertIsInstance(
        classifier._model, dinov3_classifier.Dinov3ClassificationModule
    )
    self.assertFalse(classifier._model.training)

  def test_from_config_wraps_state_dict_load_errors(self):
    """Verifies load_state_dict RuntimeError is wrapped as ClassifierError."""
    mock_hub_load = self.enter_context(
        mock.patch.object(torch.hub, "load", autospec=True)
    )
    mock_torch_load = self.enter_context(
        mock.patch.object(torch, "load", autospec=True)
    )
    mock_hub_load.return_value = DummyBackbone(hidden_size=32)
    # head.bias shape mismatch triggers a RuntimeError in load_state_dict.
    incompatible_state_dict = {
        "backbone_model.norm.weight": torch.ones((32,)),
        "backbone_model.norm.bias": torch.zeros((32,)),
        "head.weight": torch.zeros((3, 32)),
        "head.bias": torch.zeros((999,)),
    }
    mock_torch_load.return_value = {"model_state_dict": incompatible_state_dict}

    with self.assertRaisesRegex(
        dinov3_classifier.ClassifierError,
        "Failed to load state dict into model",
    ):
      dinov3_classifier.DINOv3Classifier.from_config(
          config=_make_dummy_config(),
          class_names=self.class_names,
          device="cpu",
      )


if __name__ == "__main__":
  absltest.main()
