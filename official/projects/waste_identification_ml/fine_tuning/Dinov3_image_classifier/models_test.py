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

"""Unit tests for models.py."""

import pathlib
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn

from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import models


class _DummyBackbone(nn.Module):
  """Minimal backbone exposing the interface the classifier expects."""

  def __init__(self, hidden_size: int = 32):
    super().__init__()
    self.norm = nn.LayerNorm(hidden_size)
    self._hidden_size = hidden_size

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    batch_size = image_batch.shape[0]
    return torch.ones((batch_size, self._hidden_size))

  def forward_features(
      self, image_batch: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    batch_size = image_batch.shape[0]
    return {
        "x_norm_clstoken": torch.ones((batch_size, self._hidden_size)),
        "x_norm_patchtokens": torch.ones((batch_size, 8, self._hidden_size)),
    }


class LoadModelTest(absltest.TestCase):
  """Tests for the module-level load_model helper."""

  def setUp(self):
    super().setUp()
    self.mock_hub_load = self.enter_context(
        mock.patch.object(torch.hub, "load", autospec=True)
    )
    self.fake_backbone = _DummyBackbone()
    self.mock_hub_load.return_value = self.fake_backbone

  def test_raises_when_model_name_is_empty(self):
    """Verifies an empty model_name is rejected up front."""
    with self.assertRaisesRegex(ValueError, "model_name"):
      models.load_model(model_name="", repo_dir=pathlib.Path("/tmp/dinov3"))

  def test_raises_when_repo_dir_is_empty(self):
    """Verifies an empty repo_dir is rejected up front."""
    with self.assertRaisesRegex(ValueError, "repo_dir"):
      models.load_model(model_name="dinov3_vits16", repo_dir=pathlib.Path(""))

  def test_loads_without_weights_when_none_provided(self):
    """Verifies torch.hub.load is called without a weights kwarg."""
    backbone = models.load_model(
        model_name="dinov3_vits16",
        repo_dir=pathlib.Path("/tmp/dinov3"),
    )
    self.assertIs(backbone, self.fake_backbone)
    self.mock_hub_load.assert_called_once_with(
        "/tmp/dinov3", "dinov3_vits16", source="local"
    )

  def test_loads_with_weights_when_provided(self):
    """Verifies torch.hub.load receives the weights path as a string."""
    models.load_model(
        model_name="dinov3_vits16",
        repo_dir=pathlib.Path("/tmp/dinov3"),
        weights=pathlib.Path("/tmp/weights.pth"),
    )
    self.mock_hub_load.assert_called_once_with(
        "/tmp/dinov3",
        "dinov3_vits16",
        source="local",
        weights="/tmp/weights.pth",
    )


class Dinov3ClassificationTest(absltest.TestCase):
  """Tests for the Dinov3Classification module."""

  def setUp(self):
    super().setUp()
    self.dummy_backbone = _DummyBackbone(hidden_size=32)

  def test_rejects_unsupported_pooling_strategy(self):
    """Verifies constructor rejects a pooling value outside the allowed set."""
    with self.assertRaisesRegex(ValueError, "Unsupported pooling strategy"):
      models.Dinov3Classification(
          backbone_model=self.dummy_backbone,
          pooling="not_a_real_strategy",
      )

  def test_head_input_dimension_for_cls_pooling(self):
    """Verifies head input equals hidden size for CLS pooling."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        number_of_classes=5,
        pooling=models.POOLING_CLS,
    )
    self.assertEqual(classifier.head.in_features, 32)
    self.assertEqual(classifier.head.out_features, 5)

  def test_head_input_dimension_for_cls_mean_patch_pooling(self):
    """Verifies head input equals twice hidden size for CLS_MEAN_PATCH."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        number_of_classes=5,
        pooling=models.POOLING_CLS_MEAN_PATCH,
    )
    self.assertEqual(classifier.head.in_features, 64)

  def test_backbone_frozen_when_fine_tune_false(self):
    """Verifies backbone parameters have requires_grad=False when frozen."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        fine_tune=False,
    )
    for parameter in classifier.backbone_model.parameters():
      self.assertFalse(parameter.requires_grad)
    # Head should remain trainable regardless of fine_tune.
    for parameter in classifier.head.parameters():
      self.assertTrue(parameter.requires_grad)

  def test_backbone_trainable_when_fine_tune_true(self):
    """Verifies backbone parameters stay trainable when fine_tune=True."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        fine_tune=True,
    )
    trainable_backbone_parameters = [
        parameter
        for parameter in classifier.backbone_model.parameters()
        if parameter.requires_grad
    ]
    self.assertNotEmpty(trainable_backbone_parameters)

  def test_extract_features_shape_for_cls_pooling(self):
    """Verifies CLS pooling produces a (batch, hidden_size) feature tensor."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        pooling=models.POOLING_CLS,
    )
    features = classifier.extract_features(torch.zeros((2, 3, 32, 32)))
    self.assertEqual(features.shape, (2, 32))

  def test_extract_features_shape_for_cls_mean_patch_pooling(self):
    """Verifies CLS_MEAN_PATCH pooling produces (batch, 2 * hidden_size)."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        pooling=models.POOLING_CLS_MEAN_PATCH,
    )
    features = classifier.extract_features(torch.zeros((2, 3, 32, 32)))
    self.assertEqual(features.shape, (2, 64))

  def test_forward_produces_expected_logit_shape(self):
    """Verifies forward returns (batch, number_of_classes) logits."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        number_of_classes=7,
        pooling=models.POOLING_CLS,
    )
    logits = classifier(torch.zeros((3, 3, 32, 32)))
    self.assertEqual(logits.shape, (3, 7))

  def test_from_model_name_creates_classifier(self):
    """Verifies from_model_name calls load_model and returns a classifier."""
    mock_load_model = self.enter_context(
        mock.patch.object(models, "load_model", autospec=True)
    )
    mock_load_model.return_value = self.dummy_backbone
    classifier = models.Dinov3Classification.from_model_name(
        model_name="dinov3_vits16",
        repo_dir=pathlib.Path("/tmp/dinov3"),
        number_of_classes=3,
        pooling=models.POOLING_CLS,
    )
    mock_load_model.assert_called_once_with(
        model_name="dinov3_vits16",
        repo_dir=pathlib.Path("/tmp/dinov3"),
        weights=None,
    )
    self.assertIsInstance(classifier, models.Dinov3Classification)
    self.assertIs(classifier.backbone_model, self.dummy_backbone)
    self.assertEqual(classifier.head.out_features, 3)

  def test_classification_head_has_bias(self):
    """Verifies that the classification head has a bias parameter."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
        number_of_classes=3,
    )
    self.assertIsNotNone(classifier.head.bias)
    self.assertEqual(classifier.head.bias.shape, (3,))
    self.assertTrue(classifier.head.bias.requires_grad)

  def test_default_number_of_classes(self):
    """Verifies default number of classes equals DEFAULT_NUMBER_OF_CLASSES."""
    classifier = models.Dinov3Classification(
        backbone_model=self.dummy_backbone,
    )
    self.assertEqual(
        classifier.head.out_features, models.DEFAULT_NUMBER_OF_CLASSES
    )


if __name__ == "__main__":
  absltest.main()
