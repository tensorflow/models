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

"""Tests for pet_grade_classifier module."""

import unittest
from unittest import mock

from absl.testing import parameterized
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from official.projects.waste_identification_ml.Deploy.pet_grading_cloud_deployment import pet_grade_classifier


MODULE_PATH = pet_grade_classifier.__name__


class MockBackbone(nn.Module):

  def __init__(self, hidden_size: int = 768):
    super().__init__()
    self.norm = nn.LayerNorm(hidden_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((x.shape[0], self.norm.normalized_shape[0]))

  def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_size = self.norm.normalized_shape[0]
    return {
        "x_norm_clstoken": torch.zeros((x.shape[0], hidden_size)),
        "x_norm_patchtokens": torch.zeros((x.shape[0], 10, hidden_size)),
    }


class PetGradeClassifierTest(parameterized.TestCase):

  @mock.patch(f"{MODULE_PATH}.torch.hub.load")
  def test_load_backbone(self, mock_hub_load):
    # Act
    pet_grade_classifier._load_backbone("mock_repo", "mock_model_name")

    # Assert
    mock_hub_load.assert_called_once_with(
        "mock_repo",
        "mock_model_name",
        source="local",
        pretrained=False,
    )

  @mock.patch(f"{MODULE_PATH}._load_backbone")
  def test_dinov3_classification_init_unsupported_pooling(self):
    with self.assertRaises(ValueError) as context:
      _ = pet_grade_classifier.Dinov3Classification(
          dinov3_repo_dir="mock_repo",
          model_name="mock_model",
          num_classes=3,
          pooling="unsupported_pooling",
      )

    self.assertIn("Unsupported pooling strategy", str(context.exception))

  @mock.patch(f"{MODULE_PATH}._load_backbone")
  def test_dinov3_classification_forward_pooling_cls(self, mock_load):
    # Arrange
    mock_load.return_value = MockBackbone(hidden_size=256)
    model = pet_grade_classifier.Dinov3Classification(
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        num_classes=3,
        pooling="cls",
    )
    x = torch.zeros((2, 3, 224, 224))

    # Act
    out = model(x)

    # Assert
    self.assertEqual(out.shape, (2, 3))

  @mock.patch(f"{MODULE_PATH}._load_backbone")
  def test_dinov3_classification_forward_pooling_cls_mean_patch(
      self, mock_load
  ):
    # Arrange
    mock_load.return_value = MockBackbone(hidden_size=256)
    model = pet_grade_classifier.Dinov3Classification(
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        num_classes=3,
        pooling="cls_mean_patch",
    )
    # Head input is 2x hidden = 512
    self.assertEqual(model.head.in_features, 512)
    x = torch.zeros((2, 3, 224, 224))

    # Act
    out = model(x)

    # Assert
    self.assertEqual(out.shape, (2, 3))

  @mock.patch(f"{MODULE_PATH}.torch.load")
  def test_dino_classifier_read_state_dict_missing_model_state_dict(
      self, mock_torch_load
  ):
    mock_torch_load.return_value = {}

    with self.assertRaises(KeyError) as context:
      pet_grade_classifier.DinoClassifier._read_state_dict(
          "mock_ckpt", torch.device("cpu")
      )

    self.assertIn("missing 'model_state_dict'", str(context.exception))

  @mock.patch(f"{MODULE_PATH}.torch.load")
  def test_dino_classifier_read_state_dict_missing_head_weight(
      self, mock_torch_load
  ):
    mock_torch_load.return_value = {"model_state_dict": {}}

    with self.assertRaises(KeyError) as context:
      pet_grade_classifier.DinoClassifier._read_state_dict(
          "mock_ckpt", torch.device("cpu")
      )

    self.assertIn("missing 'head.weight'", str(context.exception))

  @parameterized.named_parameters(
      ("cls", torch.zeros((3, 256)), "cls"),
      ("cls_mean_patch", torch.zeros((3, 512)), "cls_mean_patch"),
  )
  @mock.patch(f"{MODULE_PATH}._load_backbone")
  def test_detect_pooling(self, head_weight, expected_pooling, mock_load):
    mock_load.return_value = MockBackbone(hidden_size=256)
    state_dict = {"head.weight": head_weight}

    pooling = pet_grade_classifier.DinoClassifier._detect_pooling(
        state_dict, "mock_repo", "mock_model"
    )

    self.assertEqual(pooling, expected_pooling)

  @mock.patch(f"{MODULE_PATH}._load_backbone")
  def test_detect_pooling_value_error(self, mock_load):
    mock_load.return_value = MockBackbone(hidden_size=256)
    # Target head dim 400 doesn't match 256 or 512
    state_dict = {"head.weight": torch.zeros((3, 400))}

    with self.assertRaises(ValueError) as context:
      pet_grade_classifier.DinoClassifier._detect_pooling(
          state_dict, "mock_repo", "mock_model"
      )

    self.assertIn("Cannot infer pooling strategy", str(context.exception))

  @mock.patch(f"{MODULE_PATH}.Dinov3Classification.load_state_dict")
  @mock.patch(f"{MODULE_PATH}._load_backbone")
  @mock.patch(f"{MODULE_PATH}.DinoClassifier._read_state_dict")
  def test_dino_classifier_init_success(
      self, mock_read_state, mock_load, mock_load_state_dict
  ):
    # Arrange
    state_dict = {"head.weight": torch.zeros((3, 256))}
    mock_read_state.return_value = state_dict
    mock_load.return_value = MockBackbone(hidden_size=256)

    # Act
    classifier = pet_grade_classifier.DinoClassifier(
        classifier_checkpoint_path="mock_ckpt.pth",
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        class_names=["class_a", "class_b", "class_c"],
        device="cpu",
    )

    # Assert
    self.assertEqual(classifier.pooling, "cls")
    self.assertEqual(classifier.class_names, ["class_a", "class_b", "class_c"])
    mock_load_state_dict.assert_called_once_with(state_dict)

  @mock.patch(f"{MODULE_PATH}.Dinov3Classification.load_state_dict")
  @mock.patch(f"{MODULE_PATH}._load_backbone")
  @mock.patch(f"{MODULE_PATH}.DinoClassifier._read_state_dict")
  def test_predict_empty_input(self, mock_read_state, mock_load):
    # Arrange
    state_dict = {"head.weight": torch.zeros((3, 256))}
    mock_read_state.return_value = state_dict
    mock_load.return_value = MockBackbone(hidden_size=256)

    classifier = pet_grade_classifier.DinoClassifier(
        classifier_checkpoint_path="mock_ckpt.pth",
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        class_names=["class_a", "class_b", "class_c"],
        device="cpu",
    )

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      classifier.predict([])

    self.assertIn("predict requires at least one image", str(context.exception))

  @mock.patch(f"{MODULE_PATH}.Dinov3Classification.load_state_dict")
  @mock.patch(f"{MODULE_PATH}._load_backbone")
  @mock.patch(f"{MODULE_PATH}.DinoClassifier._read_state_dict")
  def test_predict_success(self, mock_read_state, mock_load):
    # Arrange
    state_dict = {"head.weight": torch.zeros((2, 256))}
    mock_read_state.return_value = state_dict
    mock_load.return_value = MockBackbone(hidden_size=256)

    classifier = pet_grade_classifier.DinoClassifier(
        classifier_checkpoint_path="mock_ckpt.pth",
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        class_names=["class_a", "class_b"],
        device="cpu",
    )

    # Mock model call: must return logits shape [num_images, num_classes]
    # For image 1: logits [10.0, 0.0] -> class_a is highly likely.
    # For image 2: logits [0.0, 10.0] -> class_b is highly likely.
    mock_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    classifier.model = mock.MagicMock(return_value=mock_logits)

    image1 = Image.new("RGB", (100, 100))
    image2 = Image.new("RGB", (100, 100))

    # Act
    predictions = classifier.predict([image1, image2])

    # Assert
    self.assertLen(predictions, 2)

    # Image 1 prediction
    self.assertEqual(predictions[0]["predicted_class"], "class_a")
    self.assertGreater(predictions[0]["predicted_probability"], 99.0)
    self.assertIn("class_a", predictions[0]["all_probabilities"])
    self.assertIn("class_b", predictions[0]["all_probabilities"])

    # Image 2 prediction
    self.assertEqual(predictions[1]["predicted_class"], "class_b")
    self.assertGreater(predictions[1]["predicted_probability"], 99.0)

  def test_build_eval_transform(self):
    transform = pet_grade_classifier.DinoClassifier._build_eval_transform(224)
    self.assertIsInstance(transform, transforms.Compose)

    # Test on a dummy PIL Image
    img = Image.new("RGB", (100, 100))
    tensor = transform(img)
    self.assertEqual(tensor.shape, (3, 224, 224))

  @parameterized.named_parameters(
      ("cpu", False, "cpu"),
      ("cuda", True, "cuda"),
  )
  @mock.patch(f"{MODULE_PATH}.torch.cuda.is_available")
  @mock.patch(f"{MODULE_PATH}.Dinov3Classification.load_state_dict")
  @mock.patch(f"{MODULE_PATH}._load_backbone")
  @mock.patch(f"{MODULE_PATH}.DinoClassifier._read_state_dict")
  @mock.patch(f"{MODULE_PATH}.Dinov3Classification.to")
  def test_dino_classifier_device_default(
      self,
      cuda_available,
      expected_device,
      mock_to,
      mock_read_state,
      mock_load,
      mock_cuda_available,
  ):
    mock_cuda_available.return_value = cuda_available
    state_dict = {"head.weight": torch.zeros((3, 256))}
    mock_read_state.return_value = state_dict
    mock_load.return_value = MockBackbone(hidden_size=256)
    mock_to.side_effect = lambda dev: mock.MagicMock()

    classifier = pet_grade_classifier.DinoClassifier(
        classifier_checkpoint_path="mock_ckpt.pth",
        dinov3_repo_dir="mock_repo",
        model_name="mock_model",
        class_names=["class_a", "class_b"],
    )
    self.assertEqual(classifier.device, expected_device)


if __name__ == "__main__":
  unittest.main()
