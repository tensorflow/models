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

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from official.projects.waste_identification_ml.fine_tuning.Pytorch_Image_Classifier import inference_utils

FEATURE_DIM = 768


class InferenceUtilsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.image_path = os.path.join(self.temp_dir.name, "random_image.png")

    random_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(random_array)
    image.save(self.image_path)

    self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
    ])

    # Create temp model file (from original code).
    self.num_classes = 10  # Use 10 classes for this test.
    self.device = torch.device("cpu")  # Always use CPU for unit tests.

    # Create a dummy model and save its state_dict to a temp file.
    dummy_model = torchvision.models.vit_b_16(weights=None)
    dummy_model.heads = torch.nn.Linear(
        in_features=FEATURE_DIM, out_features=self.num_classes
    )

    # Create a named temporary file.
    # We use delete=False so we can close it, save to it, and then.
    # manually delete it in tearDown.
    self.temp_model_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pth"
    )
    self.model_path = self.temp_model_file.name

    # Save the state_dict and close the file.
    torch.save(dummy_model.state_dict(), self.model_path)
    self.temp_model_file.close()

  def tearDown(self):
    """Clean up the temporary directory and model file."""
    # --- Clean up temp directory (which holds the image) ---
    self.temp_dir.cleanup()

    # --- Clean up temp model file (from original code) ---
    if os.path.exists(self.model_path):
      os.remove(self.model_path)

    super().tearDown()  # Call base class teardown

  def test_plot_prediction_runs_without_error(self):
    pred_class = "random"
    pred_prob = 88.88

    # Prevent actual plot display during test.
    with unittest.mock.patch("matplotlib.pyplot.show"):
      inference_utils.plot_prediction(self.image_path, pred_class, pred_prob)

  def test_get_prediction_details(self):
    logits = torch.tensor([[1.0, 2.0, 0.5]])  # Shape: (1, 3)
    class_names = ["cat", "dog", "bird"]

    pred_class, pred_prob = inference_utils.get_prediction_details(
        logits, class_names
    )

    self.assertEqual(pred_class, "dog")
    self.assertIsInstance(pred_prob, float)
    self.assertGreaterEqual(pred_prob, 0.0)
    self.assertLessEqual(pred_prob, 1.0)

  def test_process_image_output(self):
    result = inference_utils.process_image(self.image_path, self.transform)

    self.assertIsInstance(result, torch.Tensor)
    self.assertEqual(result.dim(), 4)
    self.assertEqual(result.shape[0], 1)  # batch size
    self.assertEqual(result.shape[2:], torch.Size([64, 64]))

  def test_transform_output(self):
    dummy_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_array)

    transform = inference_utils.get_default_transform((224, 224))
    output = transform(dummy_image)
    self.assertIsInstance(output, torch.Tensor)
    self.assertEqual(output.shape, (3, 224, 224))
    self.assertTrue(torch.all(output < 3.0) and torch.all(output > -3.0))

  @unittest.mock.patch("torch.load")
  def test_feature_dim_and_freezing(self, mock_torch_load):
    num_classes = 5
    device = torch.device("cpu")

    # Create a dummy model to generate a valid state_dict.
    dummy_model = torchvision.models.vit_b_16(weights=None)
    dummy_model.heads = torch.nn.Linear(
        in_features=FEATURE_DIM, out_features=num_classes
    )
    mock_torch_load.return_value = dummy_model.state_dict()

    model = inference_utils.load_vit_classifier(
        "dummy_path.pth", num_classes, device
    )

    # Check head input feature dim.
    self.assertEqual(model.heads.in_features, FEATURE_DIM)

    # Check head output feature dim.
    self.assertEqual(model.heads.out_features, num_classes)

    # Check all parameters except head are frozen.
    frozen_params = [
        p.requires_grad for n, p in model.named_parameters() if "heads" not in n
    ]
    self.assertTrue(all(not p for p in frozen_params))

    # Check that head parameters are NOT frozen.
    head_params = [
        p.requires_grad for n, p in model.named_parameters() if "heads" in n
    ]
    self.assertTrue(all(p for p in head_params))

  def test_load_vit_classifier_freezing_and_dims(self):
    # Load the model using the function under test.
    model = inference_utils.load_vit_classifier(
        model_path=self.model_path,
        num_classes=self.num_classes,
        device=self.device,
    )

    # Test that FEATURE_DIM is 768.
    # We check the `in_features` of the model's head.
    self.assertIsInstance(model.heads, torch.nn.Linear)
    self.assertEqual(
        model.heads.in_features,
        FEATURE_DIM,
        f"Model head in_features should be {FEATURE_DIM}, but got"
        f" {model.heads.in_features}",
    )
    self.assertEqual(
        model.heads.in_features, 768, "Model head in_features should be 768"
    )

    # Test that all layers except the head are frozen, and head is not.
    for name, param in model.named_parameters():
      if "heads" in name:
        self.assertTrue(
            param.requires_grad,
            f"Head parameter '{name}' should not be frozen.",
        )
      else:
        self.assertFalse(
            param.requires_grad, f"Parameter '{name}' should be frozen."
        )

    # Also check that the model is in eval mode.
    self.assertFalse(
        model.training,
        "Model was not in evaluation mode (model.training is True)",
    )

  @unittest.mock.patch("seaborn.heatmap")
  def test_confusion_matrix_dataframe(self, mock_heatmap):
    cm = np.array([[5, 2], [1, 7]])
    class_names = ["ClassA", "ClassB"]

    expected = pd.DataFrame(
        [[5 / 7, 2 / 7], [1 / 8, 7 / 8]], index=class_names, columns=class_names
    )

    inference_utils.show_confusion_matrix(cm, class_names)
    called_df = mock_heatmap.call_args[0][0]
    pd.testing.assert_frame_equal(called_df, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
