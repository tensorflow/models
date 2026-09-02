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

"""Unit tests for dinov3_classifier.py.

The module exposes small pure helpers (device resolution, pooling inference,
checkpoint reading) that are tested directly, plus two constructors on
`DINOv3Classifier`: the DI-friendly `__init__` used by these tests, and the
`from_config` factory whose I/O (`torch.hub.load`, `torch.load`) is patched
so no real model or checkpoint is loaded.
"""

from unittest import mock

from absl.testing import absltest
import torch

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import dinov3_classifier


def _make_probe_backbone(hidden_size: int) -> mock.Mock:
  """Returns a fake backbone that exposes `norm.normalized_shape`."""
  backbone = mock.Mock()
  backbone.norm.normalized_shape = (hidden_size,)
  return backbone


def _make_dinov3_config() -> mock.Mock:
  """Returns a stub DINOv3Config with only the fields the module reads."""
  config = mock.Mock()
  config.checkpoint_path = "/tmp/ckpt.pth"
  config.repo_dir = "/tmp/dinov3"
  config.model_name = "dinov3_vitl16"
  config.inference_image_size = 32
  config.image_mean = (0.485, 0.456, 0.406)
  config.image_std = (0.229, 0.224, 0.225)
  return config


class ResolveDeviceTest(absltest.TestCase):
  """Tests for _resolve_device."""

  def test_returns_cpu_when_requested(self):
    """Verifies an explicit CPU request always resolves to CPU."""
    self.assertEqual(
        dinov3_classifier._resolve_device("cpu"), torch.device("cpu")
    )

  def test_falls_back_to_cpu_when_cuda_unavailable(self):
    """Verifies a CUDA request without CUDA falls back to CPU."""
    with mock.patch.object(
        dinov3_classifier.torch.cuda, "is_available", return_value=False
    ):
      self.assertEqual(
          dinov3_classifier._resolve_device("cuda"), torch.device("cpu")
      )

  def test_returns_cuda_when_available(self):
    """Verifies a CUDA request resolves to CUDA when available."""
    with mock.patch.object(
        dinov3_classifier.torch.cuda, "is_available", return_value=True
    ):
      self.assertEqual(
          dinov3_classifier._resolve_device("cuda"), torch.device("cuda")
      )


class InferPoolingFromStateDictTest(absltest.TestCase):
  """Tests for _infer_pooling_from_state_dict."""

  def test_infers_cls_when_head_matches_hidden(self):
    """Verifies a head width equal to hidden size yields CLS pooling."""
    state_dict = {"head.weight": torch.zeros(3, 32)}
    result = dinov3_classifier._infer_pooling_from_state_dict(
        saved_state_dict=state_dict, hidden_size=32
    )
    self.assertIs(result, dinov3_classifier.PoolingStrategy.CLS)

  def test_infers_cls_mean_patch_when_head_is_double_hidden(self):
    """Verifies a head width of 2*hidden yields CLS_MEAN_PATCH pooling."""
    state_dict = {"head.weight": torch.zeros(3, 64)}
    result = dinov3_classifier._infer_pooling_from_state_dict(
        saved_state_dict=state_dict, hidden_size=32
    )
    self.assertIs(result, dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH)

  def test_raises_when_head_matches_neither(self):
    """Verifies a mismatched head width raises ClassifierError."""
    state_dict = {"head.weight": torch.zeros(3, 99)}
    with self.assertRaisesRegex(
        dinov3_classifier.ClassifierError, "Cannot infer"
    ):
      dinov3_classifier._infer_pooling_from_state_dict(
          saved_state_dict=state_dict, hidden_size=32
      )


class LoadCheckpointStateDictTest(absltest.TestCase):
  """Tests for _load_checkpoint_state_dict."""

  def test_returns_state_dict_when_valid(self):
    """Verifies a well-formed checkpoint yields its model_state_dict."""
    state_dict = {"head.weight": torch.zeros(2, 32)}
    with mock.patch.object(
        dinov3_classifier.torch,
        "load",
        return_value={"model_state_dict": state_dict},
    ):
      result = dinov3_classifier._load_checkpoint_state_dict(
          checkpoint_path="/tmp/ckpt.pth", device=torch.device("cpu")
      )
    self.assertIs(result, state_dict)

  def test_raises_when_model_state_dict_missing(self):
    """Verifies a checkpoint without model_state_dict raises."""
    with mock.patch.object(dinov3_classifier.torch, "load", return_value={}):
      with self.assertRaisesRegex(
          dinov3_classifier.ClassifierError, "model_state_dict"
      ):
        dinov3_classifier._load_checkpoint_state_dict(
            checkpoint_path="/tmp/ckpt.pth", device=torch.device("cpu")
        )

  def test_raises_when_head_weight_missing(self):
    """Verifies a checkpoint without head.weight raises."""
    with mock.patch.object(
        dinov3_classifier.torch,
        "load",
        return_value={"model_state_dict": {"other.weight": torch.zeros(2, 2)}},
    ):
      with self.assertRaisesRegex(
          dinov3_classifier.ClassifierError, "head.weight"
      ):
        dinov3_classifier._load_checkpoint_state_dict(
            checkpoint_path="/tmp/ckpt.pth", device=torch.device("cpu")
        )


class DINOv3ClassificationModuleTest(absltest.TestCase):
  """Tests for DINOv3ClassificationModule head sizing and pooling dispatch."""

  def test_head_input_dimension_for_cls_pooling(self):
    """Verifies CLS pooling sizes the head to the hidden dimension."""
    module = dinov3_classifier.DINOv3ClassificationModule(
        backbone_model=_make_probe_backbone(32),
        hidden_size=32,
        number_of_classes=5,
        pooling=dinov3_classifier.PoolingStrategy.CLS,
    )
    self.assertEqual(module.head.in_features, 32)
    self.assertEqual(module.head.out_features, 5)

  def test_head_input_dimension_for_cls_mean_patch_pooling(self):
    """Verifies CLS_MEAN_PATCH sizing doubles the head input dimension."""
    module = dinov3_classifier.DINOv3ClassificationModule(
        backbone_model=_make_probe_backbone(32),
        hidden_size=32,
        number_of_classes=5,
        pooling=dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )
    self.assertEqual(module.head.in_features, 64)

  def test_extract_features_cls_calls_backbone_directly(self):
    """Verifies CLS pooling passes the input straight through the backbone."""
    backbone = _make_probe_backbone(32)
    features = torch.ones((2, 32))
    backbone.return_value = features
    module = dinov3_classifier.DINOv3ClassificationModule(
        backbone_model=backbone,
        hidden_size=32,
        number_of_classes=3,
        pooling=dinov3_classifier.PoolingStrategy.CLS,
    )
    result = module.extract_features(torch.zeros((2, 3, 32, 32)))
    self.assertEqual(result.shape, (2, 32))

  def test_extract_features_cls_mean_patch_concatenates_tokens(self):
    """Verifies CLS_MEAN_PATCH concatenates CLS with the mean patch token."""
    backbone = _make_probe_backbone(32)
    backbone.forward_features.return_value = {
        "x_norm_clstoken": torch.ones((2, 32)),
        "x_norm_patchtokens": torch.ones((2, 8, 32)),
    }
    module = dinov3_classifier.DINOv3ClassificationModule(
        backbone_model=backbone,
        hidden_size=32,
        number_of_classes=3,
        pooling=dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )
    result = module.extract_features(torch.zeros((2, 3, 32, 32)))
    self.assertEqual(result.shape, (2, 64))


class PredictBatchTest(absltest.TestCase):
  """Tests for DINOv3Classifier.predict_batch post-processing."""

  def _make_classifier(
      self, class_names: list[str], logits: torch.Tensor
  ) -> dinov3_classifier.DINOv3Classifier:
    """Builds a classifier with a stubbed transform and model."""
    model = mock.Mock(return_value=logits)
    # Transform ignores the image and returns a fixed CHW tensor.
    image_transform = lambda image: torch.zeros(3, 4, 4)
    return dinov3_classifier.DINOv3Classifier(
        model=model,
        class_names=class_names,
        image_transform=image_transform,
        device=torch.device("cpu"),
    )

  def test_empty_images_returns_empty_list(self):
    """Verifies an empty image list short-circuits to an empty result."""
    classifier = self._make_classifier(["a"], torch.zeros(1, 1))
    self.assertEqual(classifier.predict_batch([]), [])

  def test_returns_argmax_class_per_row(self):
    """Verifies each row's predicted class is the argmax of its logits."""
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    classifier = self._make_classifier(["a", "b", "c"], logits)
    result = classifier.predict_batch([object(), object()])
    self.assertEqual(result[0]["predicted_class"], "a")
    self.assertEqual(result[1]["predicted_class"], "c")

  def test_probabilities_are_percentages_summing_to_one_hundred(self):
    """Verifies probabilities are percentages that cover every class."""
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    classifier = self._make_classifier(["a", "b", "c"], logits)
    prediction = classifier.predict_batch([object()])[0]
    self.assertGreater(prediction["predicted_probability_percent"], 90.0)
    self.assertCountEqual(
        prediction["all_probabilities_percent"].keys(), ["a", "b", "c"]
    )
    total = sum(prediction["all_probabilities_percent"].values())
    self.assertAlmostEqual(total, 100.0, places=3)


class FromConfigTest(absltest.TestCase):
  """Tests for DINOv3Classifier.from_config end-to-end wiring."""

  def _patched_environment(
      self, head_width: int, hidden_size: int = 32
  ) -> mock.Mock:
    """Patches torch.hub.load, torch.load, and load_state_dict as a group.

    Args:
      head_width: The width of the classification head.
      hidden_size: The hidden size of the backbone.

    Returns:
      The mock for torch.hub.load so tests can assert its call count.
    """
    fake_backbone = _make_probe_backbone(hidden_size)
    mock_hub_load = self.enter_context(
        mock.patch.object(
            dinov3_classifier.torch.hub,
            "load",
            return_value=fake_backbone,
        )
    )
    self.enter_context(
        mock.patch.object(
            dinov3_classifier.torch,
            "load",
            return_value={
                "model_state_dict": {
                    "head.weight": torch.zeros(2, head_width),
                }
            },
        )
    )
    # Bypass the real state-dict load, which would complain about the
    # fake backbone's missing parameters.
    self.enter_context(
        mock.patch.object(
            dinov3_classifier.DINOv3ClassificationModule,
            "load_state_dict",
            autospec=True,
        )
    )
    return mock_hub_load

  def test_loads_backbone_exactly_once(self):
    """Verifies from_config invokes torch.hub.load a single time."""
    mock_hub_load = self._patched_environment(head_width=32)
    dinov3_classifier.DINOv3Classifier.from_config(
        config=_make_dinov3_config(),
        class_names=["a", "b"],
        device="cpu",
    )
    self.assertEqual(mock_hub_load.call_count, 1)

  def test_returns_classifier_in_eval_mode(self):
    """Verifies the returned classifier's model is in eval mode."""
    self._patched_environment(head_width=32)
    classifier = dinov3_classifier.DINOv3Classifier.from_config(
        config=_make_dinov3_config(),
        class_names=["a", "b"],
        device="cpu",
    )
    self.assertFalse(classifier._model.training)

  def test_selects_pooling_strategy_from_checkpoint(self):
    """Verifies pooling is inferred from the head width in the checkpoint."""
    self._patched_environment(head_width=64, hidden_size=32)
    classifier = dinov3_classifier.DINOv3Classifier.from_config(
        config=_make_dinov3_config(),
        class_names=["a", "b"],
        device="cpu",
    )
    self.assertIs(
        classifier._model.pooling,
        dinov3_classifier.PoolingStrategy.CLS_MEAN_PATCH,
    )

  def test_wraps_state_dict_load_error(self):
    """Verifies a RuntimeError from load_state_dict is re-raised as ClassifierError."""
    fake_backbone = _make_probe_backbone(32)
    with mock.patch.object(
        dinov3_classifier.torch.hub, "load", return_value=fake_backbone
    ), mock.patch.object(
        dinov3_classifier.torch,
        "load",
        return_value={"model_state_dict": {"head.weight": torch.zeros(2, 32)}},
    ), mock.patch.object(
        dinov3_classifier.DINOv3ClassificationModule,
        "load_state_dict",
        side_effect=RuntimeError("shape mismatch"),
    ):
      with self.assertRaisesRegex(
          dinov3_classifier.ClassifierError, "state dict"
      ):
        dinov3_classifier.DINOv3Classifier.from_config(
            config=_make_dinov3_config(),
            class_names=["a", "b"],
            device="cpu",
        )


if __name__ == "__main__":
  absltest.main()
