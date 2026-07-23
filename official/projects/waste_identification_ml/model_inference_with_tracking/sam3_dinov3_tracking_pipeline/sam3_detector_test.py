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

"""Unit tests for sam3_detector.py."""

import pathlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy
from PIL import Image
import torch

# Mock supervision before it is imported anywhere since it is an external
# pip package not checked into //third_party/py.
mock_supervision = mock.MagicMock()


class MockDetections:

  def __init__(self, xyxy=None, confidence=None, class_id=None):
    self.xyxy = xyxy
    self.confidence = confidence
    self.class_id = class_id
    self.tracker_id = None

  def __len__(self):
    return len(self.xyxy) if self.xyxy is not None else 0

  @classmethod
  def empty(cls):
    return cls(
        xyxy=numpy.zeros((0, 4), dtype=numpy.float32),
        confidence=numpy.zeros(0, dtype=numpy.float32),
        class_id=numpy.zeros(0, dtype=int),
    )


mock_supervision.Detections = MockDetections
sys.modules["supervision"] = mock_supervision

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader  # pylint: disable=g-bad-import-order, g-import-not-at-top
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import sam3_detector  # pylint: disable=g-bad-import-order, g-import-not-at-top


def _make_prompt_config(
    score_threshold: float = 0.3,
    containment_threshold: float = 0.9,
    merge_overlap_threshold: float = 0.7,
) -> config_loader.PromptConfig:
  """Returns a PromptConfig with values suitable for tests."""
  return config_loader.PromptConfig(
      confidence_threshold=0.5,
      score_threshold=score_threshold,
      containment_threshold=containment_threshold,
      max_short_side=800,
      crop_size=(224, 224),
      crop_buffer_pixels=10,
      merge_overlap_threshold=merge_overlap_threshold,
  )


def _make_detector_without_construction(
    prompt_config: config_loader.PromptConfig | None = None,
) -> sam3_detector.SAM3Detector:
  """Returns a SAM3Detector instance initialized with mock dependencies.

  Args:
    prompt_config: Optional prompt configuration to inject into the detector.
  """
  return sam3_detector.SAM3Detector(
      model=mock.MagicMock(),
      processor=mock.MagicMock(),
      device=torch.device("cpu"),
      prompt_config=prompt_config or _make_prompt_config(),
  )


class ResolveDeviceTest(parameterized.TestCase):
  """Tests for the module-level _resolve_device helper."""

  @parameterized.named_parameters(
      dict(
          testcase_name="cpu_resolves_directly",
          requested_device="cpu",
          cuda_available=None,
          mps_available=None,
          expected_device=torch.device("cpu"),
      ),
      dict(
          testcase_name="cuda_resolves_when_available",
          requested_device="cuda:0",
          cuda_available=True,
          mps_available=None,
          expected_device=torch.device("cuda:0"),
      ),
      dict(
          testcase_name="cuda_falls_back_to_cpu_when_unavailable",
          requested_device="cuda:0",
          cuda_available=False,
          mps_available=None,
          expected_device=torch.device("cpu"),
      ),
      dict(
          testcase_name="mps_falls_back_to_cpu_when_unavailable",
          requested_device="mps",
          cuda_available=None,
          mps_available=False,
          expected_device=torch.device("cpu"),
      ),
  )
  def test_resolve_device(
      self,
      requested_device: str,
      cuda_available: bool | None,
      mps_available: bool | None,
      expected_device: torch.device,
  ):
    """Verifies device resolution and fallback conditions."""
    if cuda_available is not None:
      self.enter_context(
          mock.patch.object(
              sam3_detector.torch.cuda,
              "is_available",
              autospec=True,
              return_value=cuda_available,
          )
      )
    if mps_available is not None:
      self.enter_context(
          mock.patch.object(
              sam3_detector.torch.backends.mps,
              "is_available",
              autospec=True,
              return_value=mps_available,
          )
      )
    self.assertEqual(
        sam3_detector._resolve_device(requested_device), expected_device
    )


class FromCheckpointTest(absltest.TestCase):
  """Tests for SAM3Detector.from_checkpoint and __init__."""

  def setUp(self):
    """Patches SAM3 model construction and processor construction."""
    super().setUp()
    self.mock_sam3_model_builder = mock.MagicMock()
    self.mock_sam3_image_processor = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            sam3_detector,
            "sam3_model_builder",
            self.mock_sam3_model_builder,
        )
    )
    self.enter_context(
        mock.patch.object(
            sam3_detector,
            "sam3_image_processor",
            self.mock_sam3_image_processor,
        )
    )
    self.mock_build_model = self.mock_sam3_model_builder.build_sam3_image_model
    self.mock_processor_class = self.mock_sam3_image_processor.Sam3Processor
    self.fake_model = mock.MagicMock()
    self.mock_build_model.return_value = self.fake_model

  def test_raises_import_error_when_sam3_not_available(self):
    """Verifies from_checkpoint raises ImportError if sam3 is missing."""
    with mock.patch.object(sam3_detector, "sam3_model_builder", None):
      with self.assertRaises(ImportError):
        sam3_detector.SAM3Detector.from_checkpoint(
            checkpoint_path=pathlib.Path("/tmp/ckpt.pt"),
            device="cpu",
            prompt_config=_make_prompt_config(),
        )

  def test_moves_model_to_resolved_device(self):
    """Verifies the loaded model is moved to the resolved torch.device."""
    sam3_detector.SAM3Detector.from_checkpoint(
        checkpoint_path=pathlib.Path("/tmp/ckpt.pt"),
        device="cpu",
        prompt_config=_make_prompt_config(),
    )
    self.mock_build_model.assert_called_once_with(
        checkpoint_path="/tmp/ckpt.pt"
    )
    self.fake_model.to.assert_called_once_with(device=torch.device("cpu"))

  def test_processor_is_constructed_with_confidence_threshold(self):
    """Verifies the processor receives the prompt's confidence threshold."""
    prompt_config = _make_prompt_config()
    sam3_detector.SAM3Detector.from_checkpoint(
        checkpoint_path=pathlib.Path("/tmp/ckpt.pt"),
        device="cpu",
        prompt_config=prompt_config,
    )
    self.mock_processor_class.assert_called_once_with(
        self.fake_model,
        confidence_threshold=prompt_config.confidence_threshold,
    )

  def test_init_sets_attributes_directly(self):
    """Verifies __init__ directly stores dependencies without I/O."""
    model = mock.MagicMock()
    processor = mock.MagicMock()
    device = torch.device("cpu")
    prompt_config = _make_prompt_config()
    detector = sam3_detector.SAM3Detector(
        model=model,
        processor=processor,
        device=device,
        prompt_config=prompt_config,
    )
    self.assertIs(detector._model, model)
    self.assertIs(detector._processor, processor)
    self.assertIs(detector._device, device)
    self.assertIs(detector._config, prompt_config)


class ToSupervisionDetectionsTest(absltest.TestCase):
  """Tests for SAM3Detector.to_supervision_detections."""

  def test_returns_empty_when_boxes_tensor_is_one_dimensional(self):
    """Verifies a 1-D boxes tensor is treated as an empty detection set."""
    detector = _make_detector_without_construction()
    state = {
        "boxes": torch.zeros(0),
        "scores": torch.zeros(0),
    }
    detections = detector.to_supervision_detections(state)
    self.assertEmpty(detections)

  def test_filters_out_boxes_below_score_threshold(self):
    """Verifies boxes with score < threshold are dropped."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(score_threshold=0.5)
    )
    state = {
        "boxes": torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
        ]),
        "scores": torch.tensor([0.9, 0.2]),
    }
    detections = detector.to_supervision_detections(state)
    self.assertLen(detections, 1)
    numpy.testing.assert_array_equal(
        detections.xyxy, numpy.array([[0.0, 0.0, 10.0, 10.0]])
    )
    numpy.testing.assert_array_equal(
        detections.confidence, numpy.array([0.9], dtype=numpy.float32)
    )

  def test_returns_empty_when_all_boxes_are_below_threshold(self):
    """Verifies an empty Detections is returned when nothing passes filter."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(score_threshold=0.99)
    )
    state = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.5]),
    }
    detections = detector.to_supervision_detections(state)
    self.assertEmpty(detections)


class MoveStateToCpuTest(absltest.TestCase):
  """Tests for SAM3Detector._move_state_to_cpu."""

  def test_moves_top_level_tensors_to_cpu(self):
    """Verifies each tensor value at the top level is moved to CPU."""
    detector = _make_detector_without_construction()
    tensor_on_cpu = torch.zeros(2)
    state = {"boxes": tensor_on_cpu, "scores": torch.ones(3)}
    result = detector._move_state_to_cpu(state)
    self.assertIs(result, state)
    self.assertEqual(result["boxes"].device.type, "cpu")
    self.assertEqual(result["scores"].device.type, "cpu")

  def test_recurses_into_nested_dicts(self):
    """Verifies nested-dict tensors are also moved."""
    detector = _make_detector_without_construction()
    state = {"outer": {"inner": torch.ones(2)}}
    detector._move_state_to_cpu(state)
    self.assertEqual(state["outer"]["inner"].device.type, "cpu")

  def test_leaves_non_tensor_values_untouched(self):
    """Verifies non-tensor values are passed through unchanged."""
    detector = _make_detector_without_construction()
    state = {"name": "detection", "boxes": torch.zeros(1)}
    detector._move_state_to_cpu(state)
    self.assertEqual(state["name"], "detection")


class FilterContainedSubMasksTest(absltest.TestCase):
  """Tests for SAM3Detector._filter_contained_sub_masks."""

  def _make_state(self, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Returns a state dict with matching-length arrays for filtering."""
    count = masks.shape[0]
    return {
        "masks": masks,
        "masks_logits": torch.zeros_like(masks, dtype=torch.float32),
        "boxes": torch.zeros((count, 4), dtype=torch.float32),
        "scores": torch.ones(count, dtype=torch.float32),
    }

  def test_returns_unchanged_state_for_empty_input(self):
    """Verifies an empty masks tensor is returned unchanged."""
    detector = _make_detector_without_construction()
    state = self._make_state(torch.zeros((0, 4, 4), dtype=torch.bool))
    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 0)

  def test_removes_smaller_mask_when_contained(self):
    """Verifies a smaller mask fully contained in a larger one is removed."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(containment_threshold=0.9)
    )
    # A 4x4 large mask fully containing a 2x2 small mask (both non-zero
    # everywhere they cover). The small mask sits at rows 1-2, cols 1-2.
    large_mask = torch.ones((4, 4), dtype=torch.bool)
    small_mask = torch.zeros((4, 4), dtype=torch.bool)
    small_mask[1:3, 1:3] = True
    masks = torch.stack([large_mask, small_mask])
    state = self._make_state(masks)

    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 1)
    # The remaining mask should be the larger one (all True).
    self.assertTrue(bool(result["masks"][0].all()))

  def test_removes_smaller_mask_when_contained_first(self):
    """Verifies a smaller mask appearing first is removed and breaks early."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(containment_threshold=0.9)
    )
    large_mask = torch.ones((4, 4), dtype=torch.bool)
    small_mask = torch.zeros((4, 4), dtype=torch.bool)
    small_mask[1:3, 1:3] = True
    # Small mask first, so smaller_index == outer_index (0) when compared
    # with large.
    masks = torch.stack([small_mask, large_mask])
    state = self._make_state(masks)

    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 1)
    # The remaining mask should be the larger one (all True).
    self.assertTrue(bool(result["masks"][0].all()))

  def test_keeps_both_when_containment_below_threshold(self):
    """Verifies masks with negligible overlap are both kept."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(containment_threshold=0.9)
    )
    # Two 2x2 masks with only a 1-pixel overlap (25% containment of the
    # smaller, well below the 90% threshold).
    mask_a = torch.zeros((4, 4), dtype=torch.bool)
    mask_a[0:2, 0:2] = True
    mask_b = torch.zeros((4, 4), dtype=torch.bool)
    mask_b[1:3, 1:3] = True
    state = self._make_state(torch.stack([mask_a, mask_b]))
    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 2)

  def test_skips_inner_mask_if_already_marked_for_removal(self):
    """Verifies that an already removed inner mask is skipped in nested loop."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(containment_threshold=0.9)
    )
    # Mask 0 (large, top half) does not overlap with mask 1 (bottom half).
    # Mask 2 (small, top-left 1x1) is fully contained inside mask 0.
    # When outer_index=0: mask 2 is added to indices_to_remove.
    # When outer_index=1: inner_index=2 is checked and skipped via continue.
    mask_0 = torch.zeros((4, 4), dtype=torch.bool)
    mask_0[0:2, 0:4] = True
    mask_1 = torch.zeros((4, 4), dtype=torch.bool)
    mask_1[2:4, 0:4] = True
    mask_2 = torch.zeros((4, 4), dtype=torch.bool)
    mask_2[0:1, 0:1] = True
    state = self._make_state(torch.stack([mask_0, mask_1, mask_2]))

    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 2)

  def test_removes_zero_area_mask(self):
    """Verifies that a mask with zero area is removed during comparison."""
    detector = _make_detector_without_construction()
    mask_a = torch.ones((4, 4), dtype=torch.bool)
    mask_b = torch.zeros((4, 4), dtype=torch.bool)
    state = self._make_state(torch.stack([mask_a, mask_b]))

    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 1)
    self.assertTrue(bool(result["masks"][0].all()))


class MergeContainedBoxesTest(absltest.TestCase):
  """Tests for SAM3Detector._merge_contained_boxes."""

  def _make_state(
      self,
      boxes: torch.Tensor,
      scores: torch.Tensor,
      masks: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Returns a state dict compatible with the merge pass."""
    if masks is None:
      # One-channel masks whose shape matches the number of boxes.
      masks = torch.zeros((boxes.shape[0], 1, 4, 4), dtype=torch.bool)
    return {"masks": masks, "boxes": boxes, "scores": scores}

  def test_returns_unchanged_state_for_empty_input(self):
    """Verifies an empty scores tensor is returned unchanged."""
    detector = _make_detector_without_construction()
    state = self._make_state(
        boxes=torch.zeros((0, 4), dtype=torch.float32),
        scores=torch.zeros(0, dtype=torch.float32),
    )
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["scores"].shape[0], 0)

  def test_merges_small_box_into_containing_larger_box(self):
    """Verifies a smaller box fully inside a larger one is merged with it."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(merge_overlap_threshold=0.7)
    )
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [2.0, 2.0, 6.0, 6.0],
    ])
    scores = torch.tensor([0.4, 0.5])
    state = self._make_state(boxes=boxes, scores=scores)

    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["boxes"].shape[0], 1)
    numpy.testing.assert_array_equal(
        result["boxes"].numpy(), numpy.array([[0.0, 0.0, 10.0, 10.0]])
    )
    # Summed score should be 0.9, well below the 1.0 clamp.
    self.assertAlmostEqual(float(result["scores"][0]), 0.9, places=5)

  def test_keeps_separate_boxes_when_no_containment(self):
    """Verifies non-overlapping boxes are not merged."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(merge_overlap_threshold=0.7)
    )
    boxes = torch.tensor([
        [0.0, 0.0, 5.0, 5.0],
        [20.0, 20.0, 25.0, 25.0],
    ])
    scores = torch.tensor([0.6, 0.7])
    state = self._make_state(boxes=boxes, scores=scores)
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["boxes"].shape[0], 2)

  def test_clamps_summed_score_to_one(self):
    """Verifies the merged score is clamped to at most 1.0."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(merge_overlap_threshold=0.7)
    )
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [2.0, 2.0, 6.0, 6.0],
    ])
    scores = torch.tensor([0.8, 0.7])
    state = self._make_state(boxes=boxes, scores=scores)
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["boxes"].shape[0], 1)
    self.assertAlmostEqual(float(result["scores"][0]), 1.0, places=5)

  def test_handles_zero_area_box(self):
    """Verifies that a zero-area box is marked as absorbed during merge."""
    detector = _make_detector_without_construction(
        prompt_config=_make_prompt_config(merge_overlap_threshold=0.7)
    )
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 5.0, 5.0],
    ])
    scores = torch.tensor([0.8, 0.2])
    state = self._make_state(boxes=boxes, scores=scores)

    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["boxes"].shape[0], 1)


class DetectTest(absltest.TestCase):
  """Tests for SAM3Detector.detect orchestration."""

  def _make_detector(self, prompt_config=None):
    """Returns a detector with mocked processor + short-circuited helpers."""
    detector = _make_detector_without_construction(prompt_config=prompt_config)
    # set_image / set_text_prompt return the same synthetic state dict.
    self.state = {
        "masks": torch.zeros((1, 1, 4, 4), dtype=torch.bool),
        "masks_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.6]),
        "backbone_out": torch.zeros(1),
        "geometric_prompt": torch.zeros(1),
        "image_embeddings": torch.zeros(1),
    }
    detector._processor.set_image.return_value = self.state
    detector._processor.set_text_prompt.return_value = self.state
    return detector

  def test_drops_intermediate_keys_and_returns_cpu_state(self):
    """Verifies inference-only keys are removed and result is on CPU."""
    detector = self._make_detector()
    # Bypass mask filtering by replacing it with a pass-through.
    with mock.patch.object(
        detector,
        "_filter_contained_sub_masks",
        autospec=True,
        side_effect=lambda state: state,
    ):
      result = detector.detect(
          image=Image.new("RGB", (16, 16), color="red"),
          prompt="bottle",
      )
    for dropped_key in sam3_detector._INFERENCE_KEYS_TO_DROP:
      self.assertNotIn(dropped_key, result)
    self.assertEqual(result["boxes"].device.type, "cpu")

  def test_merges_only_for_the_merge_prompt(self):
    """Verifies merge runs only when the prompt matches _MERGE_PROMPT."""
    detector = self._make_detector()
    with mock.patch.object(
        detector,
        "_filter_contained_sub_masks",
        autospec=True,
        side_effect=lambda state: state,
    ), mock.patch.object(
        detector,
        "_merge_contained_boxes",
        autospec=True,
        side_effect=lambda state: state,
    ) as mock_merge:
      detector.detect(
          image=Image.new("RGB", (16, 16), color="red"),
          prompt="bottle",
      )
      mock_merge.assert_not_called()

      detector.detect(
          image=Image.new("RGB", (16, 16), color="red"),
          prompt=sam3_detector._MERGE_PROMPT,
      )
      mock_merge.assert_called_once()


if __name__ == "__main__":
  absltest.main()
