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

"""Unit tests for rfdetr_detector.py.

RFDETRSegMedium is patched at construction so no real model is loaded; the
pure tensor post-processing (mask filter, box merge, state conversion,
supervision export) is exercised on hand-built state dicts.
"""

import sys
from unittest import mock

from absl.testing import absltest
import numpy as np
import torch

# Mock external dependencies before importing rfdetr_detector since they are
# external pip packages not checked into //third_party/py.
mock_rfdetr = mock.MagicMock()


class MockRFDETRSegMedium:
  """Mock class for RFDETRSegMedium."""

  def __init__(self, pretrain_weights=None):
    pass

  def optimize_for_inference(self):
    pass

  def predict(self, image, threshold=0.0):
    pass


mock_rfdetr.RFDETRSegMedium = MockRFDETRSegMedium
sys.modules.setdefault("rfdetr", mock_rfdetr)

mock_supervision = mock.MagicMock()


class MockDetections:
  """Mock class for supervision.Detections."""

  def __init__(self, xyxy=None, confidence=None, class_id=None, mask=None):
    self.xyxy = xyxy
    self.confidence = confidence
    self.class_id = class_id
    self.mask = mask

  def __len__(self):
    return len(self.xyxy) if self.xyxy is not None else 0

  @classmethod
  def empty(cls):
    return cls(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        class_id=np.zeros(0, dtype=int),
    )


mock_supervision.Detections = MockDetections
sys.modules.setdefault("supervision", mock_supervision)

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import rfdetr_detector  # pylint: disable=g-bad-import-order, g-import-not-at-top


def _box_mask(height: int, width: int, y0: int, y1: int, x0: int, x1: int):
  """Returns a single ``[1, H, W]`` bool mask with a filled rectangle."""
  mask = torch.zeros((1, height, width), dtype=torch.bool)
  mask[0, y0:y1, x0:x1] = True
  return mask


def _make_detector(
    containment_threshold: float = 0.98,
    merge_containment_threshold: float = 0.7,
    score_threshold: float = 0.0,
) -> rfdetr_detector.RFDETRDetector:
  """Builds an RFDETRDetector with the model and CUDA checks mocked out."""
  rfdetr_config = mock.Mock(
      device="cpu", checkpoint_path="/tmp/ckpt.pth", predict_threshold=0.2
  )
  post_processing_config = mock.Mock(
      containment_threshold=containment_threshold,
      merge_containment_threshold=merge_containment_threshold,
      score_threshold=score_threshold,
  )
  with mock.patch.object(
      rfdetr_detector, "RFDETRSegMedium", autospec=True
  ), mock.patch.object(
      rfdetr_detector.torch.cuda, "is_available", return_value=False
  ):
    return rfdetr_detector.RFDETRDetector(
        rfdetr_config=rfdetr_config,
        post_processing_config=post_processing_config,
    )


class ConstructorTest(absltest.TestCase):
  """Tests for RFDETRDetector construction."""

  def test_optimizes_model_for_inference(self):
    """Verifies the model is optimized for inference at build time."""
    rfdetr_config = mock.Mock(
        device="cpu", checkpoint_path="/tmp/ckpt.pth", predict_threshold=0.2
    )
    post_processing_config = mock.Mock(
        containment_threshold=0.98,
        merge_containment_threshold=0.7,
        score_threshold=0.0,
    )
    with mock.patch.object(
        rfdetr_detector, "RFDETRSegMedium", autospec=True
    ) as mock_model_class, mock.patch.object(
        rfdetr_detector.torch.cuda, "is_available", return_value=False
    ):
      rfdetr_detector.RFDETRDetector(
          rfdetr_config=rfdetr_config,
          post_processing_config=post_processing_config,
      )
      mock_model_class.assert_called_once_with(pretrain_weights="/tmp/ckpt.pth")
      mock_model_class.return_value.optimize_for_inference.assert_called_once()


class ConvertDetectionsToStateTest(absltest.TestCase):
  """Tests for _convert_detections_to_state."""

  def test_empty_detections_yield_zero_length_tensors(self):
    """Verifies a None-mask detection object yields empty tensors."""
    detector = _make_detector()
    detections = mock.Mock()
    detections.mask = None
    detections.__len__ = mock.Mock(return_value=0)

    state = detector._convert_detections_to_state(
        detections, image_height=20, image_width=30
    )
    self.assertEqual(state["masks"].shape, (0, 1, 20, 30))
    self.assertEqual(state["boxes"].shape, (0, 4))
    self.assertEqual(state["scores"].shape, (0,))

  def test_none_mask_with_non_zero_len_yields_empty_tensors(self):
    """Verifies non-zero detections with None mask yields empty tensors."""
    detector = _make_detector()
    detections = mock.Mock()
    detections.mask = None
    detections.__len__ = mock.Mock(return_value=1)

    state = detector._convert_detections_to_state(
        detections, image_height=20, image_width=30
    )
    self.assertEqual(state["masks"].shape, (0, 1, 20, 30))
    self.assertEqual(state["boxes"].shape, (0, 4))
    self.assertEqual(state["scores"].shape, (0,))

  def test_populated_detections_add_channel_dim(self):
    """Verifies masks gain a channel dim and scores are converted."""
    detector = _make_detector()
    detections = mock.Mock()
    detections.mask = np.ones((2, 8, 8), dtype=bool)
    detections.xyxy = np.array([[0, 0, 4, 4], [1, 1, 6, 6]], dtype=np.float32)
    detections.confidence = np.array([0.9, 0.6], dtype=np.float32)
    detections.__len__ = mock.Mock(return_value=2)

    state = detector._convert_detections_to_state(
        detections, image_height=8, image_width=8
    )
    self.assertEqual(state["masks"].shape, (2, 1, 8, 8))
    self.assertEqual(state["masks"].dtype, torch.bool)
    self.assertEqual(state["scores"].shape, (2,))


class FilterContainedSubMasksTest(absltest.TestCase):
  """Tests for _filter_contained_sub_masks."""

  def test_empty_state_unchanged(self):
    """Verifies a zero-mask state passes through untouched."""
    detector = _make_detector()
    state = {
        "masks": torch.zeros((0, 1, 10, 10), dtype=torch.bool),
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
    }
    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 0)

  def test_drops_contained_smaller_mask(self):
    """Verifies a small mask fully inside a large one is removed."""
    detector = _make_detector(containment_threshold=0.9)
    big = _box_mask(20, 20, 0, 20, 0, 20)
    small = _box_mask(20, 20, 5, 10, 5, 10)
    state = {
        "masks": torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0),
        "boxes": torch.tensor(
            [[0, 0, 20, 20], [5, 5, 10, 10]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.9, 0.8]),
    }
    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 1)
    self.assertAlmostEqual(result["scores"].item(), 0.9, places=5)

  def test_keeps_disjoint_masks(self):
    """Verifies two non-overlapping masks are both kept."""
    detector = _make_detector(containment_threshold=0.9)
    left = _box_mask(20, 20, 0, 10, 0, 5)
    right = _box_mask(20, 20, 0, 10, 15, 20)
    state = {
        "masks": torch.cat([left.unsqueeze(0), right.unsqueeze(0)], dim=0),
        "boxes": torch.tensor(
            [[0, 0, 5, 10], [15, 0, 20, 10]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.9, 0.8]),
    }
    result = detector._filter_contained_sub_masks(state)
    self.assertEqual(result["masks"].shape[0], 2)


class MergeContainedBoxesTest(absltest.TestCase):
  """Tests for _merge_contained_boxes."""

  def test_empty_state_unchanged(self):
    """Verifies a zero-detection state passes through untouched."""
    detector = _make_detector()
    state = {
        "masks": torch.zeros((0, 1, 10, 10), dtype=torch.bool),
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
    }
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["scores"].shape[0], 0)

  def test_merges_contained_box_and_clamps_score(self):
    """Verifies a contained box merges and the summed score is clamped."""
    detector = _make_detector(merge_containment_threshold=0.7)
    big = _box_mask(50, 50, 0, 40, 0, 40)
    small = _box_mask(50, 50, 5, 15, 5, 15)
    state = {
        "masks": torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0),
        "boxes": torch.tensor(
            [[0, 0, 40, 40], [5, 5, 15, 15]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.8, 0.9]),
    }
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["masks"].shape[0], 1)
    self.assertEqual(result["masks"].ndim, 4)
    self.assertAlmostEqual(result["scores"][0].item(), 1.0, places=5)
    torch.testing.assert_close(
        result["boxes"][0], torch.tensor([0.0, 0.0, 40.0, 40.0])
    )

  def test_disjoint_boxes_not_merged(self):
    """Verifies non-contained boxes stay separate."""
    detector = _make_detector(merge_containment_threshold=0.7)
    left = _box_mask(50, 50, 0, 10, 0, 10)
    right = _box_mask(50, 50, 30, 40, 30, 40)
    state = {
        "masks": torch.cat([left.unsqueeze(0), right.unsqueeze(0)], dim=0),
        "boxes": torch.tensor(
            [[0, 0, 10, 10], [30, 30, 40, 40]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.6, 0.7]),
    }
    result = detector._merge_contained_boxes(state)
    self.assertEqual(result["masks"].shape[0], 2)


class ToSupervisionDetectionsTest(absltest.TestCase):
  """Tests for to_supervision_detections."""

  def test_applies_score_threshold(self):
    """Verifies detections below the score threshold are dropped."""
    detector = _make_detector(score_threshold=0.5)
    state = {
        "masks": torch.zeros((2, 1, 10, 10), dtype=torch.bool),
        "boxes": torch.tensor(
            [[0, 0, 5, 5], [1, 1, 6, 6]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.4, 0.9], dtype=torch.float32),
    }
    fake_detections = mock.Mock()
    with mock.patch.object(
        rfdetr_detector.supervision, "Detections"
    ) as mock_detections:
      mock_detections.return_value = fake_detections
      detector.to_supervision_detections(state)
      # One detection survives the 0.5 threshold.
      _, kwargs = mock_detections.call_args
      self.assertEqual(kwargs["xyxy"].shape[0], 1)
      np.testing.assert_allclose(kwargs["confidence"], np.array([0.9]))
      # class_id is always zeroed out.
      self.assertTrue(np.all(kwargs["class_id"] == 0))

  def test_returns_empty_when_all_below_threshold(self):
    """Verifies an empty Detections is returned when nothing survives."""
    detector = _make_detector(score_threshold=0.99)
    state = {
        "masks": torch.zeros((1, 1, 10, 10), dtype=torch.bool),
        "boxes": torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
        "scores": torch.tensor([0.4], dtype=torch.float32),
    }
    sentinel = mock.Mock()
    with mock.patch.object(
        rfdetr_detector.supervision, "Detections"
    ) as mock_detections:
      mock_detections.empty.return_value = sentinel
      result = detector.to_supervision_detections(state)
      mock_detections.empty.assert_called_once()
    self.assertIs(result, sentinel)


class DetectTest(absltest.TestCase):
  """Tests for the detect orchestration method."""

  def test_runs_predict_then_filters(self):
    """Verifies detect calls predict and returns a filtered state dict."""
    detector = _make_detector()
    image = mock.Mock()
    image.size = (30, 20)  # (width, height)

    raw_detections = mock.Mock()
    detector._model.predict = mock.Mock(return_value=raw_detections)

    built_state = {
        "masks": torch.zeros((0, 1, 20, 30), dtype=torch.bool),
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
    }
    with mock.patch.object(
        detector, "_convert_detections_to_state", return_value=built_state
    ), mock.patch.object(
        detector, "_filter_contained_sub_masks", side_effect=lambda s: s
    ) as mock_filter, mock.patch.object(
        detector, "_merge_contained_boxes", side_effect=lambda s: s
    ) as mock_merge:
      result = detector.detect(image)

    detector._model.predict.assert_called_once()
    mock_filter.assert_called_once()
    mock_merge.assert_called_once()
    self.assertIs(result, built_state)


if __name__ == "__main__":
  absltest.main()
