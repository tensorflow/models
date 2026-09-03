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

"""Unit tests for track_manager.py.

ByteTrackTracker is mocked/injected at construction so no real tracker is built.
The label-voting, hole-fill, blended-crop, and no-tracking assignment logic are
exercised directly.
"""

import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np
from PIL import Image
import torch

# Mock trackers before it is imported anywhere since it is an external
# pip package not checked into //third_party/py.
mock_trackers = mock.MagicMock()


class MockByteTrackTracker:

  def __init__(
      self,
      minimum_iou_threshold: float = 0.1,
      minimum_consecutive_frames: int = 2,
  ) -> None:
    pass

  def update(self, detections: Any) -> Any:
    return detections

  def reset(self) -> None:
    pass


mock_trackers.ByteTrackTracker = MockByteTrackTracker
sys.modules.setdefault("trackers", mock_trackers)

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import track_manager  # pylint: disable=g-bad-import-order, g-import-not-at-top


class _FakeDetections:
  """Minimal stand-in for supervision.Detections used by the no-track path."""

  def __init__(self, count: int) -> None:
    self._count = count
    self.tracker_id = None
    self.confidence = np.full((count,), 0.9, dtype=np.float32)
    self.xyxy = np.zeros((count, 4), dtype=np.float32)

  def __len__(self) -> int:
    return self._count


def _make_manager(
    tracking_enabled: bool = True,
    crop_size: tuple[int, int] = (64, 64),
    crop_buffer_pixels: int = 5,
    tracker: Any | None = None,
) -> track_manager.TrackManager:
  """Builds a TrackManager with ByteTrackTracker injected or mocked."""
  tracking_config = mock.Mock(
      enable=tracking_enabled,
      bytetrack_minimum_iou_threshold=0.1,
      bytetrack_minimum_consecutive_frames=2,
  )
  cropping_config = mock.Mock(
      crop_size=crop_size, crop_buffer_pixels=crop_buffer_pixels
  )
  vis_config = mock.Mock(background_blend_color_rgb=(124, 116, 104))
  if tracker is None:
    tracker = mock.create_autospec(MockByteTrackTracker, instance=True)
  return track_manager.TrackManager(
      tracking_config=tracking_config,
      cropping_config=cropping_config,
      vis_config=vis_config,
      tracker=tracker,
  )


def _pred(class_name: str, probability: float) -> dict[str, Any]:
  """Builds a single per-crop prediction dict for the voting tests."""
  return {
      "predicted_class": class_name,
      "predicted_probability_percent": probability,
  }


class ResolveTrackLabelTest(absltest.TestCase):
  """Tests for resolve_track_label (majority vote + tie-breaks)."""

  def setUp(self):
    super().setUp()
    self.manager = _make_manager()

  def test_clear_majority_wins(self):
    """Verifies the class with the most votes wins."""
    predictions = [
        _pred("a", 90.0),
        _pred("a", 80.0),
        _pred("b", 99.0),
    ]
    label, votes = self.manager.resolve_track_label(predictions)
    self.assertEqual(label, "a")
    self.assertEqual(votes, 2)

  def test_vote_tie_broken_by_confidence_sum(self):
    """Verifies a vote tie is broken by the higher confidence sum."""
    predictions = [
        _pred("a", 60.0),
        _pred("b", 95.0),
    ]
    label, votes = self.manager.resolve_track_label(predictions)
    self.assertEqual(label, "b")
    self.assertEqual(votes, 1)

  def test_full_tie_broken_alphabetically(self):
    """Verifies a vote and confidence tie is broken alphabetically."""
    predictions = [
        _pred("banana", 50.0),
        _pred("apple", 50.0),
    ]
    label, _ = self.manager.resolve_track_label(predictions)
    self.assertEqual(label, "apple")


class FillMaskHolesTest(absltest.TestCase):
  """Tests for _fill_mask_holes."""

  def setUp(self):
    super().setUp()
    self.manager = _make_manager()

  def test_fills_interior_hole(self):
    """Verifies an interior hole is filled and dtype is bool."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    mask[12:18, 12:18] = False
    filled = self.manager._fill_mask_holes(mask)
    self.assertEqual(filled.dtype, np.bool_)
    self.assertTrue(filled[15, 15])

  def test_leaves_solid_mask_unchanged(self):
    """Verifies a hole-free mask is preserved."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    filled = self.manager._fill_mask_holes(mask)
    np.testing.assert_array_equal(filled, mask)


class ExtractBlendedCropTest(absltest.TestCase):
  """Tests for _extract_blended_crop."""

  def test_returns_letterboxed_pil_image(self):
    """Verifies the blended crop is a PIL image of the configured size."""
    manager = _make_manager(crop_size=(64, 64))
    image = Image.new("RGB", (50, 50), (10, 20, 30))
    mask = torch.zeros((1, 1, 50, 50), dtype=torch.bool)
    mask[0, 0, 10:40, 10:40] = True
    state = {
        "masks": mask,
        "boxes": torch.tensor([[0, 0, 50, 50]], dtype=torch.float32),
    }
    crop = manager._extract_blended_crop(image, state, idx=0)
    self.assertIsInstance(crop, Image.Image)
    self.assertEqual(crop.size, (64, 64))


class TrackerInjectionTest(absltest.TestCase):
  """Tests for tracker dependency injection."""

  def test_custom_tracker_injected(self):
    """Verifies custom tracker instance is used when provided."""
    custom_tracker = mock.Mock()
    manager = _make_manager(tracker=custom_tracker)
    self.assertIs(manager._tracker, custom_tracker)

  def test_default_tracker_constructed_from_config(self):
    """Verifies default tracker is constructed when tracker is None."""
    tracking_config = mock.Mock(
        enable=True,
        bytetrack_minimum_iou_threshold=0.2,
        bytetrack_minimum_consecutive_frames=3,
    )
    cropping_config = mock.Mock(crop_size=(64, 64), crop_buffer_pixels=5)
    vis_config = mock.Mock(background_blend_color_rgb=(124, 116, 104))
    manager = track_manager.TrackManager(
        tracking_config=tracking_config,
        cropping_config=cropping_config,
        vis_config=vis_config,
    )
    self.assertIsInstance(manager._tracker, MockByteTrackTracker)


class ResetTest(absltest.TestCase):
  """Tests for reset."""

  def test_clears_records_and_resets_counter(self):
    """Verifies reset clears crop records and the standalone-ID counter."""
    manager = _make_manager()
    manager._crop_records[5] = [{"frame_name": "f", "crop": None}]
    manager._next_standalone_id = 99
    manager.reset()
    self.assertEmpty(manager._crop_records)
    self.assertEqual(manager._next_standalone_id, 1)
    manager._tracker.reset.assert_called_once()


class UpdateWithoutTrackingTest(absltest.TestCase):
  """Tests for the no-tracking assignment path."""

  def test_assigns_sequential_ids_and_records_crops(self):
    """Verifies each detection gets a fresh sequential ID and a crop record."""
    manager = _make_manager(tracking_enabled=False)
    detections = _FakeDetections(count=2)
    state = {
        "masks": torch.zeros((2, 1, 20, 20), dtype=torch.bool),
        "boxes": torch.zeros((2, 4), dtype=torch.float32),
    }
    with mock.patch.object(
        manager, "_extract_blended_crop", return_value="CROP"
    ):
      returned, unassigned = manager.update_and_extract_crops(
          detections, state, Image.new("RGB", (20, 20)), "frame_0.png"
      )
    np.testing.assert_array_equal(returned.tracker_id, np.array([1, 2]))
    self.assertEqual(unassigned, [])
    self.assertIn(1, manager._crop_records)
    self.assertIn(2, manager._crop_records)

  def test_ids_continue_across_calls(self):
    """Verifies standalone IDs keep incrementing across frames."""
    manager = _make_manager(tracking_enabled=False)
    state = {
        "masks": torch.zeros((1, 1, 20, 20), dtype=torch.bool),
        "boxes": torch.zeros((1, 4), dtype=torch.float32),
    }
    with mock.patch.object(
        manager, "_extract_blended_crop", return_value="CROP"
    ):
      manager.update_and_extract_crops(
          _FakeDetections(1), state, Image.new("RGB", (20, 20)), "f0.png"
      )
      returned, _ = manager.update_and_extract_crops(
          _FakeDetections(1), state, Image.new("RGB", (20, 20)), "f1.png"
      )
    np.testing.assert_array_equal(returned.tracker_id, np.array([2]))

  def test_empty_detections_return_empty(self):
    """Verifies zero detections produce no records and no unassigned scores."""
    manager = _make_manager(tracking_enabled=False)
    state = {
        "masks": torch.zeros((0, 1, 20, 20), dtype=torch.bool),
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
    }
    returned, unassigned = manager.update_and_extract_crops(
        _FakeDetections(0), state, Image.new("RGB", (20, 20)), "f.png"
    )
    self.assertEmpty(returned)
    self.assertEqual(unassigned, [])
    self.assertEmpty(manager._crop_records)


class UpdateWithTrackingTest(absltest.TestCase):
  """Tests for the ByteTrack-enabled path."""

  def test_records_crops_for_assigned_tracks(self):
    """Verifies assigned detections get crop records keyed by tracker_id."""
    manager = _make_manager(tracking_enabled=True)

    boxes = np.array([[0, 0, 5, 5], [6, 6, 9, 9]], dtype=np.float32)
    tracked = _FakeDetections(count=2)
    tracked.xyxy = boxes
    tracked.tracker_id = np.array([10, 11])
    manager._tracker.update = mock.Mock(return_value=tracked)

    incoming = _FakeDetections(count=2)
    incoming.xyxy = boxes
    state = {
        "masks": torch.zeros((2, 1, 20, 20), dtype=torch.bool),
        "boxes": torch.from_numpy(boxes),
    }
    with mock.patch.object(
        manager, "_extract_blended_crop", return_value="CROP"
    ):
      _, unassigned = manager.update_and_extract_crops(
          incoming, state, Image.new("RGB", (20, 20)), "frame.png"
      )
    self.assertEqual(unassigned, [])
    self.assertIn(10, manager._crop_records)
    self.assertIn(11, manager._crop_records)

  def test_unassigned_scores_reported_for_dropped_detections(self):
    """Verifies detections ByteTrack leaves at id -1 surface as unassigned."""
    manager = _make_manager(tracking_enabled=True)

    boxes = np.array([[0, 0, 5, 5]], dtype=np.float32)
    tracked = _FakeDetections(count=1)
    tracked.xyxy = boxes
    tracked.tracker_id = np.array([-1])
    tracked.confidence = np.array([0.42], dtype=np.float32)
    manager._tracker.update = mock.Mock(return_value=tracked)

    incoming = _FakeDetections(count=1)
    incoming.xyxy = boxes
    incoming.confidence = np.array([0.42], dtype=np.float32)
    state = {
        "masks": torch.zeros((1, 1, 20, 20), dtype=torch.bool),
        "boxes": torch.from_numpy(boxes),
    }
    _, unassigned = manager.update_and_extract_crops(
        incoming, state, Image.new("RGB", (20, 20)), "frame.png"
    )
    self.assertLen(unassigned, 1)
    self.assertAlmostEqual(unassigned[0], 0.42, places=5)
    self.assertEmpty(manager._crop_records)


class ClassifyAllTracksTest(absltest.TestCase):
  """Tests for classify_all_tracks batching."""

  def test_batches_and_merges_predictions(self):
    """Verifies crops are classified in batches and merged with metadata."""
    manager = _make_manager()
    manager._crop_records = {
        7: [
            {"frame_name": "f0.png", "crop": "C0"},
            {"frame_name": "f1.png", "crop": "C1"},
            {"frame_name": "f2.png", "crop": "C2"},
        ]
    }

    classifier = mock.Mock()
    # Return one prediction dict per crop passed in.
    classifier.predict_batch.side_effect = lambda images: [
        {"predicted_class": "a", "predicted_probability_percent": 90.0}
        for _ in images
    ]

    result = manager.classify_all_tracks(classifier, batch_size=2)
    # 3 crops with batch_size 2 -> two predict_batch calls (2 + 1).
    self.assertEqual(classifier.predict_batch.call_count, 2)
    self.assertLen(result[7], 3)
    self.assertEqual(result[7][0]["frame_name"], "f0.png")
    self.assertEqual(result[7][0]["predicted_class"], "a")


if __name__ == "__main__":
  absltest.main()
