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

"""Unit tests for detection_utils.py."""

import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import torch

# Mock supervision before importing detection_utils since it is an external
# pip package not checked into //third_party/py.
mock_supervision = mock.MagicMock()
sys.modules["supervision"] = mock_supervision

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import detection_utils  # pylint: disable=g-bad-import-order,g-import-not-at-top


def _make_state(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    height: int = 100,
    width: int = 100,
) -> dict[str, Any]:
  """Builds a state dict in the pipeline's canonical layout.

  Args:
    masks: Bool tensor of shape ``[N, 1, H, W]``.
    boxes: Float tensor of shape ``[N, 4]`` in ``(x_min, y_min, x_max, y_max)``.
    scores: Float tensor of shape ``[N]``.
    height: Value stored under ``original_height``.
    width: Value stored under ``original_width``.

  Returns:
    A state dict with the six canonical keys.
  """
  return {
      "masks": masks,
      "masks_logits": torch.zeros_like(masks, dtype=torch.float32),
      "boxes": boxes,
      "scores": scores,
      "original_height": height,
      "original_width": width,
  }


def _box_mask(
    height: int, width: int, y0: int, y1: int, x0: int, x1: int
) -> torch.Tensor:
  """Returns a single ``[1, H, W]`` bool mask with a filled rectangle."""
  mask = torch.zeros((1, height, width), dtype=torch.bool)
  mask[0, y0:y1, x0:x1] = True
  return mask


class ResizeImageForInferenceTest(parameterized.TestCase):
  """Tests for resize_image_for_inference."""

  def test_returns_same_image_when_within_limit(self):
    """Verifies an image whose short side is within the limit is untouched."""
    image = Image.new("RGB", (200, 120))
    resized = detection_utils.resize_image_for_inference(
        image, max_short_side=128
    )
    self.assertIs(resized, image)

  def test_downscales_preserving_aspect_ratio(self):
    """Verifies the short side is capped and aspect ratio is preserved."""
    image = Image.new("RGB", (800, 400))  # short side = 400
    resized = detection_utils.resize_image_for_inference(
        image, max_short_side=200
    )
    # scale = 200/400 = 0.5 -> (400, 200)
    self.assertEqual(resized.size, (400, 200))

  def test_uses_shorter_side_as_reference(self):
    """Verifies a portrait image is scaled by its (shorter) width."""
    image = Image.new("RGB", (300, 900))  # short side = 300
    resized = detection_utils.resize_image_for_inference(
        image, max_short_side=150
    )
    self.assertEqual(resized.size, (150, 450))


class ConvertRfdetrDetectionsToStateTest(absltest.TestCase):
  """Tests for convert_rfdetr_detections_to_state."""

  def test_empty_detections_yield_zero_length_state(self):
    """Verifies a None-mask detections object produces empty tensors."""
    detections = mock.Mock()
    detections.mask = None
    detections.__len__ = mock.Mock(return_value=0)

    state = detection_utils.convert_rfdetr_detections_to_state(
        detections, image_height=64, image_width=48
    )
    self.assertEqual(state["masks"].shape, (0, 1, 64, 48))
    self.assertEqual(state["boxes"].shape, (0, 4))
    self.assertEqual(state["scores"].shape, (0,))
    self.assertEqual(state["original_height"], 64)
    self.assertEqual(state["original_width"], 48)

  def test_populated_detections_are_converted(self):
    """Verifies masks/boxes/scores are converted with the channel dim added."""
    detections = mock.Mock()
    detections.mask = np.ones((2, 10, 12), dtype=bool)
    detections.xyxy = np.array(
        [[0, 0, 5, 5], [1, 1, 8, 8]], dtype=np.float32
    )
    detections.confidence = np.array([0.9, 0.7], dtype=np.float32)
    detections.__len__ = mock.Mock(return_value=2)

    state = detection_utils.convert_rfdetr_detections_to_state(
        detections, image_height=10, image_width=12
    )
    self.assertEqual(state["masks"].shape, (2, 1, 10, 12))
    self.assertEqual(state["masks"].dtype, torch.bool)
    self.assertEqual(state["boxes"].shape, (2, 4))
    torch.testing.assert_close(
        state["scores"], torch.tensor([0.9, 0.7], dtype=torch.float32)
    )

  def test_masks_logits_are_zeros_matching_masks(self):
    """Verifies masks_logits is float32 zeros shaped like masks."""
    detections = mock.Mock()
    detections.mask = np.ones((1, 4, 4), dtype=bool)
    detections.xyxy = np.array([[0, 0, 3, 3]], dtype=np.float32)
    detections.confidence = np.array([0.5], dtype=np.float32)
    detections.__len__ = mock.Mock(return_value=1)

    state = detection_utils.convert_rfdetr_detections_to_state(
        detections, image_height=4, image_width=4
    )
    self.assertEqual(state["masks_logits"].dtype, torch.float32)
    self.assertEqual(state["masks_logits"].shape, state["masks"].shape)
    self.assertEqual(state["masks_logits"].sum().item(), 0.0)


class FilterContainedSubMasksTest(absltest.TestCase):
  """Tests for filter_contained_sub_masks."""

  def test_empty_state_is_returned_unchanged(self):
    """Verifies a zero-detection state passes through untouched."""
    state = _make_state(
        torch.zeros((0, 1, 10, 10), dtype=torch.bool),
        torch.zeros((0, 4)),
        torch.zeros((0,)),
    )
    result = detection_utils.filter_contained_sub_masks(
        state, containment_threshold=0.9
    )
    self.assertEqual(result["masks"].shape[0], 0)

  def test_drops_fully_contained_smaller_mask(self):
    """Verifies a small mask inside a large one is removed."""
    big = _box_mask(20, 20, 0, 20, 0, 20)  # area 400
    small = _box_mask(20, 20, 5, 10, 5, 10)  # area 25, fully inside big
    masks = torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 20, 20], [5, 5, 10, 10]], dtype=torch.float32),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.filter_contained_sub_masks(
        state, containment_threshold=0.9
    )
    # Only the larger mask should survive.
    self.assertEqual(result["masks"].shape[0], 1)
    self.assertAlmostEqual(result["scores"].item(), 0.9, places=5)

  def test_keeps_disjoint_masks(self):
    """Verifies two non-overlapping masks are both kept."""
    left = _box_mask(20, 20, 0, 10, 0, 5)
    right = _box_mask(20, 20, 0, 10, 15, 20)
    masks = torch.cat([left.unsqueeze(0), right.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 5, 10], [15, 0, 20, 10]], dtype=torch.float32),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.filter_contained_sub_masks(
        state, containment_threshold=0.9
    )
    self.assertEqual(result["masks"].shape[0], 2)

  def test_zero_area_mask_is_dropped(self):
    """Verifies an all-false (zero-area) mask is removed."""
    real = _box_mask(20, 20, 0, 10, 0, 10)
    empty = torch.zeros((1, 20, 20), dtype=torch.bool)
    masks = torch.cat([real.unsqueeze(0), empty.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 10, 10], [0, 0, 0, 0]], dtype=torch.float32),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.filter_contained_sub_masks(
        state, containment_threshold=0.9
    )
    self.assertEqual(result["masks"].shape[0], 1)
    self.assertAlmostEqual(result["scores"].item(), 0.9, places=5)


class GetValidBottleIndicesTest(absltest.TestCase):
  """Tests for get_valid_bottle_indices (edge-visibility filter)."""

  def test_keeps_all_inner_detections(self):
    """Verifies detections fully inside the frame are always kept."""
    masks = torch.cat(
        [
            _box_mask(100, 100, 20, 40, 20, 40).unsqueeze(0),
            _box_mask(100, 100, 50, 70, 50, 70).unsqueeze(0),
        ],
        dim=0,
    )
    state = _make_state(
        masks,
        torch.tensor(
            [[20, 20, 40, 40], [50, 50, 70, 70]], dtype=torch.float32
        ),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.get_valid_bottle_indices(state, margin=5)
    self.assertEqual(result["masks"].shape[0], 2)

  def test_drops_barely_visible_edge_detection(self):
    """Verifies a tiny edge-touching detection below the ratio is dropped."""
    inner = _box_mask(100, 100, 40, 60, 40, 60)  # area 400
    edge = _box_mask(100, 100, 0, 3, 0, 3)  # touches edge, area 9
    masks = torch.cat([inner.unsqueeze(0), edge.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[40, 40, 60, 60], [0, 0, 3, 3]], dtype=torch.float32),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.get_valid_bottle_indices(
        state, margin=5, visibility_threshold=0.5
    )
    self.assertEqual(result["masks"].shape[0], 1)

  def test_keeps_large_edge_detection(self):
    """Verifies an edge detection above the visibility ratio survives."""
    inner = _box_mask(100, 100, 40, 60, 40, 60)  # area 400
    big_edge = _box_mask(100, 100, 0, 30, 0, 30)  # touches edge, area 900
    masks = torch.cat([inner.unsqueeze(0), big_edge.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[40, 40, 60, 60], [0, 0, 30, 30]], dtype=torch.float32),
        torch.tensor([0.9, 0.8]),
    )
    result = detection_utils.get_valid_bottle_indices(
        state, margin=5, visibility_threshold=0.5
    )
    self.assertEqual(result["masks"].shape[0], 2)

  def test_no_inner_detections_returns_state_unchanged(self):
    """Verifies the original state is returned when nothing is inner."""
    edge = _box_mask(100, 100, 0, 3, 0, 3)
    state = _make_state(
        edge.unsqueeze(0),
        torch.tensor([[0, 0, 3, 3]], dtype=torch.float32),
        torch.tensor([0.9]),
    )
    result = detection_utils.get_valid_bottle_indices(state, margin=5)
    self.assertIs(result, state)


class MergeContainedBoxesTest(absltest.TestCase):
  """Tests for merge_contained_boxes."""

  def test_empty_state_is_returned_unchanged(self):
    """Verifies a zero-detection state passes through untouched."""
    state = _make_state(
        torch.zeros((0, 1, 10, 10), dtype=torch.bool),
        torch.zeros((0, 4)),
        torch.zeros((0,)),
    )
    result = detection_utils.merge_contained_boxes(state)
    self.assertEqual(result["scores"].shape[0], 0)

  def test_merges_contained_box(self):
    """Verifies a small box inside a larger box collapses into one."""
    big = _box_mask(50, 50, 0, 40, 0, 40)
    small = _box_mask(50, 50, 5, 15, 5, 15)
    masks = torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 40, 40], [5, 5, 15, 15]], dtype=torch.float32),
        torch.tensor([0.6, 0.7]),
    )
    result = detection_utils.merge_contained_boxes(
        state, containment_threshold=0.7
    )
    self.assertEqual(result["masks"].shape[0], 1)
    # Enclosing box is the larger box.
    torch.testing.assert_close(
        result["boxes"][0], torch.tensor([0.0, 0.0, 40.0, 40.0])
    )

  def test_merged_score_is_clamped_to_one(self):
    """Verifies the merged score is the sum of members capped at 1.0."""
    big = _box_mask(50, 50, 0, 40, 0, 40)
    small = _box_mask(50, 50, 5, 15, 5, 15)
    masks = torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 40, 40], [5, 5, 15, 15]], dtype=torch.float32),
        torch.tensor([0.8, 0.9]),  # sum 1.7 -> clamp to 1.0
    )
    result = detection_utils.merge_contained_boxes(state)
    self.assertEqual(result["scores"].shape[0], 1)
    self.assertAlmostEqual(result["scores"][0].item(), 1.0, places=5)

  def test_disjoint_boxes_are_not_merged(self):
    """Verifies boxes with no containment stay separate."""
    left = _box_mask(50, 50, 0, 10, 0, 10)
    right = _box_mask(50, 50, 30, 40, 30, 40)
    masks = torch.cat([left.unsqueeze(0), right.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 10, 10], [30, 30, 40, 40]], dtype=torch.float32),
        torch.tensor([0.6, 0.7]),
    )
    result = detection_utils.merge_contained_boxes(state)
    self.assertEqual(result["masks"].shape[0], 2)

  def test_output_masks_keep_channel_dim(self):
    """Verifies merged masks are returned as [N, 1, H, W]."""
    big = _box_mask(50, 50, 0, 40, 0, 40)
    small = _box_mask(50, 50, 5, 15, 5, 15)
    masks = torch.cat([big.unsqueeze(0), small.unsqueeze(0)], dim=0)
    state = _make_state(
        masks,
        torch.tensor([[0, 0, 40, 40], [5, 5, 15, 15]], dtype=torch.float32),
        torch.tensor([0.6, 0.7]),
    )
    result = detection_utils.merge_contained_boxes(state)
    self.assertEqual(result["masks"].ndim, 4)
    self.assertEqual(result["masks"].shape[1], 1)


class LetterboxImageTest(parameterized.TestCase):
  """Tests for letterbox_image."""

  def test_output_matches_target_size(self):
    """Verifies the canvas is exactly the requested size."""
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    result = detection_utils.letterbox_image(image, size=(64, 64))
    self.assertEqual(result.shape, (64, 64, 3))

  def test_padding_uses_fill_color(self):
    """Verifies letterbox padding is filled with the given color."""
    image = np.full((10, 100, 3), 255, dtype=np.uint8)  # very wide
    result = detection_utils.letterbox_image(
        image, size=(64, 64), color=(7, 8, 9)
    )
    # Top row is padding (wide image centered vertically).
    top_pixel = result[0, 0]
    np.testing.assert_array_equal(top_pixel, np.array([7, 8, 9]))

  def test_preserves_aspect_ratio(self):
    """Verifies a square input fills a square canvas edge to edge."""
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    result = detection_utils.letterbox_image(image, size=(80, 80))
    # Center pixel should come from the (scaled) image, not padding.
    np.testing.assert_array_equal(result[40, 40], np.array([200, 200, 200]))


class LetterboxSingleChannelTest(absltest.TestCase):
  """Tests for letterbox_single_channel."""

  def test_output_shape_and_dtype(self):
    """Verifies output shape matches target and dtype is preserved."""
    mask = np.full((30, 60), 255, dtype=np.uint8)
    result = detection_utils.letterbox_single_channel(mask, size=(64, 64))
    self.assertEqual(result.shape, (64, 64))
    self.assertEqual(result.dtype, np.uint8)

  def test_stays_binary_with_nearest_interpolation(self):
    """Verifies a binary mask stays in {0, 255} after letterboxing."""
    mask = np.full((30, 60), 255, dtype=np.uint8)
    result = detection_utils.letterbox_single_channel(mask, size=(64, 64))
    unique_values = set(np.unique(result).tolist())
    self.assertTrue(unique_values.issubset({0, 255}))


class GetPaddedBoxTest(parameterized.TestCase):
  """Tests for get_padded_box."""

  def test_expands_box_by_buffer(self):
    """Verifies the box grows by the buffer on every side."""
    box = [20, 20, 40, 40]
    result = detection_utils.get_padded_box(box, (100, 100), buffer=5)
    self.assertEqual(result, (15, 15, 45, 45))

  def test_clamps_to_zero_and_bounds(self):
    """Verifies expansion is clamped to the mask boundaries."""
    box = [2, 2, 98, 98]
    result = detection_utils.get_padded_box(box, (100, 100), buffer=5)
    self.assertEqual(result, (0, 0, 100, 100))

  def test_rounds_float_coordinates(self):
    """Verifies float box coordinates are rounded before padding."""
    box = [10.4, 10.6, 20.5, 20.4]
    result = detection_utils.get_padded_box(box, (100, 100), buffer=0)
    self.assertEqual(result, (10, 11, 20, 20))


class CropRawMaskedImageTest(absltest.TestCase):
  """Tests for crop_raw_masked_image."""

  def test_returns_none_for_degenerate_box(self):
    """Verifies an inverted/empty box yields None."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.crop_raw_masked_image(
        image, mask, [30, 30, 10, 10]
    )
    self.assertIsNone(result)

  def test_crops_to_box_size(self):
    """Verifies the crop has exactly the box dimensions."""
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.crop_raw_masked_image(
        image, mask, [10, 10, 30, 40]
    )
    self.assertEqual(result.size, (20, 30))  # PIL size = (width, height)

  def test_background_outside_mask_is_black(self):
    """Verifies pixels outside the mask are zeroed."""
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    result = detection_utils.crop_raw_masked_image(image, mask, [0, 0, 20, 20])
    result_array = np.array(result)
    # Corner pixel is outside the mask -> black.
    np.testing.assert_array_equal(result_array[0, 0], np.array([0, 0, 0]))
    # Inside-mask pixel keeps the source value.
    np.testing.assert_array_equal(result_array[6, 6], np.array([200, 200, 200]))


class CropMaskedImageTest(absltest.TestCase):
  """Tests for crop_masked_image."""

  def test_output_is_letterboxed_to_size(self):
    """Verifies the returned crop matches the requested letterbox size."""
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.crop_masked_image(
        image, mask, [0, 0, 50, 50], size=(64, 64)
    )
    self.assertEqual(result.size, (64, 64))

  def test_background_color_applied_outside_mask(self):
    """Verifies the configured background color fills outside the mask."""
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    result = detection_utils.crop_masked_image(
        image, mask, [0, 0, 40, 40], size=(40, 40), background_color=(1, 2, 3)
    )
    result_array = np.array(result)
    np.testing.assert_array_equal(result_array[0, 0], np.array([1, 2, 3]))


class BuildRawVariantMaskTest(absltest.TestCase):
  """Tests for build_raw_variant_mask."""

  def test_returns_none_for_degenerate_box(self):
    """Verifies an inverted box yields None (matching the raw crop)."""
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.build_raw_variant_mask(mask, [30, 30, 10, 10])
    self.assertIsNone(result)

  def test_matches_raw_crop_shape(self):
    """Verifies the variant mask aligns with the raw crop's dimensions."""
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:45, 5:45] = True
    box = [10, 10, 30, 40]

    raw_crop = detection_utils.crop_raw_masked_image(image, mask, box)
    variant_mask = detection_utils.build_raw_variant_mask(mask, box)

    self.assertIsNotNone(variant_mask)
    # PIL size is (w, h); numpy shape is (h, w).
    self.assertEqual(
        variant_mask.shape, (raw_crop.size[1], raw_crop.size[0])
    )

  def test_values_are_binary_255(self):
    """Verifies the mask is uint8 with values in {0, 255}."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True
    result = detection_utils.build_raw_variant_mask(mask, [0, 0, 30, 30])
    self.assertEqual(result.dtype, np.uint8)
    self.assertTrue(set(np.unique(result).tolist()).issubset({0, 255}))


class BuildLetterboxedVariantMaskTest(absltest.TestCase):
  """Tests for build_letterboxed_variant_mask."""

  def test_matches_letterboxed_crop_shape(self):
    """Verifies the variant mask has the same shape as the letterboxed crop."""
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.build_letterboxed_variant_mask(
        mask, [0, 0, 50, 50], size=(64, 64)
    )
    self.assertEqual(result.shape, (64, 64))
    self.assertEqual(result.dtype, np.uint8)


class FillMaskHolesTest(absltest.TestCase):
  """Tests for fill_mask_holes."""

  def test_fills_interior_hole(self):
    """Verifies an interior hole is filled."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    mask[12:18, 12:18] = False  # punch a hole
    filled = detection_utils.fill_mask_holes(mask)
    self.assertTrue(filled[15, 15])  # hole now filled

  def test_returns_bool_dtype(self):
    """Verifies the output dtype is bool."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    filled = detection_utils.fill_mask_holes(mask)
    self.assertEqual(filled.dtype, np.bool_)

  def test_leaves_solid_mask_unchanged(self):
    """Verifies a mask with no holes is preserved."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    filled = detection_utils.fill_mask_holes(mask)
    np.testing.assert_array_equal(filled, mask)

  def test_does_not_fill_background_bay(self):
    """Verifies a concavity open to the border is not filled."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    # Carve a channel from the object out to the right border.
    mask[14:16, 15:30] = False
    filled = detection_utils.fill_mask_holes(mask)
    # A pixel in the border-connected channel stays background.
    self.assertFalse(filled[15, 29])


class CropWithMeanBackgroundBlendTest(absltest.TestCase):
  """Tests for crop_with_mean_background_blend."""

  def test_output_is_letterboxed_to_size(self):
    """Verifies the blended crop matches the requested letterbox size."""
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    result = detection_utils.crop_with_mean_background_blend(
        image, mask, [0, 0, 50, 50], size=(64, 64)
    )
    self.assertEqual(result.size, (64, 64))
    self.assertEqual(result.mode, "RGB")


if __name__ == "__main__":
  absltest.main()
