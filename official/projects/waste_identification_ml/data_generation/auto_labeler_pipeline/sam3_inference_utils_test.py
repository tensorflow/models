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

"""Unit tests for sam3_inference_utils.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import torch

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import sam3_inference_utils


class Sam3InferenceUtilsTest(parameterized.TestCase):

  def test_resize_image_for_inference_no_resize_needed(self):
    img = Image.new("RGB", (300, 400))
    resized = sam3_inference_utils.resize_image_for_inference(
        img, max_short_side=500
    )
    self.assertIs(resized, img)
    self.assertEqual(resized.size, (300, 400))

  def test_resize_image_for_inference_resizes_short_side(self):
    img = Image.new("RGB", (1000, 800))
    resized = sam3_inference_utils.resize_image_for_inference(
        img, max_short_side=400
    )
    # Short side is 800 -> scale = 400 / 800 = 0.5 -> new size (500, 400)
    self.assertEqual(resized.size, (500, 400))

  def test_move_inference_state_to_cpu(self):
    t = torch.tensor([1.0, 2.0])
    if torch.cuda.is_available():
      t = t.cuda()
    mock_t = mock.MagicMock(spec=torch.Tensor)
    state = {
        "scores": t,
        "nested": {"mask": torch.tensor([[True, False]]), "mock_t": mock_t},
        "non_tensor": 42,
    }
    cpu_state = sam3_inference_utils.move_inference_state_to_cpu(state)
    self.assertFalse(cpu_state["scores"].is_cuda)
    self.assertFalse(cpu_state["nested"]["mask"].is_cuda)
    self.assertEqual(cpu_state["non_tensor"], 42)
    mock_t.cpu.assert_called_once()

  def test_run_inference_drops_keys_and_moves_to_cpu(self):
    mock_processor = mock.Mock()
    mock_state_1 = {"step1": True}
    mock_state_2 = {
        "scores": torch.tensor([0.9, 0.8]),
        "backbone_out": torch.tensor([1.0]),
        "geometric_prompt": torch.tensor([2.0]),
        "image_embeddings": torch.tensor([3.0]),
        "masks": torch.tensor([[[True, False]]]),
    }
    mock_processor.set_image.return_value = mock_state_1
    mock_processor.set_text_prompt.return_value = mock_state_2

    img = Image.new("RGB", (64, 64))
    res = sam3_inference_utils.run_inference(mock_processor, img, "test_prompt")

    mock_processor.set_image.assert_called_once_with(img)
    mock_processor.set_text_prompt.assert_called_once_with(
        state=mock_state_1, prompt="test_prompt"
    )
    for dropped_key in sam3_inference_utils._INFERENCE_KEYS_TO_DROP:
      self.assertNotIn(dropped_key, res)
    self.assertIn("scores", res)
    self.assertIn("masks", res)

  def test_filter_contained_sub_masks(self):
    # Mask 0 is small (area 8)
    mask0 = torch.zeros((10, 10), dtype=torch.bool)
    mask0[0:4, 0:2] = True  # area 8, fully inside mask1

    # Mask 1 is large (area 10) containing mask 0
    mask1 = torch.zeros((10, 10), dtype=torch.bool)
    mask1[0:5, 0:2] = True  # area 10

    # Mask 2 is disjoint (area 4)
    mask2 = torch.zeros((10, 10), dtype=torch.bool)
    mask2[7:9, 7:9] = True

    # Mask 3 is empty / area 0
    mask3 = torch.zeros((10, 10), dtype=torch.bool)

    state = {
        "masks": torch.stack([mask0, mask1, mask2, mask3]),
        "masks_logits": torch.randn(4, 10, 10),
        "boxes": torch.tensor(
            [[0, 0, 2, 4], [0, 0, 2, 5], [7, 7, 9, 9], [0, 0, 0, 0]],
            dtype=torch.float32,
        ),
        "scores": torch.tensor([0.8, 0.9, 0.7, 0.1]),
    }

    filtered = sam3_inference_utils.filter_contained_sub_masks(
        state, containment_threshold=0.8
    )
    # Mask 0 (contained in Mask 1) and Mask 3 (zero area) should be removed
    self.assertLen(filtered["masks"], 2)
    self.assertLen(filtered["boxes"], 2)
    self.assertLen(filtered["scores"], 2)
    self.assertLen(filtered["masks_logits"], 2)
    self.assertAlmostEqual(filtered["scores"][0].item(), 0.9)
    self.assertAlmostEqual(filtered["scores"][1].item(), 0.7)

  def test_filter_contained_sub_masks_empty(self):
    empty_state = {
        "masks": torch.zeros((0, 10, 10), dtype=torch.bool),
        "scores": torch.zeros((0,)),
    }
    res = sam3_inference_utils.filter_contained_sub_masks(empty_state, 0.8)
    self.assertIs(res, empty_state)

  def test_get_valid_bottle_indices(self):
    # Mask 0 is an inner detection.
    mask0 = np.zeros((100, 100), dtype=bool)
    mask0[40:60, 40:60] = True  # area 400

    # Mask 1 is an edge detection touching x_min <= 5, exact minimum area.
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[0:20, 10:20] = True  # area 200

    # Mask 2 is an edge detection touching x_min <= 5, too small.
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[0:10, 0:10] = True  # area 100

    state = {
        "masks": (
            torch.tensor(
                np.stack([mask0, mask1, mask2]), dtype=torch.bool
            ).unsqueeze(1)
        ),
        "masks_logits": torch.randn(3, 100, 100),
        "boxes": torch.tensor(
            [
                [40, 40, 60, 60],
                [0, 10, 10, 30],
                [0, 0, 10, 10],
            ],
            dtype=torch.float32,
        ),
        "scores": torch.tensor([0.9, 0.8, 0.4]),
        "original_height": 100,
        "original_width": 100,
    }

    filtered = sam3_inference_utils.get_valid_bottle_indices(
        state, margin=5, visibility_threshold=0.5
    )
    for key in sam3_inference_utils._STATE_ARRAY_KEYS:
      self.assertLen(filtered[key], 2)
    self.assertAlmostEqual(filtered["scores"][0].item(), 0.9)
    self.assertAlmostEqual(filtered["scores"][1].item(), 0.8)

  def test_get_valid_bottle_indices_no_inner(self):
    mask0 = np.zeros((100, 100), dtype=bool)
    mask0[0:20, 0:20] = True  # touches edge
    state = {
        "masks": torch.tensor(np.stack([mask0]), dtype=torch.bool),
        "boxes": torch.tensor([[0, 0, 20, 20]], dtype=torch.float32),
        "original_height": 100,
        "original_width": 100,
    }
    res = sam3_inference_utils.get_valid_bottle_indices(state)
    self.assertIs(res, state)

  def test_merge_contained_boxes(self):
    mask0 = torch.zeros((1, 100, 100), dtype=torch.bool)
    mask0[0, 10:50, 10:50] = True
    mask1 = torch.zeros((1, 100, 100), dtype=torch.bool)
    mask1[0, 12:48, 12:48] = True
    mask2 = torch.zeros((1, 100, 100), dtype=torch.bool)
    mask2[0, 70:90, 70:90] = True

    state = {
        "masks": torch.stack([mask0, mask1, mask2]),
        "boxes": torch.tensor([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 48.0, 48.0],
            [70.0, 70.0, 90.0, 90.0],
        ]),
        "scores": torch.tensor([0.6, 0.3, 0.8]),
    }

    merged = sam3_inference_utils.merge_contained_boxes(
        state, containment_threshold=0.7
    )
    self.assertLen(merged["boxes"], 2)
    self.assertLen(merged["scores"], 2)
    self.assertAlmostEqual(merged["scores"][0].item(), 0.9)
    self.assertAlmostEqual(merged["scores"][1].item(), 0.8)

  def test_letterbox_image(self):
    img = np.full((100, 50, 3), 255, dtype=np.uint8)  # aspect ratio 2:1 (tall)
    canvas = sam3_inference_utils.letterbox_image(
        img, size=(200, 200), color=(0, 0, 0)
    )
    self.assertEqual(canvas.shape, (200, 200, 3))
    self.assertTrue(np.all(canvas[:, 50:150] == 255))
    self.assertTrue(np.all(canvas[:, 0:50] == 0))
    self.assertTrue(np.all(canvas[:, 150:200] == 0))

  def test_letterbox_single_channel_pads_with_fill_value(self):
    # A 2:1 tall single-channel image should be centered on a square canvas
    # with the padding equal to fill_value.
    single_channel = np.full((100, 50), 255, dtype=np.uint8)
    canvas = sam3_inference_utils.letterbox_single_channel(
        single_channel, size=(200, 200), fill_value=0
    )
    self.assertEqual(canvas.shape, (200, 200))
    self.assertTrue(np.all(canvas[:, 50:150] == 255))
    self.assertTrue(np.all(canvas[:, 0:50] == 0))
    self.assertTrue(np.all(canvas[:, 150:200] == 0))

  def test_letterbox_single_channel_is_strictly_binary(self):
    # Nearest-neighbor interpolation must not introduce intermediate values,
    # so a binary input must produce a binary output regardless of scaling.
    binary_input = np.zeros((30, 40), dtype=np.uint8)
    binary_input[5:20, 10:35] = 255
    canvas = sam3_inference_utils.letterbox_single_channel(
        binary_input, size=(200, 200), fill_value=0
    )
    unique_values = np.unique(canvas)
    self.assertTrue(set(unique_values.tolist()).issubset({0, 255}))

  def test_get_padded_box(self):
    box = [10.2, 5.8, 95.1, 98.9]
    padded = sam3_inference_utils.get_padded_box(
        box, mask_shape=(100, 100), buffer=5
    )
    # round([10, 6, 95, 99]) -> with buffer=5 -> [5, 1, 100, 100]
    self.assertEqual(padded, (5, 1, 100, 100))

  def test_fill_mask_holes(self):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[8:12, 8:12] = False

    self.assertFalse(mask[10, 10])
    filled = sam3_inference_utils.fill_mask_holes(mask)
    self.assertTrue(filled[10, 10])
    self.assertFalse(filled[2, 2])

  def test_crop_helpers(self):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True
    box = [15, 15, 35, 35]

    raw_crop = sam3_inference_utils.crop_raw_masked_image(img, mask, box)
    self.assertIsInstance(raw_crop, Image.Image)
    self.assertEqual(raw_crop.size, (20, 20))

    masked_crop = sam3_inference_utils.crop_masked_image(
        img, mask, box, size=(64, 64)
    )
    self.assertIsInstance(masked_crop, Image.Image)
    self.assertEqual(masked_crop.size, (64, 64))

    blended_crop = sam3_inference_utils.crop_with_mean_background_blend(
        img, mask, box, size=(64, 64)
    )
    self.assertIsInstance(blended_crop, Image.Image)
    self.assertEqual(blended_crop.size, (64, 64))

  def test_crop_masked_image_uses_background_color(self):
    # crop_masked_image now accepts a background_color; when it is set,
    # pixels outside the mask AND letterbox padding pixels must be exactly
    # that color.
    image_array = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True
    background_color = (10, 20, 30)

    crop = sam3_inference_utils.crop_masked_image(
        image_array,
        mask,
        [15, 15, 35, 35],
        size=(64, 64),
        background_color=background_color,
    )
    crop_array = np.array(crop)
    # The four corners of the letterboxed canvas are guaranteed to be
    # padding — they must match the background color exactly.
    self.assertEqual(tuple(crop_array[0, 0].tolist()), background_color)
    self.assertEqual(tuple(crop_array[-1, -1].tolist()), background_color)

  def test_crop_masked_image_default_background_is_black(self):
    # Without a background_color argument, behavior must match the previous
    # black-background contract so existing callers keep working.
    image_array = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True

    crop = sam3_inference_utils.crop_masked_image(
        image_array, mask, [15, 15, 35, 35], size=(64, 64)
    )
    crop_array = np.array(crop)
    self.assertEqual(tuple(crop_array[0, 0].tolist()), (0, 0, 0))

  def test_crop_raw_masked_image_degenerate(self):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    res = sam3_inference_utils.crop_raw_masked_image(
        img, mask, [20, 20, 20, 30]
    )
    self.assertIsNone(res)

  def test_build_raw_variant_mask_matches_raw_crop_shape(self):
    # The mask returned by build_raw_variant_mask must have the same shape
    # as the crop returned by crop_raw_masked_image, so augmentations can
    # composite them directly without any re-alignment.
    image_array = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True
    box = [15, 15, 35, 35]

    raw_crop = sam3_inference_utils.crop_raw_masked_image(
        image_array, mask, box
    )
    raw_mask = sam3_inference_utils.build_raw_variant_mask(mask, box)

    self.assertEqual(raw_mask.shape, (raw_crop.size[1], raw_crop.size[0]))
    self.assertTrue(set(np.unique(raw_mask).tolist()).issubset({0, 255}))

  def test_build_raw_variant_mask_degenerate_returns_none(self):
    mask = np.zeros((50, 50), dtype=bool)
    res = sam3_inference_utils.build_raw_variant_mask(mask, [20, 20, 20, 30])
    self.assertIsNone(res)

  def test_build_letterboxed_variant_mask_matches_letterboxed_crop_shape(self):
    # Same alignment guarantee for the letterboxed variants: the mask must
    # be exactly crop_size and binary-valued.
    image_array = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True
    box = [15, 15, 35, 35]
    crop_size = (64, 64)

    crop = sam3_inference_utils.crop_masked_image(
        image_array, mask, box, size=crop_size
    )
    aligned_mask = sam3_inference_utils.build_letterboxed_variant_mask(
        mask, box, size=crop_size
    )

    self.assertEqual(aligned_mask.shape, (crop.size[1], crop.size[0]))
    self.assertTrue(set(np.unique(aligned_mask).tolist()).issubset({0, 255}))

  def test_process_detections(self):
    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    mask0 = np.zeros((50, 50), dtype=bool)
    mask0[10:30, 10:30] = True
    mask1 = np.zeros((50, 50), dtype=bool)
    mask1[35:45, 35:45] = True

    state = {
        "masks": np.stack([mask0, mask1]),
        "boxes": np.array([[10, 10, 30, 30], [35, 35, 45, 45]]),
        "scores": torch.tensor([0.85, 0.10]),
    }

    results = list(
        sam3_inference_utils.process_detections(
            img, state, score_threshold=0.5, crop_size=(64, 64)
        )
    )
    self.assertLen(results, 1)
    idx, raw_c, masked_c, blended_c = results[0]
    self.assertEqual(idx, 0)
    self.assertIsNotNone(raw_c)
    self.assertIsNotNone(masked_c)
    self.assertIsNotNone(blended_c)

  def test_display_crop_thumbnails_invalid_crop_type(self):
    with self.assertRaises(ValueError):
      sam3_inference_utils.display_crop_thumbnails([], {}, crop_type="invalid")

  @mock.patch("builtins.print")
  def test_display_crop_thumbnails_empty(self, mock_print):
    sam3_inference_utils.display_crop_thumbnails(
        [], {"scores": torch.tensor([])}
    )
    mock_print.assert_called_once_with("No valid crops to display.")

  @mock.patch.object(sam3_inference_utils.plt, "show")
  @mock.patch.object(sam3_inference_utils.plt, "subplots")
  def test_display_crop_thumbnails_single(self, mock_subplots, mock_show):
    mock_fig = mock.Mock()
    mock_ax = mock.Mock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    img = Image.new("RGB", (32, 32))
    crop_pairs = [(0, img, img, img)]
    state = {"scores": torch.tensor([0.9])}
    sam3_inference_utils.display_crop_thumbnails(crop_pairs, state)
    mock_show.assert_called_once()
    mock_ax.imshow.assert_called_once_with(img)

  @mock.patch.object(sam3_inference_utils.plt, "show")
  @mock.patch.object(sam3_inference_utils.plt, "subplots")
  def test_display_crop_thumbnails_grid(self, mock_subplots, mock_show):
    mock_fig = mock.Mock()
    mock_ax0 = mock.Mock()
    mock_ax1 = mock.Mock()
    mock_ax2 = mock.Mock()
    mock_subplots.return_value = (
        mock_fig,
        np.array([mock_ax0, mock_ax1, mock_ax2]),
    )

    img = Image.new("RGB", (32, 32))
    crop_pairs = [(0, img, img, img), (1, img, img, img)]
    state = {"scores": torch.tensor([0.9, 0.8])}
    sam3_inference_utils.display_crop_thumbnails(
        crop_pairs, state, columns_per_row=3
    )
    mock_show.assert_called_once()
    self.assertEqual(mock_ax0.imshow.call_count, 1)
    self.assertEqual(mock_ax1.imshow.call_count, 1)
    mock_ax2.axis.assert_called_with("off")


if __name__ == "__main__":
  absltest.main()
