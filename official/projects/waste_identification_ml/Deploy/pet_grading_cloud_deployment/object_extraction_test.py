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

import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

# Mock supervision before it is imported anywhere
mock_supervision = mock.MagicMock()


def empty_mock_detections():
  return MockDetections()


class MockDetections:

  def __init__(self, xyxy=None, confidence=None, class_id=None):
    self.xyxy = xyxy
    self.confidence = confidence
    self.class_id = class_id
    self.tracker_id = None

  def __len__(self):
    return len(self.xyxy) if self.xyxy is not None else 0


mock_supervision.Detections = MockDetections
sys.modules["supervision"] = mock_supervision

# Mock sam3 before it is imported anywhere
mock_sam3 = mock.MagicMock()
mock_sam3_image_processor = mock.MagicMock()
sys.modules["sam3"] = mock_sam3
sys.modules["sam3.model"] = mock.MagicMock()
sys.modules["sam3.model.sam3_image_processor"] = mock_sam3_image_processor

mock_sam3.build_sam3_image_model = mock.MagicMock()


class MockSam3Processor:

  def __init__(self, model, confidence_threshold=0.5):
    self.model = model
    self.confidence_threshold = confidence_threshold


mock_sam3_image_processor.Sam3Processor = MockSam3Processor

from official.projects.waste_identification_ml.Deploy.pet_grading_cloud_deployment import object_extraction  # pylint: disable=g-bad-import-order, g-import-not-at-top

MODULE_PATH = object_extraction.__name__


class ObjectExtractorTest(unittest.TestCase):

  @mock.patch(f"{MODULE_PATH}.Sam3Processor", MockSam3Processor)
  @mock.patch(f"{MODULE_PATH}.build_sam3_image_model")
  def test_init(self, mock_build_model):
    # Arrange
    mock_model = mock.MagicMock()
    mock_build_model.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    # Act
    extractor = object_extraction.ObjectExtractor(
        checkpoint_path="mock_checkpoint.pth",
        confidence_threshold=0.6,
        score_threshold=0.5,
        containment_threshold=0.7,
        max_short_side=800,
        crop_size=(224, 224),
        device="cpu",
    )

    # Assert
    mock_build_model.assert_called_once_with(
        checkpoint_path="mock_checkpoint.pth"
    )
    mock_model.eval.assert_called_once()
    mock_model.to.assert_called_once_with(device="cpu")
    self.assertEqual(extractor._confidence_threshold, 0.6)
    self.assertEqual(extractor._score_threshold, 0.5)
    self.assertEqual(extractor._containment_threshold, 0.7)
    self.assertEqual(extractor._max_short_side, 800)
    self.assertEqual(extractor._crop_size, (224, 224))
    self.assertEqual(extractor._device, "cpu")

  def test_resize_image_for_inference_no_resize(self):
    # Image size 400x300, min side 300, max permitted is 400.
    image = Image.new("RGB", (400, 300))
    resized = object_extraction.ObjectExtractor.resize_image_for_inference(
        image, max_short_side=400
    )
    self.assertEqual(resized.size, (400, 300))

  def test_resize_image_for_inference_does_resize(self):
    # Image size 800x600, min side 600, max permitted is 300
    image = Image.new("RGB", (800, 600))
    resized = object_extraction.ObjectExtractor.resize_image_for_inference(
        image, max_short_side=300
    )
    self.assertEqual(resized.size, (400, 300))

  def test_fill_mask_holes(self):
    # Create a simple mask (10x10) with a hole (a zero surrounded by ones)
    mask = np.ones((10, 10), dtype=bool)
    mask[4, 4] = False  # Hole at index (4, 4)

    filled = object_extraction.ObjectExtractor.fill_mask_holes(mask)

    # The hole should be filled and become True
    self.assertTrue(filled[4, 4])
    self.assertTrue(np.all(filled))

  def test_get_padded_box(self):
    # Box [10, 10, 50, 50], mask shape (100, 100), buffer 5
    box = [10.0, 10.0, 50.0, 50.0]
    padded = object_extraction.ObjectExtractor.get_padded_box(
        box, (100, 100), buffer=5
    )
    self.assertEqual(padded, (5, 5, 55, 55))

  def test_get_padded_box_clamps_to_boundaries(self):
    # Box [2, 2, 98, 98], mask shape (100, 100), buffer 5
    box = [2.0, 2.0, 98.0, 98.0]
    padded = object_extraction.ObjectExtractor.get_padded_box(
        box, (100, 100), buffer=5
    )
    self.assertEqual(padded, (0, 0, 100, 100))

  def test_letterbox_image(self):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    # Target size: 200x200
    letterboxed = object_extraction.ObjectExtractor.letterbox_image(
        image, (200, 200)
    )
    self.assertEqual(letterboxed.shape, (200, 200, 3))
    # Aspect ratio should preserve
    # The padding should be at top/bottom (y range 50 to 150)
    self.assertTrue(np.all(letterboxed[:50, :, :] == 0))
    self.assertTrue(np.all(letterboxed[150:, :, :] == 0))

  def test_crop_with_mean_background_blend(self):
    image_array = np.ones((50, 50, 3), dtype=np.uint8) * 100
    mask = np.ones((50, 50), dtype=bool)
    box = [10.0, 10.0, 40.0, 40.0]
    size = (30, 30)

    cropped = object_extraction.ObjectExtractor.crop_with_mean_background_blend(
        image_array=image_array, mask=mask, box=box, size=size
    )

    self.assertIsInstance(cropped, Image.Image)
    self.assertEqual(cropped.size, size)

  def test_convert_sam3_state_to_detections(self):
    state = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [2.0, 2.0, 9.0, 9.0]]),
        "scores": torch.tensor([0.8, 0.4]),
        "masks": torch.ones((2, 100, 100), dtype=torch.bool),
    }

    detections = (
        object_extraction.ObjectExtractor.convert_sam3_state_to_detections(
            state, score_threshold=0.5
        )
    )

    self.assertEqual(len(detections), 1)
    np.testing.assert_allclose(
        detections.xyxy, np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        detections.confidence, np.array([0.8], dtype=np.float32)
    )
    np.testing.assert_equal(detections.class_id, np.array([0]))

  def test_filter_contained_sub_masks(self):
    # Two masks: one large, and one small completely inside it.
    mask1 = torch.zeros((5, 5), dtype=torch.bool)
    mask1[1:4, 1:4] = True  # Area 9

    mask2 = torch.zeros((5, 5), dtype=torch.bool)
    mask2[2, 2] = True  # Area 1, fully inside mask1

    state = {
        "masks": torch.stack([mask1, mask2]),
        "masks_logits": torch.stack([mask1.float(), mask2.float()]),
        "boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]]),
        "scores": torch.tensor([0.9, 0.85]),
    }

    extractor: object_extraction.ObjectExtractor = (
        object_extraction.ObjectExtractor.__new__(
            object_extraction.ObjectExtractor
        )
    )
    filtered = extractor.filter_contained_sub_masks(
        state, containment_threshold=0.5
    )

    # Only mask1 (the larger one) should be kept
    self.assertEqual(filtered["masks"].shape[0], 1)
    self.assertTrue(torch.equal(filtered["masks"][0], mask1))

  @mock.patch.object(object_extraction.ObjectExtractor, "extract_blended_crop")
  def test_get_cropped_objects_for_each_tracking_id(self, mock_extract_crop):
    image = Image.new("RGB", (100, 100))
    state = {
        "boxes": torch.tensor([[10.0, 15.0, 50.0, 55.0]]),
        "masks": torch.ones((1, 100, 100), dtype=torch.bool),
    }
    detections = MockDetections(
        xyxy=np.array([[10.0, 15.0, 50.0, 55.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    detections.tracker_id = np.array([42])

    mock_crop_img = Image.new("RGB", (20, 20))
    mock_extract_crop.return_value = mock_crop_img

    track_crop_records = {}

    object_extraction.ObjectExtractor.get_cropped_objects_for_each_tracking_id(
        image=image,
        state=state,
        detections=detections,
        source_frame_name="frame_0.png",
        crop_size=(20, 20),
        track_crop_records=track_crop_records,
    )

    mock_extract_crop.assert_called_once_with(
        image=image, state=state, detection_index=0, crop_size=(20, 20)
    )
    self.assertIn(42, track_crop_records)
    self.assertEqual(len(track_crop_records[42]), 1)
    self.assertEqual(track_crop_records[42][0]["frame_name"], "frame_0.png")
    self.assertEqual(track_crop_records[42][0]["crop"], mock_crop_img)

  @mock.patch(f"{MODULE_PATH}.Sam3Processor", MockSam3Processor)
  @mock.patch(f"{MODULE_PATH}.build_sam3_image_model")
  def test_extract(self, mock_build_model):
    # Set up mock model and processor
    mock_model = mock.MagicMock()
    mock_build_model.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    extractor = object_extraction.ObjectExtractor(
        checkpoint_path="mock.pth",
        confidence_threshold=0.5,
        score_threshold=0.5,
        containment_threshold=0.8,
        max_short_side=800,
        crop_size=(224, 224),
        device="cpu",
    )

    # Mock the internal processor's set_image/set_text_prompt
    mock_processor = mock.MagicMock()
    mock_processor.set_image.return_value = {
        "masks": torch.ones((1, 100, 100), dtype=torch.bool),
        "masks_logits": torch.ones((1, 100, 100), dtype=torch.float32),
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
        "scores": torch.tensor([0.9]),
    }
    mock_processor.set_text_prompt.side_effect = lambda state, prompt: state
    extractor._processor = mock_processor

    image = Image.new("RGB", (100, 100))
    resized, state, detections = extractor.extract(image, "bottle")

    # Assertions
    self.assertIsInstance(resized, Image.Image)
    self.assertIn("masks", state)
    self.assertIsInstance(detections, MockDetections)
    mock_processor.set_image.assert_called_once_with(resized)
    mock_processor.set_text_prompt.assert_called_once_with(
        state=mock.ANY, prompt="bottle"
    )

  def test_move_state_to_cpu(self):
    state = {
        "tensor1": torch.tensor([1, 2], device="cpu"),
        "nested": {
            "tensor2": torch.tensor([3, 4], device="cpu"),
            "non_tensor": "string",
        },
        "non_tensor2": 10,
    }
    extractor: object_extraction.ObjectExtractor = (
        object_extraction.ObjectExtractor.__new__(
            object_extraction.ObjectExtractor
        )
    )
    res = extractor._move_state_to_cpu(state)
    self.assertEqual(res["non_tensor2"], 10)
    self.assertEqual(res["nested"]["non_tensor"], "string")
    self.assertFalse(res["tensor1"].is_cuda)
    self.assertFalse(res["nested"]["tensor2"].is_cuda)

  def test_extract_blended_crop(self):
    image = Image.new("RGB", (50, 50))
    state = {
        "masks": torch.ones((1, 50, 50), dtype=torch.bool),
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
    }
    crop = object_extraction.ObjectExtractor.extract_blended_crop(
        image=image, state=state, detection_index=0, crop_size=(30, 30)
    )
    self.assertIsInstance(crop, Image.Image)
    self.assertEqual(crop.size, (30, 30))

  def test_get_cropped_objects_empty_detections(self):
    track_crop_records = {}
    object_extraction.ObjectExtractor.get_cropped_objects_for_each_tracking_id(
        image=Image.new("RGB", (100, 100)),
        state={},
        detections=empty_mock_detections(),
        source_frame_name="frame.png",
        crop_size=(20, 20),
        track_crop_records=track_crop_records,
    )
    self.assertEqual(track_crop_records, {})

  def test_get_cropped_objects_tracker_id_none(self):
    track_crop_records = {}
    detections = MockDetections(
        xyxy=np.array([[10.0, 15.0, 50.0, 55.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    detections.tracker_id = None
    object_extraction.ObjectExtractor.get_cropped_objects_for_each_tracking_id(
        image=Image.new("RGB", (100, 100)),
        state={},
        detections=detections,
        source_frame_name="frame.png",
        crop_size=(20, 20),
        track_crop_records=track_crop_records,
    )
    self.assertEqual(track_crop_records, {})

  def test_get_cropped_objects_tracker_id_minus_one(self):
    track_crop_records = {}
    detections = MockDetections(
        xyxy=np.array([[10.0, 15.0, 50.0, 55.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    detections.tracker_id = np.array([-1])
    object_extraction.ObjectExtractor.get_cropped_objects_for_each_tracking_id(
        image=Image.new("RGB", (100, 100)),
        state={"boxes": torch.tensor([[10.0, 15.0, 50.0, 55.0]])},
        detections=detections,
        source_frame_name="frame.png",
        crop_size=(20, 20),
        track_crop_records=track_crop_records,
    )
    self.assertEqual(track_crop_records, {})

  def test_get_cropped_objects_no_matching_state_box(self):
    track_crop_records = {}
    detections = MockDetections(
        xyxy=np.array([[10.0, 15.0, 50.0, 55.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    detections.tracker_id = np.array([42])
    object_extraction.ObjectExtractor.get_cropped_objects_for_each_tracking_id(
        image=Image.new("RGB", (100, 100)),
        state={"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]])},
        detections=detections,
        source_frame_name="frame.png",
        crop_size=(20, 20),
        track_crop_records=track_crop_records,
    )
    self.assertEqual(track_crop_records, {})


if __name__ == "__main__":
  unittest.main()
