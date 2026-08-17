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

"""Unit tests for filter_sparse_images.py."""

import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import torch

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader  # pylint: disable=g-bad-import-order
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import sam3_inference_utils

sys.modules["config_loader"] = config_loader
sys.modules["sam3_inference_utils"] = sam3_inference_utils

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import filter_sparse_images  # pylint: disable=g-import-not-at-top,g-bad-import-order


class FilterSparseImagesTest(parameterized.TestCase):

  def test_format_elapsed_time(self):
    self.assertEqual(filter_sparse_images.format_elapsed_time(0.0), "0h 0m 0s")
    self.assertEqual(
        filter_sparse_images.format_elapsed_time(3665.4), "1h 1m 5s"
    )

  def test_discover_dataset_directories_root_not_found(self):
    with self.assertRaises(FileNotFoundError):
      filter_sparse_images.discover_dataset_directories(
          "/non_existent_root_dir_123"
      )

  def test_discover_dataset_directories_empty(self):
    temp_root = self.create_tempdir().full_path
    with self.assertRaises(ValueError):
      filter_sparse_images.discover_dataset_directories(temp_root)

  def test_discover_dataset_directories_success(self):
    temp_root = self.create_tempdir().full_path
    os.makedirs(os.path.join(temp_root, "dataset_b"))
    os.makedirs(os.path.join(temp_root, "dataset_a"))
    # Also create a file to verify it ignores non-directories
    open(os.path.join(temp_root, "some_file.txt"), "w").close()

    discovered = filter_sparse_images.discover_dataset_directories(temp_root)
    self.assertEqual(
        discovered,
        [
            ("dataset_a", os.path.join(temp_root, "dataset_a")),
            ("dataset_b", os.path.join(temp_root, "dataset_b")),
        ],
    )

  def test_validate_dataset_paths_missing_images_folder(self):
    temp_root = self.create_tempdir().full_path
    ds_path = os.path.join(temp_root, "ds1")
    os.makedirs(ds_path)
    with self.assertRaises(FileNotFoundError):
      filter_sparse_images.validate_dataset_paths(
          [(
              "ds1",
              ds_path,
          )],
          "images",
      )

  def test_validate_dataset_paths_success(self):
    temp_root = self.create_tempdir().full_path
    ds_path = os.path.join(temp_root, "ds1")
    images_path = os.path.join(ds_path, "images")
    os.makedirs(images_path)
    validated = filter_sparse_images.validate_dataset_paths(
        [(
            "ds1",
            ds_path,
        )],
        "images",
    )
    self.assertEqual(validated, [("ds1", images_path)])

  def test_validate_rejected_dir(self):
    temp_dir = self.create_tempdir().full_path
    with self.assertRaises(FileExistsError):
      filter_sparse_images.validate_rejected_dir(temp_dir)
    # Should not raise when path does not exist
    filter_sparse_images.validate_rejected_dir(
        os.path.join(temp_dir, "non_existent")
    )

  def test_gather_image_paths(self):
    temp_root = self.create_tempdir().full_path
    sub_dir = os.path.join(temp_root, "sub")
    os.makedirs(sub_dir)

    img1 = os.path.join(temp_root, "1.jpg")
    img2 = os.path.join(sub_dir, "2.PNG")
    not_img = os.path.join(temp_root, "info.txt")

    for path in [img1, img2, not_img]:
      open(path, "w").close()

    gathered = filter_sparse_images.gather_image_paths(temp_root)
    self.assertEqual(gathered, [img1, img2])

  def test_move_to_rejected(self):
    source_root = self.create_tempdir().full_path
    rejected_root = os.path.join(source_root, "rejected")
    sub_dir = os.path.join(source_root, "ds", "images")
    os.makedirs(sub_dir)
    img_path = os.path.join(sub_dir, "test.jpg")
    open(img_path, "w").close()

    filter_sparse_images.move_to_rejected(img_path, source_root, rejected_root)
    self.assertFalse(os.path.exists(img_path))
    expected_dest = os.path.join(rejected_root, "ds", "images", "test.jpg")
    self.assertTrue(os.path.exists(expected_dest))

  @mock.patch.object(filter_sparse_images.sam3_inference_utils, "run_inference")
  def test_count_detections(self, mock_run_inference):
    masks = np.zeros((2, 50, 50), dtype=bool)
    masks[0, 10:30, 10:30] = True
    masks[1, 35:45, 35:45] = True
    mock_run_inference.return_value = {
        "masks": torch.tensor(masks, dtype=torch.bool),
        "masks_logits": torch.randn(2, 50, 50),
        "boxes": torch.tensor(
            [[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]]
        ),
        "scores": torch.tensor([0.9, 0.8]),
        "original_height": 50,
        "original_width": 50,
    }
    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )
    img = Image.new("RGB", (50, 50))
    count = filter_sparse_images.count_detections(
        img, mock.Mock(), det_config, "packets"
    )
    self.assertEqual(count, 2)

  @mock.patch.object(filter_sparse_images, "count_detections")
  def test_filter_dataset_images(self, mock_count_detections):
    root_dir = self.create_tempdir().full_path
    images_dir = os.path.join(root_dir, "ds", "images")
    rejected_dir = os.path.join(root_dir, "rejected")
    os.makedirs(images_dir)

    img_keep = os.path.join(images_dir, "keep.jpg")
    img_reject = os.path.join(images_dir, "reject.jpg")
    img_corrupt = os.path.join(images_dir, "corrupt.jpg")

    # Save valid PIL images for keep and reject
    Image.new("RGB", (64, 64)).save(img_keep)
    Image.new("RGB", (64, 64)).save(img_reject)
    # Write invalid bytes for corrupt image
    with open(img_corrupt, "wb") as f:
      f.write(b"not an image")

    # count_detections returns 3 for keep.jpg,
    # 1 for reject.jpg (below min_detections=2).
    def side_effect(unused_img, *unused_args):
      del unused_img, unused_args
      return 3 if mock_count_detections.call_count == 1 else 1

    mock_count_detections.side_effect = side_effect

    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )

    rejected_count, skipped_count, total = (
        filter_sparse_images.filter_dataset_images(
            "ds",
            images_dir,
            root_dir,
            rejected_dir,
            mock.Mock(),
            det_config,
            "packets",
            min_detections=2,
        )
    )

    self.assertEqual(total, 3)
    self.assertEqual(rejected_count, 1)
    self.assertEqual(skipped_count, 1)  # corrupt.jpg skipped
    self.assertTrue(os.path.exists(img_keep))
    self.assertFalse(os.path.exists(img_reject))
    expected_rejected_path = os.path.join(
        rejected_dir, "ds", "images", "reject.jpg"
    )
    self.assertTrue(os.path.exists(expected_rejected_path))

  @mock.patch.object(filter_sparse_images, "sam3_model_builder", create=True)
  @mock.patch.object(filter_sparse_images, "sam3_image_processor", create=True)
  def test_build_sam3_processor_success(
      self, mock_image_processor, mock_model_builder
  ):
    mock_model = mock.Mock()
    mock_proc = mock.Mock()
    mock_model_builder.build_sam3_image_model.return_value = mock_model
    mock_image_processor.Sam3Processor.return_value = mock_proc

    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )
    model, proc = filter_sparse_images.build_sam3_processor(
        det_config, "/path/to/chkpt"
    )
    self.assertIs(model, mock_model)
    self.assertIs(proc, mock_proc)
    mock_model.to.assert_called_once()

  @mock.patch.object(
      filter_sparse_images, "sam3_model_builder", None, create=True
  )
  def test_build_sam3_processor_missing_sam3(self):
    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )
    with self.assertRaises(ImportError):
      filter_sparse_images.build_sam3_processor(det_config, "/path/to/chkpt")

  @mock.patch.object(
      filter_sparse_images.sam3_inference_utils, "merge_contained_boxes"
  )
  @mock.patch.object(filter_sparse_images.sam3_inference_utils, "run_inference")
  def test_count_detections_non_packets_prompt(
      self, mock_run_inference, mock_merge
  ):
    masks = np.zeros((1, 50, 50), dtype=bool)
    masks[0, 10:30, 10:30] = True
    mock_run_inference.return_value = {
        "masks": torch.tensor(masks, dtype=torch.bool),
        "masks_logits": torch.randn(1, 50, 50),
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        "scores": torch.tensor([0.9]),
        "original_height": 50,
        "original_width": 50,
    }
    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )
    img = Image.new("RGB", (50, 50))
    count = filter_sparse_images.count_detections(
        img, mock.Mock(), det_config, "bottles"
    )
    self.assertEqual(count, 1)
    mock_merge.assert_not_called()

  @mock.patch.object(filter_sparse_images, "count_detections")
  def test_filter_dataset_images_inference_failure(self, mock_count_detections):
    root_dir = self.create_tempdir().full_path
    images_dir = os.path.join(root_dir, "ds", "images")
    rejected_dir = os.path.join(root_dir, "rejected")
    os.makedirs(images_dir)

    img_fail = os.path.join(images_dir, "fail.jpg")
    Image.new("RGB", (64, 64)).save(img_fail)

    mock_count_detections.side_effect = RuntimeError("GPU OOM test")
    det_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )

    rejected, skipped, total = filter_sparse_images.filter_dataset_images(
        "ds",
        images_dir,
        root_dir,
        rejected_dir,
        mock.Mock(),
        det_config,
        "packets",
        min_detections=2,
    )
    self.assertEqual(total, 1)
    self.assertEqual(rejected, 0)
    self.assertEqual(skipped, 1)

  @mock.patch.object(filter_sparse_images, "filter_dataset_images")
  @mock.patch.object(filter_sparse_images, "build_sam3_processor")
  @mock.patch.object(filter_sparse_images, "validate_dataset_paths")
  @mock.patch.object(filter_sparse_images, "discover_dataset_directories")
  @mock.patch.object(filter_sparse_images, "validate_rejected_dir")
  @mock.patch.object(filter_sparse_images.config_loader, "load_config")
  def test_main(
      self,
      mock_load_config,
      mock_validate_rej,
      mock_discover,
      mock_validate_paths,
      mock_build_proc,
      mock_filter_ds,
  ):
    detection_config = config_loader.DetectionConfig(
        confidence_threshold=0.3,
        score_threshold=0.0,
        containment_threshold=0.98,
        max_short_side=1024,
        crop_size=(256, 256),
    )
    mock_config = mock.Mock()
    mock_config.cuda_visible_devices = "0"
    mock_config.rejected_dir = "/tmp/rejected"
    mock_config.root_dir = "/tmp/root"
    mock_config.input_images_folder_name = "images"
    mock_config.prompt_to_detect = "packets"
    mock_config.min_detections = 2
    mock_config.sam3_checkpoint_path = "/path/to/chkpt"
    mock_config.active_detection = detection_config
    mock_load_config.return_value = mock_config

    ds_dir = "/tmp/root/ds1"
    images_dir = os.path.join(ds_dir, "images")

    mock_discover.return_value = [("ds1", ds_dir)]
    mock_validate_paths.return_value = [("ds1", images_dir)]
    mock_build_proc.return_value = (mock.Mock(), mock.Mock())
    mock_filter_ds.return_value = (0, 0, 5)

    filter_sparse_images.main()

    mock_validate_rej.assert_called_once_with(mock_config.rejected_dir)
    mock_validate_paths.assert_called_once_with([("ds1", ds_dir)], "images")
    mock_build_proc.assert_called_once_with(detection_config, "/path/to/chkpt")
    mock_filter_ds.assert_called_once()


if __name__ == "__main__":
  absltest.main()
