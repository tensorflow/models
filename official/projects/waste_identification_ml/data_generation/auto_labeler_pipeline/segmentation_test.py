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

"""Unit tests for segmentation.py."""

import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from PIL import Image
import torch

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader  # pylint: disable=g-bad-import-order
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import sam3_inference_utils

sys.modules["config_loader"] = config_loader
sys.modules["sam3_inference_utils"] = sam3_inference_utils

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import segmentation  # pylint: disable=g-import-not-at-top,g-bad-import-order


def _make_detection_config() -> config_loader.DetectionConfig:
  """Returns a small, valid DetectionConfig for tests."""
  return config_loader.DetectionConfig(
      confidence_threshold=0.3,
      score_threshold=0.0,
      containment_threshold=0.98,
      max_short_side=1024,
      crop_size=(256, 256),
  )


class SegmentationTest(parameterized.TestCase):

  def test_format_elapsed_time(self):
    self.assertEqual(segmentation.format_elapsed_time(0.0), "0h 0m 0s")
    self.assertEqual(segmentation.format_elapsed_time(3665.4), "1h 1m 5s")

  def test_discover_dataset_directories_root_not_found(self):
    with self.assertRaises(FileNotFoundError):
      segmentation.discover_dataset_directories("/non_existent_root_dir_123")

  def test_discover_dataset_directories_empty(self):
    temp_root = self.create_tempdir().full_path
    with self.assertRaises(ValueError):
      segmentation.discover_dataset_directories(temp_root)

  def test_discover_dataset_directories_success(self):
    temp_root = self.create_tempdir().full_path
    os.makedirs(os.path.join(temp_root, "dataset_b"))
    os.makedirs(os.path.join(temp_root, "dataset_a"))
    # Also create a file to verify it ignores non-directories.
    open(os.path.join(temp_root, "some_file.txt"), "w").close()

    discovered = segmentation.discover_dataset_directories(temp_root)
    self.assertEqual(
        discovered,
        [
            ("dataset_a", os.path.join(temp_root, "dataset_a")),
            ("dataset_b", os.path.join(temp_root, "dataset_b")),
        ],
    )

  def test_validate_dataset_paths_missing_input_folder(self):
    temp_root = self.create_tempdir().full_path
    ds_path = os.path.join(temp_root, "ds1")
    os.makedirs(ds_path)
    with self.assertRaises(FileNotFoundError):
      segmentation.validate_dataset_paths(
          [("ds1", ds_path)], "train_val_images"
      )

  def test_validate_dataset_paths_success(self):
    temp_root = self.create_tempdir().full_path
    ds_path = os.path.join(temp_root, "ds1")
    input_path = os.path.join(ds_path, "train_val_images")
    os.makedirs(input_path)
    validated = segmentation.validate_dataset_paths(
        [("ds1", ds_path)], "train_val_images"
    )
    self.assertEqual(validated, [("ds1", input_path)])

  def test_validate_classifier_output_dir(self):
    temp_dir = self.create_tempdir().full_path
    with self.assertRaises(FileExistsError):
      segmentation.validate_classifier_output_dir(temp_dir)
    # Should not raise when path does not exist.
    segmentation.validate_classifier_output_dir(
        os.path.join(temp_dir, "non_existent")
    )

  @mock.patch.object(segmentation, "sam3_model_builder", create=True)
  @mock.patch.object(segmentation, "sam3_image_processor", create=True)
  def test_build_sam3_processor_success(
      self, mock_image_processor, mock_model_builder
  ):
    mock_model = mock.Mock()
    mock_proc = mock.Mock()
    mock_model_builder.build_sam3_image_model.return_value = mock_model
    mock_image_processor.Sam3Processor.return_value = mock_proc

    model, proc = segmentation.build_sam3_processor(
        _make_detection_config(), "/path/to/chkpt"
    )
    self.assertIs(model, mock_model)
    self.assertIs(proc, mock_proc)
    mock_model.to.assert_called_once()

  @mock.patch.object(segmentation, "sam3_model_builder", None, create=True)
  def test_build_sam3_processor_missing_sam3(self):
    with self.assertRaises(ImportError):
      segmentation.build_sam3_processor(
          _make_detection_config(), "/path/to/chkpt"
      )

  @mock.patch.object(segmentation, "process_split")
  def test_process_dataset_missing_split(self, mock_process_split):
    root_dir = self.create_tempdir().full_path
    input_dir = os.path.join(root_dir, "ds1", "train_val_images")
    os.makedirs(os.path.join(input_dir, "train"))
    # Note: no "val" folder created.

    classifier_output_dir = os.path.join(root_dir, "classifier")

    with self.assertRaises(FileNotFoundError):
      segmentation.process_dataset(
          dataset_name="ds1",
          input_dir=input_dir,
          classifier_output_dir=classifier_output_dir,
          split_names=("train", "val"),
          processor=mock.Mock(),
          detection_config=_make_detection_config(),
          prompt="packets",
          crop_variants=("imagenet_mean_background",),
          max_cpu_workers=2,
          queue_maxsize=4,
      )
    mock_process_split.assert_called_once()

  @mock.patch.object(segmentation, "process_split")
  def test_process_dataset_iterates_all_splits(self, mock_process_split):
    root_dir = self.create_tempdir().full_path
    input_dir = os.path.join(root_dir, "ds1", "train_val_images")
    os.makedirs(os.path.join(input_dir, "train"))
    os.makedirs(os.path.join(input_dir, "val"))

    classifier_output_dir = os.path.join(root_dir, "classifier")

    segmentation.process_dataset(
        dataset_name="ds1",
        input_dir=input_dir,
        classifier_output_dir=classifier_output_dir,
        split_names=("train", "val"),
        processor=mock.Mock(),
        detection_config=_make_detection_config(),
        prompt="packets",
        crop_variants=("imagenet_mean_background",),
        max_cpu_workers=2,
        queue_maxsize=4,
    )
    self.assertEqual(mock_process_split.call_count, 2)

    train_call_args = mock_process_split.call_args_list[0]
    val_call_args = mock_process_split.call_args_list[1]
    self.assertEqual(train_call_args.args[0], os.path.join(input_dir, "train"))
    self.assertEqual(
        train_call_args.args[1],
        os.path.join(classifier_output_dir, "train", "ds1"),
    )
    self.assertEqual(train_call_args.args[2], "ds1/train")
    self.assertEqual(val_call_args.args[0], os.path.join(input_dir, "val"))
    self.assertEqual(
        val_call_args.args[1],
        os.path.join(classifier_output_dir, "val", "ds1"),
    )
    self.assertEqual(val_call_args.args[2], "ds1/val")

  @mock.patch.object(segmentation.sam3_inference_utils, "run_inference")
  @mock.patch.object(segmentation, "process_one_image_cpu")
  def test_process_split_skips_when_no_detections(
      self, mock_process_one_image_cpu, mock_run_inference
  ):
    temp_root = self.create_tempdir().full_path
    split_input_dir = os.path.join(temp_root, "train")
    class_folder = os.path.join(temp_root, "classifier", "train", "ds1")
    os.makedirs(split_input_dir)

    image_path = os.path.join(split_input_dir, "foo.jpg")
    Image.new("RGB", (64, 64)).save(image_path)

    # Simulate SAM3 returning no detections for this image.
    mock_run_inference.return_value = {"scores": torch.tensor([])}

    segmentation.process_split(
        split_input_dir=split_input_dir,
        class_folder=class_folder,
        log_label="ds1/train",
        processor=mock.Mock(),
        detection_config=_make_detection_config(),
        prompt="packets",
        crop_variants=("imagenet_mean_background",),
        max_cpu_workers=2,
        queue_maxsize=4,
    )
    # No detections, so no CPU work should be submitted.
    mock_process_one_image_cpu.assert_not_called()

  @mock.patch.object(segmentation, "process_dataset")
  @mock.patch.object(segmentation, "build_sam3_processor")
  @mock.patch.object(segmentation, "discover_dataset_directories")
  @mock.patch.object(segmentation, "validate_classifier_output_dir")
  @mock.patch.object(segmentation.config_loader, "load_config")
  def test_main(
      self,
      mock_load_config,
      mock_validate_out,
      mock_discover,
      mock_build_proc,
      mock_process_dataset,
  ):
    mock_config = mock.Mock()
    mock_config.cuda_visible_devices = "0"
    mock_config.classifier_dir = "/tmp/classifier"
    mock_config.root_dir = "/tmp/root"
    mock_config.train_val_folder_name = "train_val_images"
    mock_config.train_split_name = "train"
    mock_config.val_split_name = "val"
    mock_config.prompt_to_detect = "packets"
    mock_config.crop_variants = ("imagenet_mean_background",)
    mock_config.max_cpu_workers = 2
    mock_config.queue_maxsize = 4
    mock_config.sam3_checkpoint_path = "/path/to/chkpt"
    mock_config.active_detection = _make_detection_config()
    mock_load_config.return_value = mock_config

    root_dir = self.create_tempdir().full_path
    ds_dir = os.path.join(root_dir, "ds1")
    input_dir = os.path.join(ds_dir, "train_val_images")
    os.makedirs(input_dir)

    mock_discover.return_value = [("ds1", ds_dir)]
    mock_build_proc.return_value = (mock.Mock(), mock.Mock())

    segmentation.main()
    mock_validate_out.assert_called_once_with(mock_config.classifier_dir)
    mock_process_dataset.assert_called_once()


if __name__ == "__main__":
  absltest.main()
