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

"""Unit tests for segmentation.py."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
import torch

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import config_loader
from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline import segmentation

_ROTATION_FILL_COLOR = (124, 116, 104)


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

  def test_build_variant_directories_single_variant_is_flat(self):
    # A single variant means the class folder itself is the output dir.
    class_folder = self.create_tempdir().full_path
    directories = segmentation.build_variant_directories(
        class_folder, ("imagenet_mean_background",)
    )
    self.assertEqual(directories, {"imagenet_mean_background": class_folder})

  def test_build_variant_directories_multi_variant_makes_subdirs(self):
    class_folder = self.create_tempdir().full_path
    directories = segmentation.build_variant_directories(
        class_folder, ("raw", "imagenet_mean_background")
    )
    self.assertTrue(os.path.isdir(directories["raw"]))
    self.assertTrue(os.path.isdir(directories["imagenet_mean_background"]))
    self.assertNotEqual(directories["raw"], class_folder)

  def test_build_variant_crop_unknown_variant_raises(self):
    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    with self.assertRaises(ValueError):
      segmentation.build_variant_crop(
          image_array,
          mask,
          [0, 0, 32, 32],
          (32, 32),
          "unknown_variant",
          _ROTATION_FILL_COLOR,
      )

  def test_build_variant_mask_matches_crop_shape_for_all_variants(self):
    # For every variant, the returned mask must be alignable with the crop.
    # Raw crops are box-sized; letterboxed variants are crop_size.
    image_array = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 15:35] = True
    box = [15.0, 15.0, 35.0, 35.0]
    crop_size = (64, 64)

    for variant in ("raw", "black_background", "imagenet_mean_background"):
      crop = segmentation.build_variant_crop(
          image_array,
          mask,
          box,
          crop_size,
          variant,
          _ROTATION_FILL_COLOR,
      )
      variant_mask = segmentation.build_variant_mask(
          mask, box, crop_size, variant
      )
      self.assertIsNotNone(variant_mask, msg=f"variant={variant}")
      # (H, W) mask vs. PIL (W, H) size.
      self.assertEqual(
          variant_mask.shape,
          (crop.size[1], crop.size[0]),
          msg=f"variant={variant}",
      )

  def test_build_variant_mask_unknown_variant_raises(self):
    mask = np.zeros((32, 32), dtype=bool)
    with self.assertRaises(ValueError):
      segmentation.build_variant_mask(
          mask, [0, 0, 32, 32], (32, 32), "unknown_variant"
      )

  def test_generate_selected_crops_build_masks_true_returns_arrays(self):
    # When build_masks is True, every variant entry in variant_to_mask is
    # a numpy array (not None) with the correct shape.
    image = Image.new("RGB", (64, 64), color=(200, 200, 200))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    state = {
        "masks": torch.tensor(mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0),
        "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
        "scores": torch.tensor([0.9]),
    }
    records = segmentation.generate_selected_crops(
        image=image,
        state=state,
        score_threshold=0.0,
        crop_size=(64, 64),
        variants=("raw", "imagenet_mean_background"),
        rotation_fill_color=_ROTATION_FILL_COLOR,
        build_masks=True,
    )
    self.assertLen(records, 1)
    _, variant_to_crop, variant_to_mask = records[0]
    for variant in ("raw", "imagenet_mean_background"):
      self.assertIn(variant, variant_to_crop)
      self.assertIsNotNone(variant_to_mask[variant])

  def test_generate_selected_crops_build_masks_false_returns_none_masks(self):
    # When build_masks is False, every mask entry is None regardless of
    # variant. The crop entries are still populated.
    image = Image.new("RGB", (64, 64), color=(200, 200, 200))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    state = {
        "masks": torch.tensor(mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0),
        "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
        "scores": torch.tensor([0.9]),
    }
    records = segmentation.generate_selected_crops(
        image=image,
        state=state,
        score_threshold=0.0,
        crop_size=(64, 64),
        variants=("raw", "imagenet_mean_background"),
        rotation_fill_color=_ROTATION_FILL_COLOR,
        build_masks=False,
    )
    _, variant_to_crop, variant_to_mask = records[0]
    for variant in ("raw", "imagenet_mean_background"):
      self.assertIsNotNone(variant_to_crop[variant])
      self.assertIsNone(variant_to_mask[variant])

  def test_save_one_detection_writes_mask_sidecar_when_flag_true(self):
    # save_one_detection must write <name>_mask.png alongside <name>.jpg
    # when write_masks is True.
    tmp_dir = self.create_tempdir().full_path
    variant_directories = {"raw": tmp_dir}
    crop = Image.new("RGB", (16, 16), color=(100, 100, 100))
    mask = np.full((16, 16), 255, dtype=np.uint8)

    segmentation.save_one_detection(
        detection_index=0,
        variant_to_crop={"raw": crop},
        variant_to_mask={"raw": mask},
        filename="testimg",
        variant_directories=variant_directories,
        write_masks=True,
    )
    self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "testimg_0.jpg")))
    self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "testimg_0_mask.png")))

  def test_save_one_detection_omits_mask_when_flag_false(self):
    # When write_masks is False, only the crop is written; the mask sidecar
    # must not appear.
    tmp_dir = self.create_tempdir().full_path
    variant_directories = {"raw": tmp_dir}
    crop = Image.new("RGB", (16, 16), color=(100, 100, 100))
    mask = np.full((16, 16), 255, dtype=np.uint8)

    segmentation.save_one_detection(
        detection_index=0,
        variant_to_crop={"raw": crop},
        variant_to_mask={"raw": mask},
        filename="testimg",
        variant_directories=variant_directories,
        write_masks=False,
    )
    self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "testimg_0.jpg")))
    self.assertFalse(
        os.path.isfile(os.path.join(tmp_dir, "testimg_0_mask.png"))
    )

  @mock.patch.object(segmentation, "process_split")
  def test_process_dataset_missing_split(self, mock_process_split):
    root_dir = self.create_tempdir().full_path
    input_dir = os.path.join(root_dir, "ds1", "train_val_images")
    os.makedirs(os.path.join(input_dir, "train"))
    # No "val" folder created; the dataset should fail after processing
    # train.

    classifier_output_dir = os.path.join(root_dir, "classifier")

    with self.assertRaises(FileNotFoundError):
      segmentation.process_dataset(
          dataset_name="ds1",
          input_dir=input_dir,
          classifier_output_dir=classifier_output_dir,
          split_names=("train", "val"),
          train_split_name="train",
          processor=mock.Mock(),
          detection_config=_make_detection_config(),
          prompt="packets",
          crop_variants=("imagenet_mean_background",),
          rotation_fill_color=_ROTATION_FILL_COLOR,
          max_cpu_workers=2,
          queue_maxsize=4,
      )
    mock_process_split.assert_called_once()

  @mock.patch.object(segmentation, "process_split")
  def test_process_dataset_write_masks_true_only_for_train(
      self, mock_process_split
  ):
    # train call must have write_masks=True; val call must have
    # write_masks=False. This is the key policy that keeps segmentation
    # from spending time writing masks val cannot use.
    root_dir = self.create_tempdir().full_path
    input_dir = os.path.join(root_dir, "ds1", "train_val_images")
    os.makedirs(os.path.join(input_dir, "train"))
    os.makedirs(os.path.join(input_dir, "val"))

    segmentation.process_dataset(
        dataset_name="ds1",
        input_dir=input_dir,
        classifier_output_dir=os.path.join(root_dir, "classifier"),
        split_names=("train", "val"),
        train_split_name="train",
        processor=mock.Mock(),
        detection_config=_make_detection_config(),
        prompt="packets",
        crop_variants=("imagenet_mean_background",),
        rotation_fill_color=_ROTATION_FILL_COLOR,
        max_cpu_workers=2,
        queue_maxsize=4,
    )
    self.assertEqual(mock_process_split.call_count, 2)

    train_kwargs = mock_process_split.call_args_list[0].kwargs
    val_kwargs = mock_process_split.call_args_list[1].kwargs
    self.assertTrue(train_kwargs["write_masks"])
    self.assertFalse(val_kwargs["write_masks"])

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

    # SAM3 returns no detections.
    mock_run_inference.return_value = {"scores": torch.tensor([])}

    segmentation.process_split(
        split_input_dir=split_input_dir,
        class_folder=class_folder,
        log_label="ds1/train",
        processor=mock.Mock(),
        detection_config=_make_detection_config(),
        prompt="packets",
        crop_variants=("imagenet_mean_background",),
        rotation_fill_color=_ROTATION_FILL_COLOR,
        max_cpu_workers=2,
        queue_maxsize=4,
        write_masks=True,
    )
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
    mock_config.rotation_fill_color = _ROTATION_FILL_COLOR
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
    # process_dataset must have received both the train split name and the
    # rotation fill color; both are required for the per-split mask policy
    # and the per-variant background color to work correctly.
    kwargs = mock_process_dataset.call_args.kwargs
    args = mock_process_dataset.call_args.args
    # Support both keyword and positional invocation.
    all_values = list(args) + list(kwargs.values())
    self.assertIn("train", all_values)
    self.assertIn(_ROTATION_FILL_COLOR, all_values)


if __name__ == "__main__":
  absltest.main()
