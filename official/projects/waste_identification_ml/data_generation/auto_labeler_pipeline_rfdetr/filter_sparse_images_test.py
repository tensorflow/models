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

"""Unit tests for filter_sparse_images.py."""

import pathlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from PIL import Image

# Mock supervision before importing filter_sparse_images since it is an external
# pip package not checked into //third_party/py.
mock_supervision = mock.MagicMock()
sys.modules["supervision"] = mock_supervision

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import filter_sparse_images  # pylint: disable=g-bad-import-order,g-import-not-at-top


def _touch(path: pathlib.Path) -> None:
  """Creates an empty file, making parent directories as needed."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


class DiscoverDatasetDirectoriesTest(absltest.TestCase):
  """Tests for discover_dataset_directories."""

  def test_returns_sorted_tuples(self):
    """Verifies dataset subfolders are returned sorted by name."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "beta").mkdir()
    (root / "alpha").mkdir()
    result = filter_sparse_images.discover_dataset_directories(str(root))
    self.assertEqual([name for name, _ in result], ["alpha", "beta"])

  def test_raises_when_root_missing(self):
    """Verifies a missing root raises FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      filter_sparse_images.discover_dataset_directories("/nope/xyz")

  def test_raises_when_empty(self):
    """Verifies an empty root raises ValueError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    with self.assertRaisesRegex(ValueError, "No dataset subfolders"):
      filter_sparse_images.discover_dataset_directories(str(root))


class ValidateDatasetPathsTest(absltest.TestCase):
  """Tests for validate_dataset_paths."""

  def test_returns_images_dir(self):
    """Verifies the resolved images directory is returned per dataset."""
    root = pathlib.Path(self.create_tempdir().full_path)
    dataset = root / "ds"
    (dataset / "images").mkdir(parents=True)
    result = filter_sparse_images.validate_dataset_paths(
        [("ds", str(dataset))], input_images_folder_name="images"
    )
    self.assertEqual(result, [("ds", str(dataset / "images"))])

  def test_raises_when_images_folder_missing(self):
    """Verifies a dataset missing its images folder raises."""
    root = pathlib.Path(self.create_tempdir().full_path)
    dataset = root / "ds"
    dataset.mkdir()
    with self.assertRaises(FileNotFoundError):
      filter_sparse_images.validate_dataset_paths(
          [("ds", str(dataset))], input_images_folder_name="images"
      )


class ValidateRejectedDirTest(absltest.TestCase):
  """Tests for validate_rejected_dir."""

  def test_passes_when_absent(self):
    """Verifies a non-existent rejected dir does not raise."""
    root = pathlib.Path(self.create_tempdir().full_path)
    # Should not raise.
    filter_sparse_images.validate_rejected_dir(str(root / "rejected"))

  def test_raises_when_present(self):
    """Verifies an existing rejected dir raises FileExistsError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    rejected = root / "rejected"
    rejected.mkdir()
    with self.assertRaises(FileExistsError):
      filter_sparse_images.validate_rejected_dir(str(rejected))


class GatherImagePathsTest(absltest.TestCase):
  """Tests for gather_image_paths."""

  def test_collects_supported_extensions_recursively(self):
    """Verifies image files are gathered recursively and sorted naturally."""
    root = pathlib.Path(self.create_tempdir().full_path)
    _touch(root / "img2.jpg")
    _touch(root / "img10.jpg")
    _touch(root / "sub" / "img1.png")
    _touch(root / "notes.txt")

    result = filter_sparse_images.gather_image_paths(str(root))
    names = [pathlib.Path(path).name for path in result]
    self.assertNotIn("notes.txt", names)
    self.assertIn("img1.png", names)
    # Natural sort places img2 before img10.
    self.assertLess(names.index("img2.jpg"), names.index("img10.jpg"))

  def test_extension_match_is_case_insensitive(self):
    """Verifies uppercase extensions are collected."""
    root = pathlib.Path(self.create_tempdir().full_path)
    _touch(root / "a.JPG")
    result = filter_sparse_images.gather_image_paths(str(root))
    self.assertLen(result, 1)


class MoveToRejectedTest(absltest.TestCase):
  """Tests for move_to_rejected."""

  def test_preserves_relative_path(self):
    """Verifies the moved file mirrors its path relative to the source root."""
    source_root = pathlib.Path(self.create_tempdir().full_path)
    rejected_root = pathlib.Path(self.create_tempdir().full_path)
    image = source_root / "ds" / "images" / "foo.jpg"
    _touch(image)

    filter_sparse_images.move_to_rejected(
        str(image), str(source_root), str(rejected_root)
    )
    destination = rejected_root / "ds" / "images" / "foo.jpg"
    self.assertTrue(destination.exists())
    self.assertFalse(image.exists())


class FormatElapsedTimeTest(parameterized.TestCase):
  """Tests for format_elapsed_time."""

  @parameterized.named_parameters(
      ("zero", 0, "0h 0m 0s"),
      ("seconds", 45, "0h 0m 45s"),
      ("minutes", 130, "0h 2m 10s"),
      ("hours", 3661, "1h 1m 1s"),
  )
  def test_formats_elapsed_seconds(self, seconds, expected):
    """Verifies elapsed seconds render as 'Hh Mm Ss'."""
    self.assertEqual(
        filter_sparse_images.format_elapsed_time(seconds), expected
    )


class CountDetectionsTest(absltest.TestCase):
  """Tests for count_detections (model and filters mocked)."""

  def _make_config(self) -> mock.Mock:
    """Returns a config stub carrying only the thresholds this path reads."""
    config = mock.Mock()
    config.predict_threshold = 0.3
    config.containment_threshold = 0.98
    config.merge_containment_threshold = 0.7
    return config

  def test_returns_zero_when_no_detections(self):
    """Verifies an empty detection state short-circuits to zero."""
    image = Image.new("RGB", (32, 32))
    model = mock.Mock()

    empty_state = {"scores": mock.Mock(shape=(0,))}
    with mock.patch.object(
        filter_sparse_images.detection_utils,
        "convert_rfdetr_detections_to_state",
        autospec=True,
        return_value=empty_state,
    ):
      result = filter_sparse_images.count_detections(
          image, model, self._make_config()
      )
    self.assertEqual(result, 0)
    model.predict.assert_called_once()

  def test_applies_filters_and_returns_count(self):
    """Verifies the post-filters run and the surviving count is returned."""
    image = Image.new("RGB", (32, 32))
    model = mock.Mock()

    raw_state = {"scores": mock.Mock(shape=(4,))}
    filtered_state = {"scores": mock.Mock(shape=(2,))}

    with mock.patch.object(
        filter_sparse_images.detection_utils,
        "convert_rfdetr_detections_to_state",
        autospec=True,
        return_value=raw_state,
    ), mock.patch.object(
        filter_sparse_images.detection_utils,
        "filter_contained_sub_masks",
        autospec=True,
        return_value=raw_state,
    ), mock.patch.object(
        filter_sparse_images.detection_utils,
        "merge_contained_boxes",
        autospec=True,
        return_value=filtered_state,
    ):
      result = filter_sparse_images.count_detections(
          image, model, self._make_config()
      )
    self.assertEqual(result, 2)


class BuildRfdetrModelTest(absltest.TestCase):
  """Tests for build_rfdetr_model."""

  def test_raises_when_rfdetr_unavailable(self):
    """Verifies a missing rfdetr package surfaces as ImportError."""
    with mock.patch.object(filter_sparse_images, "RFDETRSegMedium", None):
      with self.assertRaises(ImportError):
        filter_sparse_images.build_rfdetr_model("/tmp/checkpoint.pth")

  def test_builds_and_optimizes_model(self):
    """Verifies the model is constructed and optimized for inference."""
    fake_model = mock.Mock()
    fake_class = mock.Mock(return_value=fake_model)
    with mock.patch.object(filter_sparse_images, "RFDETRSegMedium", fake_class):
      result = filter_sparse_images.build_rfdetr_model("/tmp/ckpt.pth")
    fake_class.assert_called_once_with(pretrain_weights="/tmp/ckpt.pth")
    fake_model.optimize_for_inference.assert_called_once()
    self.assertIs(result, fake_model)


class ConfigureLoggingTest(absltest.TestCase):
  """Tests for configure_logging."""

  def tearDown(self):
    # Remove any handler attached to the module logger so tests stay isolated.
    module_logger = filter_sparse_images.logger
    for handler in list(module_logger.handlers):
      handler.close()
      module_logger.removeHandler(handler)
    super().tearDown()

  def test_attaches_single_handler(self):
    """Verifies exactly one handler is attached on the module logger."""
    filter_sparse_images.configure_logging()
    self.assertLen(filter_sparse_images.logger.handlers, 1)

  def test_second_call_does_not_duplicate(self):
    """Verifies a repeated call does not add a second handler."""
    filter_sparse_images.configure_logging()
    filter_sparse_images.configure_logging()
    self.assertLen(filter_sparse_images.logger.handlers, 1)


if __name__ == "__main__":
  absltest.main()
