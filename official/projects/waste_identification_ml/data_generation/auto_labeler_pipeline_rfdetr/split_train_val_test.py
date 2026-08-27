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

"""Unit tests for split_train_val.py."""

import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized

from official.projects.waste_identification_ml.data_generation.auto_labeler_pipeline_rfdetr import split_train_val


def _touch(path: pathlib.Path) -> None:
  """Creates an empty file, making parent directories as needed."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


class DiscoverDatasetDirectoriesTest(absltest.TestCase):
  """Tests for discover_dataset_directories."""

  def test_returns_sorted_dataset_tuples(self):
    """Verifies dataset subfolders are returned sorted by name."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "b_set").mkdir()
    (root / "a_set").mkdir()
    (root / "loose_file.txt").write_text("ignored")

    result = split_train_val.discover_dataset_directories(str(root))
    names = [name for name, _ in result]
    self.assertEqual(names, ["a_set", "b_set"])

  def test_raises_when_root_missing(self):
    """Verifies a missing root directory raises FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      split_train_val.discover_dataset_directories("/nonexistent/root/xyz")

  def test_raises_when_no_subdirectories(self):
    """Verifies an empty root raises ValueError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    with self.assertRaisesRegex(ValueError, "No dataset subfolders"):
      split_train_val.discover_dataset_directories(str(root))


class ValidateDatasetPathsTest(absltest.TestCase):
  """Tests for validate_dataset_paths."""

  def test_returns_source_and_output_paths(self):
    """Verifies validated tuples carry source and output folder paths."""
    root = pathlib.Path(self.create_tempdir().full_path)
    dataset_dir = root / "ds"
    (dataset_dir / "images").mkdir(parents=True)

    result = split_train_val.validate_dataset_paths(
        [("ds", str(dataset_dir))],
        input_images_folder_name="images",
        train_val_folder_name="train_val_images",
    )
    self.assertLen(result, 1)
    name, source_folder, output_folder = result[0]
    self.assertEqual(name, "ds")
    self.assertEqual(source_folder, str(dataset_dir / "images"))
    self.assertEqual(output_folder, str(dataset_dir / "train_val_images"))

  def test_raises_when_input_folder_missing(self):
    """Verifies a missing images folder raises FileNotFoundError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    dataset_dir = root / "ds"
    dataset_dir.mkdir()
    with self.assertRaises(FileNotFoundError):
      split_train_val.validate_dataset_paths(
          [("ds", str(dataset_dir))],
          input_images_folder_name="images",
          train_val_folder_name="train_val_images",
      )

  def test_raises_when_output_folder_exists(self):
    """Verifies a pre-existing output folder raises FileExistsError."""
    root = pathlib.Path(self.create_tempdir().full_path)
    dataset_dir = root / "ds"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "train_val_images").mkdir()
    with self.assertRaises(FileExistsError):
      split_train_val.validate_dataset_paths(
          [("ds", str(dataset_dir))],
          input_images_folder_name="images",
          train_val_folder_name="train_val_images",
      )


class GetSubfolderNamesTest(absltest.TestCase):
  """Tests for get_subfolder_names."""

  def test_returns_only_subfolders_sorted(self):
    """Verifies only directories are returned, sorted, ignoring files."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "z").mkdir()
    (root / "a").mkdir()
    (root / "file.jpg").write_bytes(b"")
    self.assertEqual(split_train_val.get_subfolder_names(str(root)), ["a", "z"])

  def test_returns_empty_list_when_flat(self):
    """Verifies a folder with only files yields an empty list."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "file.jpg").write_bytes(b"")
    self.assertEqual(split_train_val.get_subfolder_names(str(root)), [])

  def test_raises_when_folder_missing(self):
    """Verifies a missing folder raises FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      split_train_val.get_subfolder_names("/nonexistent/folder/abc")


class GetSortedImageNamesTest(absltest.TestCase):
  """Tests for get_sorted_image_names."""

  def test_filters_to_image_extensions(self):
    """Verifies non-image files are excluded."""
    root = pathlib.Path(self.create_tempdir().full_path)
    _touch(root / "a.jpg")
    _touch(root / "b.png")
    _touch(root / "notes.txt")
    _touch(root / "archive.zip")
    result = split_train_val.get_sorted_image_names(str(root))
    self.assertEqual(result, ["a.jpg", "b.png"])

  def test_natural_sort_order(self):
    """Verifies files are ordered naturally (img2 before img10)."""
    root = pathlib.Path(self.create_tempdir().full_path)
    for name in ["img10.jpg", "img2.jpg", "img1.jpg"]:
      _touch(root / name)
    result = split_train_val.get_sorted_image_names(str(root))
    self.assertEqual(result, ["img1.jpg", "img2.jpg", "img10.jpg"])

  def test_extension_match_is_case_insensitive(self):
    """Verifies uppercase extensions are still recognized."""
    root = pathlib.Path(self.create_tempdir().full_path)
    _touch(root / "a.JPG")
    _touch(root / "b.PNG")
    result = split_train_val.get_sorted_image_names(str(root))
    self.assertCountEqual(result, ["a.JPG", "b.PNG"])


class FilterEveryNthImageTest(parameterized.TestCase):
  """Tests for filter_every_nth_image."""

  @parameterized.named_parameters(
      ("keep_all", 1, ["a", "b", "c", "d"]),
      ("every_second", 2, ["a", "c"]),
      ("every_third", 3, ["a", "d"]),
  )
  def test_keeps_expected_indices(self, keep_every_nth, expected):
    """Verifies every Nth image starting at index 0 is kept."""
    names = ["a", "b", "c", "d"]
    result = split_train_val.filter_every_nth_image(names, keep_every_nth)
    self.assertEqual(result, expected)

  def test_empty_input_yields_empty_output(self):
    """Verifies an empty list returns an empty list."""
    self.assertEqual(split_train_val.filter_every_nth_image([], 3), [])


class CheckForDuplicatesTest(absltest.TestCase):
  """Tests for check_for_duplicates."""

  def test_passes_when_no_conflicts(self):
    """Verifies no error is raised when destination is empty."""
    destination = pathlib.Path(self.create_tempdir().full_path)
    # Should not raise.
    split_train_val.check_for_duplicates(["a.jpg", "b.jpg"], str(destination))

  def test_raises_on_existing_file(self):
    """Verifies a name colliding with an existing file raises."""
    destination = pathlib.Path(self.create_tempdir().full_path)
    _touch(destination / "a.jpg")
    with self.assertRaises(FileExistsError):
      split_train_val.check_for_duplicates(["a.jpg"], str(destination))


class CopyFilesTest(absltest.TestCase):
  """Tests for copy_files."""

  def test_copies_named_files(self):
    """Verifies each named file is copied to the destination."""
    source = pathlib.Path(self.create_tempdir().full_path)
    destination = pathlib.Path(self.create_tempdir().full_path)
    _touch(source / "a.jpg")
    _touch(source / "b.jpg")

    split_train_val.copy_files(
        ["a.jpg", "b.jpg"], str(source), str(destination)
    )

    self.assertTrue((destination / "a.jpg").exists())
    self.assertTrue((destination / "b.jpg").exists())
    # Source is left intact (copy, not move).
    self.assertTrue((source / "a.jpg").exists())


class ProcessFolderTest(absltest.TestCase):
  """Tests for process_folder."""

  def _make_populated_folder(self, count: int) -> str:
    """Creates a folder with ``count`` sequentially named JPEGs."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    for index in range(count):
      _touch(folder / f"img{index:02d}.jpg")
    return str(folder)

  def test_returns_none_when_folder_empty(self):
    """Verifies an image-less folder returns None."""
    folder = pathlib.Path(self.create_tempdir().full_path)
    train_folder = pathlib.Path(self.create_tempdir().full_path)
    val_folder = pathlib.Path(self.create_tempdir().full_path)
    result = split_train_val.process_folder(
        str(folder),
        "label",
        str(train_folder),
        str(val_folder),
        keep_every_nth=1,
        train_ratio=0.2,
    )
    self.assertIsNone(result)

  def test_val_fraction_is_taken_from_front(self):
    """Verifies train_ratio is applied as the VAL fraction (front slice)."""
    # 10 images, keep_every_nth=1 -> 10 kept. train_ratio=0.2 -> 2 val, 8 train.
    folder = self._make_populated_folder(10)
    train_folder = pathlib.Path(self.create_tempdir().full_path)
    val_folder = pathlib.Path(self.create_tempdir().full_path)

    _, _, train_names, val_names = split_train_val.process_folder(
        folder,
        "label",
        str(train_folder),
        str(val_folder),
        keep_every_nth=1,
        train_ratio=0.2,
    )
    self.assertLen(val_names, 2)
    self.assertLen(train_names, 8)
    # Val is the leading slice; train is the remainder. No overlap.
    self.assertEqual(set(val_names) & set(train_names), set())

  def test_keep_every_nth_applied_before_split(self):
    """Verifies subsampling happens before the train/val split."""
    # 9 images, keep_every_nth=3 -> 3 kept. train_ratio=0.0 -> all train.
    folder = self._make_populated_folder(9)
    train_folder = pathlib.Path(self.create_tempdir().full_path)
    val_folder = pathlib.Path(self.create_tempdir().full_path)

    _, _, train_names, val_names = split_train_val.process_folder(
        folder,
        "label",
        str(train_folder),
        str(val_folder),
        keep_every_nth=3,
        train_ratio=0.0,
    )
    self.assertLen(train_names, 3)
    self.assertEmpty(val_names)


class ProcessDatasetTest(absltest.TestCase):
  """Tests for process_dataset (flat and nested layouts)."""

  def test_flat_source_folder(self):
    """Verifies a flat source folder splits into train and val."""
    dataset_root = pathlib.Path(self.create_tempdir().full_path)
    source = dataset_root / "images"
    for index in range(10):
      _touch(source / f"img{index:02d}.jpg")
    output = dataset_root / "train_val_images"

    train_count, val_count = split_train_val.process_dataset(
        dataset_name="ds",
        source_folder=str(source),
        output_folder=str(output),
        train_split_name="train",
        val_split_name="val",
        keep_every_nth=1,
        train_ratio=0.2,
    )
    self.assertEqual(val_count, 2)
    self.assertEqual(train_count, 8)
    self.assertLen(os.listdir(output / "train"), 8)
    self.assertLen(os.listdir(output / "val"), 2)

  def test_nested_subfolders_are_flattened(self):
    """Verifies images from subfolders land flat in the shared splits."""
    dataset_root = pathlib.Path(self.create_tempdir().full_path)
    source = dataset_root / "images"
    for sub in ["group_a", "group_b"]:
      for index in range(5):
        _touch(source / sub / f"{sub}_img{index}.jpg")
    output = dataset_root / "train_val_images"

    train_count, val_count = split_train_val.process_dataset(
        dataset_name="ds",
        source_folder=str(source),
        output_folder=str(output),
        train_split_name="train",
        val_split_name="val",
        keep_every_nth=1,
        train_ratio=0.0,
    )
    # 10 images total across both subfolders, all to train.
    self.assertEqual(train_count, 10)
    self.assertEqual(val_count, 0)
    self.assertLen(os.listdir(output / "train"), 10)

  def test_raises_when_no_images_found(self):
    """Verifies an image-less dataset raises ValueError."""
    dataset_root = pathlib.Path(self.create_tempdir().full_path)
    source = dataset_root / "images"
    source.mkdir(parents=True)
    output = dataset_root / "train_val_images"
    with self.assertRaisesRegex(ValueError, "No images found"):
      split_train_val.process_dataset(
          dataset_name="ds",
          source_folder=str(source),
          output_folder=str(output),
          train_split_name="train",
          val_split_name="val",
          keep_every_nth=1,
          train_ratio=0.2,
      )


if __name__ == "__main__":
  absltest.main()
