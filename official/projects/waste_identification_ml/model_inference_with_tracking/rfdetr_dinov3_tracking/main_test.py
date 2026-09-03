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

"""Unit tests for main.py.

The pipeline's model-loading __init__ is bypassed with __new__ so the
filesystem discovery helpers can be tested without constructing RFDETR or
DINOv3. The per-frame orchestration is out of scope for these focused tests.
"""

import pathlib
from unittest import mock

from absl.testing import absltest

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import main


def _touch(path: pathlib.Path) -> None:
  """Creates an empty file, making parent directories as needed."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


def _make_pipeline(
    image_file_extensions: list[str],
) -> main.PETBottlePipeline:
  """Builds a PETBottlePipeline via __new__ with only the config it needs."""
  pipeline = object.__new__(main.PETBottlePipeline)
  pipeline._config = mock.Mock()
  pipeline._config.models.rfdetr.image_file_extensions = image_file_extensions
  return pipeline


class DiscoverSubfoldersTest(absltest.TestCase):
  """Tests for _discover_subfolders."""

  def test_returns_sorted_immediate_subfolders(self):
    """Verifies only immediate child directories are returned, sorted."""
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "session_b").mkdir()
    (root / "session_a").mkdir()
    _touch(root / "loose.png")

    pipeline = _make_pipeline(["*.png"])
    result = pipeline._discover_subfolders(str(root))
    names = [pathlib.Path(path).name for path in result]
    self.assertEqual(names, ["session_a", "session_b"])

  def test_missing_directory_returns_empty(self):
    """Verifies a non-existent input directory returns an empty list."""
    pipeline = _make_pipeline(["*.png"])
    self.assertEqual(pipeline._discover_subfolders("/no/such/dir"), [])


class CollectImagePathsTest(absltest.TestCase):
  """Tests for _collect_image_paths."""

  def test_collects_only_configured_extensions_sorted(self):
    """Verifies only files matching configured globs are returned, sorted."""
    directory = pathlib.Path(self.create_tempdir().full_path)
    _touch(directory / "a.png")
    _touch(directory / "b.jpg")
    _touch(directory / "c.txt")
    _touch(directory / "d.png")

    pipeline = _make_pipeline(["*.png", "*.jpg"])
    result = pipeline._collect_image_paths(str(directory))
    names = [pathlib.Path(path).name for path in result]
    self.assertEqual(names, ["a.png", "b.jpg", "d.png"])

  def test_returns_empty_when_no_matches(self):
    """Verifies a directory with no matching files yields an empty list."""
    directory = pathlib.Path(self.create_tempdir().full_path)
    _touch(directory / "notes.txt")

    pipeline = _make_pipeline(["*.png"])
    self.assertEqual(pipeline._collect_image_paths(str(directory)), [])


if __name__ == "__main__":
  absltest.main()
